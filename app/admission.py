"""Admission control for local MCP server processes.

Every request to a local (command) server starts its own child process. One
bridge serving many local servers (the shared remote MCP host) can be asked
for all of them at once, e.g. by a client listing every server's tools, and
dozens of children starting together push the pod past its memory limit.

With MCP_CHILD_MEMORY_RESERVE_MB set, a child only starts while the pod's
memory leaves that much room for it and for every child still starting
(whose memory the cgroup does not show yet). Otherwise the request waits for
room, up to MCP_CHILD_ADMISSION_TIMEOUT seconds, and then fails with 503 and
Retry-After instead of taking the pod down. Unset, or outside a memory
limited cgroup, every child starts at once as before.

The memory limit is the container's, shared by every worker process, so the
children are counted across processes: each admitted child is a slot file
in MCP_ADMISSION_DIR, locked (flock) by the process that owns it. A slot
whose lock can be taken belongs to a process that died and is dropped.
"""

import errno
import fcntl
import itertools
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import anyio
from fastapi import HTTPException

logger = logging.getLogger("uvicorn.error")

CGROUP = Path("/sys/fs/cgroup")
MIB = 1024 * 1024
# Marks a 503 as an admission refusal: the server never ran, so a client may
# retry even a tool call (a plain 503 could come after the tool ran).
ADMISSION_HEADER = "X-MCP-Admission"
POLL_SECONDS = 0.2
# Slot file prefixes: admitted and still initializing, or initialized.
STARTING = "s-"
RUNNING = "r-"

_thread_lock = threading.Lock()
_sequence = itertools.count()


def admission_dir() -> Path:
    return Path(os.environ.get("MCP_ADMISSION_DIR", "/tmp/mcp-admission"))


def reserve_bytes() -> int:
    try:
        return max(0, int(os.environ.get("MCP_CHILD_MEMORY_RESERVE_MB", "0"))) * MIB
    except ValueError:
        return 0


def admission_timeout() -> float:
    try:
        return max(0.0, float(os.environ.get("MCP_CHILD_ADMISSION_TIMEOUT", "30")))
    except ValueError:
        return 30.0


def memory() -> Optional[tuple[int, int]]:
    """(limit, in use) of this cgroup, or None without a memory limit.

    In use is the working set, as the kubelet counts it: page cache the
    kernel can drop (inactive_file) does not crowd out a child.
    """
    try:
        limit = (CGROUP / "memory.max").read_text().strip()
        if limit == "max":
            return None
        current = int((CGROUP / "memory.current").read_text())
        inactive_file = 0
        for line in (CGROUP / "memory.stat").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "inactive_file":
                inactive_file = int(value)
                break
        return int(limit), max(0, current - inactive_file)
    except (OSError, ValueError):
        return None


def _live_slots(directory: Path) -> tuple[int, int]:
    """(starting, all) children across processes; drops dead owners' slots.

    Called with the directory lock held.
    """
    starting = total = 0
    for slot in directory.iterdir():
        if not slot.name.startswith((STARTING, RUNNING)):
            continue
        try:
            fd = os.open(slot, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            total += 1
            starting += slot.name.startswith(STARTING)
        else:
            slot.unlink(missing_ok=True)  # its owner died
        finally:
            os.close(fd)
    return starting, total


def _admit(reserve: int, starting: int, total: int) -> bool:
    # One child may always run: a lone request never waits on itself.
    if total == 0:
        return True
    mem = memory()
    if mem is None:
        return True
    limit, used = mem
    return used + reserve * (starting + 1) <= limit


class _Slot:
    """One admitted child: a slot file this process holds locked."""

    def __init__(self, directory: Path):
        self.path = directory / f"{STARTING}{os.getpid()}-{next(_sequence)}"
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def ready(self) -> None:
        """The child initialized: its memory now shows in the cgroup."""
        if self.path.name.startswith(STARTING):
            running = self.path.with_name(RUNNING + self.path.name[len(STARTING) :])
            os.rename(self.path, running)
            self.path = running

    def release(self) -> None:
        self.path.unlink(missing_ok=True)
        os.close(self.fd)


def _try_admit(reserve: int) -> Optional[_Slot]:
    directory = admission_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _thread_lock:
        lock_fd = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            starting, total = _live_slots(directory)
            if not _admit(reserve, starting, total):
                return None
            return _Slot(directory)
        finally:
            os.close(lock_fd)


@asynccontextmanager
async def child_slot(server: str) -> AsyncIterator[Callable[[], None]]:
    """Hold a slot for one child for the duration of the block.

    Yields a callback to call once the child has initialized: from then on
    its memory shows in the cgroup and its reservation is released.
    """
    reserve = reserve_bytes()
    if reserve <= 0:
        yield lambda: None
        return

    timeout = admission_timeout()
    deadline = time.monotonic() + timeout
    waited = False
    while True:
        slot = _try_admit(reserve)
        if slot is not None:
            break
        if time.monotonic() >= deadline:
            logger.warning(
                "[Admission] No memory to start %s after %.0fs; refusing",
                server,
                timeout,
            )
            raise HTTPException(
                status_code=503,
                detail="The MCP host is at capacity; retry shortly.",
                headers={"Retry-After": "5", ADMISSION_HEADER: "refused"},
            )
        if not waited:
            logger.info("[Admission] Waiting for memory to start %s", server)
            waited = True
        await anyio.sleep(POLL_SECONDS)

    try:
        yield slot.ready
    finally:
        slot.release()
