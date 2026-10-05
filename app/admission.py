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
"""

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

_lock = threading.Lock()
# Children admitted and not yet initialized: their memory is still to come.
_starting = 0
# Children admitted and not yet exited.
_running = 0


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


def _admit(reserve: int) -> bool:
    # One child may always run: a lone request never waits on itself.
    if _running == 0:
        return True
    mem = memory()
    if mem is None:
        return True
    limit, used = mem
    return used + reserve * (_starting + 1) <= limit


@asynccontextmanager
async def child_slot(server: str) -> AsyncIterator[Callable[[], None]]:
    """Hold a slot for one child for the duration of the block.

    Yields a callback to call once the child has initialized: from then on
    its memory shows in the cgroup and its reservation is released.
    """
    global _starting, _running
    reserve = reserve_bytes()
    if reserve <= 0:
        yield lambda: None
        return

    timeout = admission_timeout()
    deadline = time.monotonic() + timeout
    waited = False
    while True:
        with _lock:
            if _admit(reserve):
                _starting += 1
                _running += 1
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

    initialized = False

    def ready() -> None:
        global _starting
        nonlocal initialized
        if not initialized:
            initialized = True
            with _lock:
                _starting -= 1

    try:
        yield ready
    finally:
        with _lock:
            if not initialized:
                _starting -= 1
            _running -= 1
