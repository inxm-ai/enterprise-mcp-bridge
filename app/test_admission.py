import subprocess
import sys
import textwrap
import time
from pathlib import Path

import anyio
import pytest
from fastapi import HTTPException

from app import admission

MIB = admission.MIB


@pytest.fixture
def cgroup(monkeypatch, tmp_path):
    """A memory-limited cgroup the test sets the usage of."""
    monkeypatch.setattr(admission, "CGROUP", tmp_path)
    monkeypatch.setattr(admission, "POLL_SECONDS", 0.01)
    monkeypatch.setenv("MCP_ADMISSION_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("MCP_CHILD_MEMORY_RESERVE_MB", "400")
    monkeypatch.setenv("MCP_CHILD_ADMISSION_TIMEOUT", "2")

    def use(limit_mib, used_mib, inactive_file_mib=0):
        (tmp_path / "memory.max").write_text(f"{limit_mib * MIB}\n")
        (tmp_path / "memory.current").write_text(
            f"{(used_mib + inactive_file_mib) * MIB}\n"
        )
        (tmp_path / "memory.stat").write_text(
            f"anon 1\ninactive_file {inactive_file_mib * MIB}\nactive_file 0\n"
        )

    use(1000, 0)
    return use


def counts():
    """(starting, all) children as admission sees them."""
    directory = admission.admission_dir()
    return admission._live_slots(directory) if directory.exists() else (0, 0)


def test_droppable_page_cache_does_not_count_as_used(cgroup):
    cgroup(1000, 100, inactive_file_mib=500)
    assert admission.memory() == (1000 * MIB, 100 * MIB)


def test_without_a_memory_limit_nothing_is_measured(cgroup, tmp_path):
    (tmp_path / "memory.max").write_text("max\n")
    assert admission.memory() is None


@pytest.mark.asyncio
async def test_disabled_admission_starts_children_at_once(cgroup, monkeypatch):
    monkeypatch.delenv("MCP_CHILD_MEMORY_RESERVE_MB")
    cgroup(1000, 1000)
    async with admission.child_slot("a"):
        async with admission.child_slot("b"):
            assert counts() == (0, 0), "nothing is tracked"


@pytest.mark.asyncio
async def test_a_lone_child_always_starts_even_without_room(cgroup):
    cgroup(1000, 950)
    async with admission.child_slot("a") as ready:
        ready()
        assert counts() == (0, 1)
    assert counts() == (0, 0)


@pytest.mark.asyncio
async def test_children_still_starting_keep_their_reservation(cgroup):
    """Memory of a child that is still starting is not in the cgroup yet."""
    order = []

    async def start(name, release: anyio.Event):
        async with admission.child_slot(name) as ready:
            order.append(name)
            await release.wait()
            ready()
            await anyio.sleep(1)

    first, second, third = anyio.Event(), anyio.Event(), anyio.Event()
    async with anyio.create_task_group() as tg:
        tg.start_soon(start, "first", first)
        await anyio.sleep(0.05)
        tg.start_soon(start, "second", second)
        await anyio.sleep(0.05)
        tg.start_soon(start, "third", third)
        await anyio.sleep(0.1)
        # 0 used + 400 MiB for each of three starting children > 1000 MiB.
        assert order == ["first", "second"]
        first.set()
        await anyio.sleep(0.1)
        # first initialized (its memory would now show; still 0 here).
        assert order == ["first", "second", "third"]
        second.set()
        third.set()
    assert counts() == (0, 0)


@pytest.mark.asyncio
async def test_a_full_host_refuses_with_retry_after(cgroup):
    cgroup(1000, 800)
    async with admission.child_slot("a") as ready:
        ready()
        with pytest.raises(HTTPException) as refused:
            async with admission.child_slot("b"):
                pass
    assert refused.value.status_code == 503
    assert refused.value.headers == {
        "Retry-After": "5",
        "X-MCP-Admission": "refused",
    }
    assert counts() == (0, 0)


@pytest.mark.asyncio
async def test_a_waiting_child_starts_once_memory_frees(cgroup):
    cgroup(1000, 800)
    started = anyio.Event()

    async def waiting():
        async with admission.child_slot("b"):
            started.set()

    async with admission.child_slot("a") as ready:
        ready()
        async with anyio.create_task_group() as tg:
            tg.start_soon(waiting)
            await anyio.sleep(0.1)
            assert not started.is_set()
            cgroup(1000, 300)
            with anyio.fail_after(1):
                await started.wait()


@pytest.mark.asyncio
async def test_a_child_that_fails_to_start_releases_its_slot(cgroup):
    with pytest.raises(RuntimeError):
        async with admission.child_slot("a"):
            raise RuntimeError("spawn failed")
    assert counts() == (0, 0)


def test_children_of_other_worker_processes_count_and_dead_ones_do_not(cgroup):
    """Workers share the container's memory, so they share the slots."""
    holder = textwrap.dedent("""
        import asyncio, sys
        from app import admission

        async def main():
            async with admission.child_slot("other-worker"):
                print("admitted", flush=True)
                sys.stdin.readline()

        asyncio.run(main())
        """)
    worker = subprocess.Popen(
        [sys.executable, "-c", holder],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    try:
        assert worker.stdout.readline().strip() == "admitted"
        assert counts() == (1, 1), "the other worker's starting child counts"
        # 0 used + 400 MiB for its starting child + 400 for a new one <= 1000.
        assert admission._try_admit(400 * MIB) is not None
    finally:
        worker.kill()
        worker.wait()
    time.sleep(0.05)
    slots = counts()
    assert slots[1] == 1, f"only this process's slot is left: {slots}"
