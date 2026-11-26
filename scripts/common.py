"""Shared helpers for the driver scripts: logging, size parsing, CLI flag
building, binary checks, problem loading, the priority semaphore and `fan_out`,
subprocess-JSON plumbing, and running a single `attempt` process."""

import asyncio
import contextlib
import heapq
import itertools
import json
import re
import sys
from collections import deque
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

from schemes import ATTEMPT_DTYPES

# Release Rust binaries driven by the scripts.
START_BIN = Path("target/release/start")
SAMPLE_BIN = Path("target/release/sample")
ATTEMPT_BIN = Path("target/release/attempt")

# Machine-readable eqsat progress events (`EVENT_PREFIX` in src/eqsat.rs),
# emitted under `--print-success-iters`.
EQSAT_ITER_RE = re.compile(r"^@EQSAT iter=(\d+)$", re.MULTILINE)

# Rust's exit code for a panic
PANIC_RETURNCODE = 101

# Linux's `MAX_ARG_STRLEN`: the most bytes, including the trailing NUL, that a
# single argv entry may have, else `exec` fails with `E2BIG`.
MAX_ARG_STRLEN = 32 * 4096

# Where a Rust panic message header
# `thread 'main' (609420) panicked at src/langs/math/mod.rs:97:13:`.
PANIC_HEADER_RE = re.compile(r"^thread '.*' .*panicked at ", re.MULTILINE)


# Where `log` writes; `experiment.py` sets it per run.
LOG_FILE: ContextVar[TextIO | None] = ContextVar("LOG_FILE", default=None)


def log(message: str) -> None:
    """Write `message` to the current run's log file, or above the progress bars on stderr."""
    tqdm.write(message, file=LOG_FILE.get() or sys.stderr)


def parse_size(s: str) -> int:
    """Parse a human byte size like `4G` into bytes."""
    s = s.strip().upper()
    mult = 1
    for suf, m in (("K", 1024), ("M", 1024**2), ("G", 1024**3), ("T", 1024**4)):
        if s.endswith(suf):
            mult = m
            s = s[:-1]
            break
    return int(float(s) * mult)


def cli_flags(**values: object) -> list[str]:
    """Turn `snake_case=value` pairs into the Rust binaries' `--kebab-case value` arguments.

    `True` becomes a bare `--flag`, and `False` and `None` are dropped.
    """
    flags: list[str] = []

    for key, value in values.items():
        flag = key.replace("_", "-")
        if value is None or value is False:
            continue
        flags.append(f"--{flag}")
        if value is not True:
            flags.append(str(value))

    return flags


def with_flag(cmd: list[str], flag: str, value: str) -> list[str]:
    """Copy of `cmd` with `flag` set to `value`, replacing an existing value."""
    try:
        i = cmd.index(flag)
        return [*cmd[: i + 1], value, *cmd[i + 2 :]]
    except ValueError:
        return [*cmd, flag, value]


def exit_if_missing(*binaries: Path) -> None:
    """Print an error and exit 2 if any binary is missing."""
    missing = [b for b in binaries if not b.exists()]
    if missing:
        names = " ".join(f"--bin {b.name}" for b in missing)
        log(
            f"Binary not found: {', '.join(str(b) for b in missing)}. "
            f"Build with `cargo build --release {names}`."
        )
        raise SystemExit(2)


# TODO: the `smallest_novel`/`smallest_overall` policies are gone for now
class SamplePolicy(StrEnum):
    Count = "count"
    Uniform = "uniform"


@dataclass(frozen=True)
class Pair:
    """One start/goal problem."""

    start: str
    goal: str


def load_pairs(path: Path, n_starts: int | None = None) -> list[Pair]:
    """Load `problems.json`'s rows as start/goal pairs.

    Keeps the first `n_starts` start terms in sorted order, and all goals per
    start term in file order; all starts if `n_starts` is omitted.
    """
    rows = json.loads((path / "problems.json").read_text())
    goals: dict[str, list[str]] = {}
    for row in rows:
        goals.setdefault(row["start"], []).append(row["goal"])
    return [Pair(start, goal) for start in sorted(goals)[:n_starts] for goal in goals[start]]


def problem_language(path: Path) -> str:
    """The language the problems in `path` were generated in."""
    return str(json.loads((path / "problem_args.json").read_text())["language"])


class PrioritySemaphore:
    """A semaphore that hands a freed physical CPU core to the waiter of the
    lowest priority, first come first served among equals.
    """

    def __init__(self, cpus: Sequence[int], cgroup: Path) -> None:
        if not cpus:
            raise ValueError("cpus must be non-empty")
        self.cpus = tuple(cpus)
        self.cgroup = cgroup
        self._free: deque[int] = deque(self.cpus)
        # A cancelled waiter stays in the heap until `release` skips it.
        self._waiters: list[tuple[int, int, asyncio.Future[int]]] = []
        self._arrivals = 0

    def at(self, priority: int) -> PrioritySlot:
        """This semaphore as seen by the waiters of `priority`."""
        return PrioritySlot(self, priority)

    async def acquire(self, priority: int) -> int:
        # A free core means no one is waiting, since `release` only frees one then.
        if self._free:
            return self._free.popleft()
        waiter: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._arrivals += 1
        heapq.heappush(self._waiters, (priority, self._arrivals, waiter))
        try:
            return await waiter
        except asyncio.CancelledError:
            # Handed a core just before it was cancelled, so pass it on.
            if not waiter.cancelled():
                self.release(waiter.result())
            raise

    def release(self, cpu: int) -> None:
        while self._waiters:
            _, _, waiter = heapq.heappop(self._waiters)
            if not waiter.done():
                waiter.set_result(cpu)
                return
        self._free.append(cpu)


@dataclass(frozen=True)
class PrioritySlot:
    """One priority's view of a `PrioritySemaphore`, used like a semaphore."""

    semaphore: PrioritySemaphore
    priority: int

    @property
    def cgroup(self) -> Path:
        return self.semaphore.cgroup

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncGenerator[int]:
        cpu = await self.semaphore.acquire(self.priority)
        try:
            yield cpu
        finally:
            self.semaphore.release(cpu)


async def fan_out(
    fn: Callable[[Any], Coroutine[Any, Any, Any]],
    items: Sequence[Any],
    desc: str,
    unit: str = "job",
    leave: bool = True,
) -> list:
    """Run `fn` over all `items` at once, dropping the ones that returned None.

    Only the subprocesses are limited, through `run_json_subprocess`'s `limit`.
    Results come back in `items` order. Without `leave`, the bar is cleared once done.
    """
    with tqdm(
        total=len(items),
        desc=desc,
        unit=unit,
        bar_format="{l_bar}{bar:30}{r_bar}",
        dynamic_ncols=True,
        leave=leave,
    ) as bar:
        # A task group rather than `gather`, so the first failure cancels the
        # items still waiting for a slot instead of letting them start more
        # processes on the way down.
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(fn(item)) for item in items]
            for task in tasks:
                task.add_done_callback(lambda _: bar.update(1))

    return [result for task in tasks if (result := task.result()) is not None]


@dataclass(frozen=True)
class Measured:
    payload: Any
    peak_rss: int
    # Kernel-accounted cgroup CPU time (`cpu.stat` `usage_usec`), in seconds.
    wall_time: float


class ArgTooLong(ValueError):
    """An argv entry exceeds the kernel's per-argument limit."""

    def __init__(self, what: str, flag: str, size: int) -> None:
        super().__init__(
            f"{what}: argument of {flag} is {size} bytes, "
            f"over the kernel's per-argument limit of {MAX_ARG_STRLEN - 1}"
        )


class MemoryKilled(RuntimeError):
    """A capped child was SIGKILLed by its cgroup memory limit."""

    def __init__(self, what: str, stderr: str, wall_time: float) -> None:
        super().__init__(f"{what} was killed at its RSS cap")
        self.stderr = stderr
        self.productive_iters = [int(m) for m in EQSAT_ITER_RE.findall(stderr)]
        self.wall_time = wall_time


class TimeoutKilled(RuntimeError):
    """A child that outran its hard timeout and was killed by hand."""

    def __init__(self, what: str, timeout: float, wall_time: float) -> None:
        super().__init__(f"{what} was killed at its hard timeout of {timeout:g}s")
        self.what = what
        self.wall_time = wall_time


class BinaryPanicked(RuntimeError):
    """A child exited with a Rust panic that it did not catch itself."""

    def __init__(self, what: str, stderr: str, wall_time: float) -> None:
        super().__init__(f"{what} panicked (code {PANIC_RETURNCODE}):\n{stderr}")
        self.what = what
        self.stderr = stderr
        self.wall_time = wall_time

    @property
    def panic_message(self) -> str:
        """The panic's own lines from stderr, without the progress output before
        it or the `RUST_BACKTRACE` hint after it."""
        header = PANIC_HEADER_RE.search(self.stderr)
        tail = self.stderr[header.start() :] if header else self.stderr
        return "\n".join(
            line for line in tail.strip().splitlines() if not line.startswith("note: run with")
        )

    def warn(self) -> None:
        """Log the panic."""
        message = "\n".join(f"  {line}" for line in self.panic_message.splitlines())
        log(f"WARNING: {self.what} panicked (code {PANIC_RETURNCODE}):\n{message}")


def check_arg_sizes(cmd: list[str], what: str) -> None:
    """Raise `ArgTooLong` if any entry of `cmd` would not fit through `exec`."""
    for i, arg in enumerate(cmd):
        size = len(arg.encode())
        if size >= MAX_ARG_STRLEN:
            flag = cmd[i - 1] if i > 0 and cmd[i - 1].startswith("--") else f"argv[{i}]"
            raise ArgTooLong(what, flag, size)


_CGROUP_IDS = itertools.count(1)


def prefix_cgroup(argv: list[str], procs_path: Path) -> list[str]:
    """Move the spawned process into `procs_path` before execing `argv`."""
    return ["/bin/sh", "-c", 'echo $$ > "$1" && shift && exec "$@"', "_", str(procs_path), *argv]


@contextlib.contextmanager
def _job_cgroup(root: Path, max_rss_bytes: int | None, cpu: int) -> Generator[Path]:
    """Create a per-child cgroup under delegated `root` for RSS cap and CPU pinning."""
    job_dir = root / f"job-{next(_CGROUP_IDS)}"
    job_dir.mkdir()
    try:
        if max_rss_bytes is not None:
            (job_dir / "memory.max").write_text(f"{max_rss_bytes}\n")
            if (swap := job_dir / "memory.swap.max").exists():
                swap.write_text("0\n")
            if (oom_group := job_dir / "memory.oom.group").exists():
                oom_group.write_text("1\n")
        if (cpuset := job_dir / "cpuset.cpus").exists():
            cpuset.write_text(f"{cpu}\n")
        yield job_dir
    finally:
        if (kill := job_dir / "cgroup.kill").exists():
            with contextlib.suppress(OSError):
                kill.write_text("1\n")
        with contextlib.suppress(OSError):
            job_dir.rmdir()


def _cgroup_cpu_time(job_dir: Path) -> float:
    """Kernel-accounted CPU time (`cpu.stat` `usage_usec`) for `job_dir`, in seconds."""
    for line in (job_dir / "cpu.stat").read_text().splitlines():
        key, _, val = line.partition(" ")
        if key == "usage_usec":
            return int(val) / 1_000_000
    raise RuntimeError(f"no usage_usec in {job_dir / 'cpu.stat'}")


def _cgroup_oom_killed(job_dir: Path) -> bool:
    """Whether the kernel's cgroup OOM killer killed a process in `job_dir`."""
    try:
        for line in (job_dir / "memory.events").read_text().splitlines():
            key, _, val = line.partition(" ")
            if key == "oom_kill":
                return int(val) > 0
    except OSError, ValueError:
        pass
    return False


async def run_json_subprocess(
    cmd: list[str],
    *,
    what: str,
    limit: PrioritySlot,
    max_rss_bytes: int | None = None,
    timeout: float | None = None,
    stdin: bytes | None = None,
) -> Measured:
    """Run a JSON child and return its payload plus its peak RSS.

    The child prints a `Measured` envelope (`src/cli.rs`): the payload it would
    otherwise print, under a `peak_rss` the binary reads from its own
    `VmHWM` just before serializing.

    The child only spawns once it holds a core from `limit`, and runs in its own
    sub-cgroup inside the delegated `bench.slice` tree, enforcing `memory.max`
    and per-core `cpuset.cpus`. Its elapsed time is read from `cpu.stat`
    (`usage_usec`), isolating timing from asyncio event-loop stalls.

    Raises `ArgTooLong` before spawning when an argv entry exceeds the kernel's
    per-argument limit, `MemoryKilled` when `max_rss_bytes` is set and the cap
    killed it, `TimeoutKilled` when `timeout` is set and the child outran it,
    and `BinaryPanicked` when the child panicked.
    """
    check_arg_sizes(cmd, what)
    async with limit.hold() as cpu:
        with _job_cgroup(limit.cgroup, max_rss_bytes, cpu) as job_dir:
            proc = await asyncio.create_subprocess_exec(
                *prefix_cgroup(cmd, job_dir / "cgroup.procs"),
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                async with asyncio.timeout(timeout):
                    out, err = await proc.communicate(stdin)
            except (asyncio.CancelledError, TimeoutError) as stopped:
                # Cancelling or timing out only unwinds the *read*, so the child has to be killed by hand.
                # `prefix_cgroup` execs the workload in place rather than forking.
                if proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    # An already-cancelled caller is handed a second `CancelledError`
                    # The child watcher reaps the child either way.
                    with contextlib.suppress(asyncio.CancelledError):
                        await proc.wait()
                if isinstance(stopped, TimeoutError):
                    assert timeout is not None
                    raise TimeoutKilled(what, timeout, _cgroup_cpu_time(job_dir)) from None
                raise
            wall_time = _cgroup_cpu_time(job_dir)
            oom_killed = max_rss_bytes is not None and _cgroup_oom_killed(job_dir)

    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")

    if oom_killed:
        raise MemoryKilled(what, stderr, wall_time)
    if proc.returncode == PANIC_RETURNCODE:
        raise BinaryPanicked(what, stderr, wall_time)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed (code {proc.returncode}):\n{stderr}")
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{what} returned non-JSON stdout: {e}\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        ) from e
    return Measured(envelope["payload"], int(envelope["peak_rss"]), wall_time)


@dataclass(frozen=True)
class AttemptResult:
    summary: dict
    peak_rss: int | None
    wall_time: float


def attempt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """`attempt`'s `AttemptSummary` payload (`src/bin/attempt.rs`) as an attempt row.

    The payload already carries exactly the `ATTEMPT_DTYPES` columns; this only
    checks that the two still agree.
    """
    if payload.keys() != ATTEMPT_DTYPES.keys():
        raise ValueError(
            f"attempt payload keys {sorted(payload)} differ from {sorted(ATTEMPT_DTYPES)}"
        )
    return payload


def failure_summary(stop_reason: str) -> dict[str, Any]:
    """An `attempt_summary`-shaped row with all measurements `None`, for a child
    that never printed one (`out_of_memory`, `binary_panic`, `arg_too_long`, `timeout`,
    `budget_exhausted`).

    `binary_panic` is an uncaught panic, unlike eqsat's `panic`; `timeout` is the
    hard timeout, unlike egg's `TimeLimit`; `budget_exhausted` is the hard timeout
    set by the guided search's pair budget.
    """
    empty = dict.fromkeys(ATTEMPT_DTYPES)
    panic = stop_reason == "binary_panic"
    return {**empty, "reached": False, "panic": panic, "stop_reason": stop_reason}


async def measure_attempt(
    cmd: list[str],
    *,
    what: str,
    max_rss_bytes: int,
    limit: PrioritySlot,
    timeout: float | None = None,
    timeout_reason: str = "timeout",
    stdin: bytes | None = None,
) -> AttemptResult:
    """Run one `attempt` process under the RSS cap and the hard timeout.

    An attempt killed at the cap, by an uncaught panic, at its hard timeout, or
    never spawned since an argument exceeds the kernel's limit comes back as a
    failed attempt with ``stop_reason="out_of_memory"``/``"binary_panic"``/
    `timeout_reason`/``"arg_too_long"`` rather than an exception.
    """
    try:
        measured = await run_json_subprocess(
            cmd,
            what=what,
            max_rss_bytes=max_rss_bytes,
            limit=limit,
            timeout=timeout,
            stdin=stdin,
        )
    except ArgTooLong as too_long:
        log(f"WARNING: {too_long}")
        return AttemptResult(failure_summary("arg_too_long"), None, 0.0)
    except MemoryKilled as killed:
        return AttemptResult(failure_summary("out_of_memory"), None, killed.wall_time)
    except TimeoutKilled as timed_out:
        log(f"WARNING: {timed_out}")
        return AttemptResult(failure_summary(timeout_reason), None, timed_out.wall_time)
    except BinaryPanicked as panicked:
        panicked.warn()
        return AttemptResult(failure_summary("binary_panic"), None, panicked.wall_time)
    return AttemptResult(attempt_summary(measured.payload), measured.peak_rss, measured.wall_time)
