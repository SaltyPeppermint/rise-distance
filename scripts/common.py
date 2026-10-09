"""Shared helpers for the driver scripts: size parsing, subprocess-JSON
plumbing, problem loading, binary checks, the `attempt` payload schema, CLI flag
building, and the check that a stored baseline fits a run."""

import asyncio
import contextlib
import heapq
import json
import re
import sys
import time
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TextIO

import polars as pl
from tqdm import tqdm

from schemes import ATTEMPT_DTYPES


# TODO: the `smallest_novel`/`smallest_overall` policies are gone for now
class SamplePolicy(StrEnum):
    Count = "count"
    Uniform = "uniform"


# Where `log` writes; `experiment.py` sets it per run.
LOG_FILE: ContextVar[TextIO | None] = ContextVar("LOG_FILE", default=None)


def log(message: str) -> None:
    """Write `message` to the current run's log file, or above the progress bars on stderr."""
    tqdm.write(message, file=LOG_FILE.get() or sys.stderr)


class PrioritySemaphore:
    """A semaphore that hands a freed slot to the waiter of the lowest priority,
    first come first served among equals."""

    def __init__(self, value: int) -> None:
        self._free = value
        # A cancelled waiter stays in the heap until `release` skips it.
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._arrivals = 0

    def at(self, priority: int) -> PrioritySlot:
        """This semaphore as seen by the waiters of `priority`."""
        return PrioritySlot(self, priority)

    async def acquire(self, priority: int) -> None:
        # A free slot means no one is waiting, since `release` only frees one then.
        if self._free > 0:
            self._free -= 1
            return
        waiter = asyncio.get_running_loop().create_future()
        self._arrivals += 1
        heapq.heappush(self._waiters, (priority, self._arrivals, waiter))
        try:
            await waiter
        except asyncio.CancelledError:
            # Handed a slot just before it was cancelled, so pass that slot on.
            if not waiter.cancelled():
                self.release()
            raise

    def release(self) -> None:
        while self._waiters:
            _, _, waiter = heapq.heappop(self._waiters)
            if not waiter.done():
                waiter.set_result(None)
                return
        self._free += 1


@dataclass(frozen=True)
class PrioritySlot:
    """One priority's view of a `PrioritySemaphore`, used like a semaphore."""

    semaphore: PrioritySemaphore
    priority: int

    async def __aenter__(self) -> None:
        await self.semaphore.acquire(self.priority)

    async def __aexit__(self, *exc: object) -> None:
        self.semaphore.release()


@dataclass(frozen=True)
class Pair:
    """One start/goal problem."""

    start: str
    goal: str


def load_pairs(path: Path, n_starts: int | None = None, n_goals: int | None = None) -> list[Pair]:
    """Load `problems.json`'s rows as start/goal pairs.

    Keeps the first `n_starts` start terms in sorted order and the first
    `n_goals` goals per start term in file order; all of them if omitted.
    """
    rows = json.loads((path / "problems.json").read_text())
    goals: dict[str, list[str]] = {}
    for row in rows:
        goals.setdefault(row["start"], []).append(row["goal"])
    return [
        Pair(start, goal) for start in sorted(goals)[:n_starts] for goal in goals[start][:n_goals]
    ]


def problem_language(path: Path) -> str:
    """The language the problems in `path` were generated in."""
    return str(json.loads((path / "problem_args.json").read_text())["language"])


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


@dataclass(frozen=True)
class Measured:
    payload: Any
    peak_rss: int
    # From the child's spawn to its exit, so time queued for a slot is excluded.
    wall_time: float


# Machine-readable eqsat progress events (`EVENT_PREFIX` in src/eqsat.rs),
# emitted under `--print-success-iters`.
EQSAT_ITER_RE = re.compile(r"^@EQSAT iter=(\d+)$", re.MULTILINE)
# EQSAT_DONE_RE = re.compile(r"^@EQSAT done\b", re.MULTILINE)

# The cgroup OOM killer SIGKILLs the child; `systemd-run` reports that as
# 128+SIGKILL, a direct child as -SIGKILL.
OOM_RETURNCODES = (-9, 137)

# Rust's exit code for a panic
PANIC_RETURNCODE = 101

# How many seconds past `max_pair_time` a guided-search child may run before it is killed.
TIMEOUT_GRACE = 5.0

# Linux's `MAX_ARG_STRLEN`: the most bytes, including the trailing NUL, that a
# single argv entry may have, else `exec` fails with `E2BIG`.
MAX_ARG_STRLEN = 32 * 4096

# Where a Rust panic message header
# `thread 'main' (609420) panicked at src/langs/math/mod.rs:97:13:`.
PANIC_HEADER_RE = re.compile(r"^thread '.*' .*panicked at ", re.MULTILINE)


class ArgTooLong(ValueError):
    """An argv entry exceeds the kernel's per-argument limit."""

    def __init__(self, what: str, flag: str, size: int) -> None:
        super().__init__(
            f"{what}: argument of {flag} is {size} bytes, "
            f"over the kernel's per-argument limit of {MAX_ARG_STRLEN - 1}"
        )


def check_arg_sizes(cmd: list[str], what: str) -> None:
    """Raise `ArgTooLong` if any entry of `cmd` would not fit through `exec`."""
    for i, arg in enumerate(cmd):
        size = len(arg.encode())
        if size >= MAX_ARG_STRLEN:
            flag = cmd[i - 1] if i > 0 and cmd[i - 1].startswith("--") else f"argv[{i}]"
            raise ArgTooLong(what, flag, size)


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


def prefix_rss_cap(argv: list[str], limit_bytes: int) -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={limit_bytes}",
        "-p",
        "MemorySwapMax=0",
        # An OOM-killed scope stops in `failed` state and stays loaded, so a run
        # that caps on purpose leaks one unit per kill until the user manager
        # goes `degraded`. This lets systemd collect them as they die.
        "-p",
        "CollectMode=inactive-or-failed",
        "--",
    ] + argv


async def run_json_subprocess(
    cmd: list[str],
    *,
    what: str,
    limit: PrioritySlot,
    max_rss_bytes: int | None = None,
    timeout: float | None = None,
) -> Measured:
    """Run a JSON child and return its payload plus its peak RSS.

    The child prints a `Measured` envelope (`src/cli.rs`): the payload it would
    otherwise print, under a `peak_rss` the binary reads from its own
    `VmHWM` just before serializing.

    The child only spawns once it holds a slot of `limit`, and its
    `wall_time` starts counting then.

    Raises `ArgTooLong` before spawning when an argv entry exceeds the kernel's
    per-argument limit, `MemoryKilled` when `max_rss_bytes` is set and the cap
    killed it, `TimeoutKilled` when `timeout` is set and the child outran it,
    and `BinaryPanicked` when the child panicked.
    """
    check_arg_sizes(cmd, what)
    async with limit:
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *(prefix_rss_cap(cmd, max_rss_bytes) if max_rss_bytes else cmd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(timeout):
                out, err = await proc.communicate()
        except (asyncio.CancelledError, TimeoutError) as stopped:
            # Cancelling or timing out only unwinds the *read*, so the child has to be killed by hand.
            # `systemd-run --scope` execs the workload in place rather than forking,
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                # An already-cancelled caller is handed a second `CancelledError`
                # The child watcher reaps the child either way.
                with contextlib.suppress(asyncio.CancelledError):
                    await proc.wait()
            if isinstance(stopped, TimeoutError):
                assert timeout is not None
                raise TimeoutKilled(what, timeout, time.monotonic() - started) from None
            raise
        wall_time = time.monotonic() - started

    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")

    if max_rss_bytes is not None and proc.returncode in OOM_RETURNCODES:
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
    """An `attempt_summary`-shaped row, with every measurement left at `None`,
    for a child that never printed one: `out_of_memory` for one SIGKILLed at its
    cgroup RSS cap, `binary_panic` for one that died of an uncaught panic (unlike
    a panic caught inside the eqsat run, `stop_reason="panic"`), `arg_too_long`
    for one never spawned since an argument would not fit through `exec`,
    `timeout` for one killed at its hard timeout (unlike egg's `TimeLimit`,
    which the eqsat run reports itself).
    """
    empty = dict.fromkeys(ATTEMPT_DTYPES)
    panic = stop_reason == "binary_panic"
    return {**empty, "reached": False, "panic": panic, "stop_reason": stop_reason}


@dataclass(frozen=True)
class AttemptResult:
    summary: dict
    peak_rss: int | None
    wall_time: float


async def measure_attempt(
    cmd: list[str],
    *,
    what: str,
    max_rss_bytes: int,
    limit: PrioritySlot,
    timeout: float | None = None,
) -> AttemptResult:
    """Run one `attempt` process under the RSS cap and the hard timeout.

    An attempt killed at the cap, by an uncaught panic, at its hard timeout, or
    never spawned since an argument exceeds the kernel's limit comes back as a
    failed attempt with ``stop_reason="out_of_memory"``/``"binary_panic"``/
    ``"timeout"``/``"arg_too_long"`` rather than an exception.
    """
    try:
        measured = await run_json_subprocess(
            cmd, what=what, max_rss_bytes=max_rss_bytes, limit=limit, timeout=timeout
        )
    except ArgTooLong as too_long:
        log(f"WARNING: {too_long}")
        return AttemptResult(failure_summary("arg_too_long"), None, 0.0)
    except MemoryKilled as killed:
        return AttemptResult(failure_summary("out_of_memory"), None, killed.wall_time)
    except TimeoutKilled as timed_out:
        log(f"WARNING: {timed_out}")
        return AttemptResult(failure_summary("timeout"), None, timed_out.wall_time)
    except BinaryPanicked as panicked:
        panicked.warn()
        return AttemptResult(failure_summary("binary_panic"), None, panicked.wall_time)
    return AttemptResult(attempt_summary(measured.payload), measured.peak_rss, measured.wall_time)


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


async def fan_out(
    fn: Callable[[Any], Coroutine[Any, Any, Any]],
    items: list,
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


class BaselineMismatch(RuntimeError):
    """A stored baseline that was computed under other flags or misses pairs."""


def check_baseline(directory: Path, expected: dict, pairs: list[Pair]) -> None:
    """Check that the stored baseline in `directory` was computed under the
    `baseline_key` `expected` and covers `pairs`.

    Raises `BaselineMismatch` if there is no finished baseline in `directory`,
    or it was computed under a different `baseline_key` or does not cover every pair.
    """
    if not (directory / "config.json").is_file():
        raise BaselineMismatch(f"no finished baseline in {directory}; run `baseline.py` first")
    config = json.loads((directory / "config.json").read_text())
    stored = config["baseline_key"]
    if stored != expected:
        diff = {
            key: (stored.get(key), value)
            for key, value in expected.items()
            if stored.get(key) != value
        }
        raise BaselineMismatch(f"baseline in {directory} differs (stored, wanted): {diff}")

    wanted = pl.DataFrame(
        {"start": [p.start for p in pairs], "goal": [p.goal for p in pairs]},
        schema={"start": pl.String, "goal": pl.String},
    )
    rows = pl.read_parquet(directory / "unguided.parquet")
    missing = wanted.join(rows, on=["start", "goal"], how="anti")
    if len(missing):
        raise BaselineMismatch(f"baseline in {directory} misses {len(missing)} pair(s)")
