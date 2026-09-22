"""Shared helpers for the driver scripts: size parsing, subprocess-JSON
plumbing, binary checks, the `attempt` payload schema, and eqsat CLI flag
building."""

import asyncio
import contextlib
import json
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from schemes import ATTEMPT_DTYPES


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


def subprocess_timeout(max_time: float) -> int:
    """Per-term subprocess timeout: eqsat's `max_time` plus slack for
    non-eqsat overhead (startup, serialization)."""
    return max(1, int(max_time * 4) + 5)


def check_binaries(*binaries: Path) -> str | None:
    """Return an error message if any binary is missing, else `None`."""
    missing = [b for b in binaries if not b.exists()]
    if missing:
        names = " ".join(f"--bin {b.name}" for b in missing)
        return (
            f"Binary not found: {', '.join(str(b) for b in missing)}. "
            f"Build with `cargo build --release {names}`."
        )
    return None


def exit_if_missing(*binaries: Path) -> None:
    """Print an error and exit 2 if any binary is missing."""
    error = check_binaries(*binaries)
    if error is not None:
        print(error, file=sys.stderr)
        raise SystemExit(2)


@dataclass(frozen=True)
class MeasuredJson:
    payload: Any
    peak_rss_bytes: int


def prefix_rss_cap(argv: list[str], limit_bytes) -> list[str]:
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


# Machine-readable eqsat progress events (`EVENT_PREFIX` in src/eqsat.rs),
# emitted under `--print-success-iters`.
EQSAT_ITER_RE = re.compile(r"^@EQSAT iter=(\d+)$", re.MULTILINE)
EQSAT_DONE_RE = re.compile(r"^@EQSAT done\b", re.MULTILINE)

# The cgroup OOM killer SIGKILLs the child; `systemd-run` reports that as
# 128+SIGKILL, a direct child as -SIGKILL.
OOM_RETURNCODES = (-9, 137)


def last_success_iter(stderr: str) -> int | None:
    """Last announced iteration, which is the number of iterations that
    completed before the child died."""
    matches = EQSAT_ITER_RE.findall(stderr)
    return int(matches[-1]) if matches else None


class MemoryKilled(RuntimeError):
    """A capped child was SIGKILLed by its cgroup memory limit."""

    def __init__(self, what: str, stderr: str) -> None:
        super().__init__(f"{what} was killed at its RSS cap")
        self.stderr = stderr
        self.last_iter = last_success_iter(stderr)


async def run_json_subprocess(
    cmd: list[str],
    *,
    what: str,
    rss_max_bytes: int | None = None,
    input: str | None = None,
    timeout: float | None = None,
) -> MeasuredJson:
    """Run a JSON child and return its payload plus its peak RSS.

    The child prints a `Measured` envelope (`src/cli.rs`): the payload it would
    otherwise print, under a `peak_rss_bytes` the binary reads from its own
    `VmHWM` just before serializing.

    Raises `MemoryKilled` when `rss_max_bytes` is set and the cap killed it.
    """
    proc = await asyncio.create_subprocess_exec(
        *(prefix_rss_cap(cmd, rss_max_bytes) if rss_max_bytes else cmd),
        stdin=asyncio.subprocess.DEVNULL if input is None else asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(None if input is None else input.encode()), timeout
        )
    except TimeoutError, asyncio.CancelledError:
        # Both only unwind the *read*, so the child has to be killed by hand.
        # `systemd-run --scope` execs the workload in place rather than forking,
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            # An already-cancelled caller is handed a second `CancelledError`
            # The child watcher reaps the child either way.
            with contextlib.suppress(asyncio.CancelledError):
                await proc.wait()
        raise

    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")

    if rss_max_bytes is not None and proc.returncode in OOM_RETURNCODES:
        raise MemoryKilled(what, stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed (code {proc.returncode}):\n{stderr}")
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{what} returned non-JSON stdout: {e}\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        ) from e
    return MeasuredJson(payload=envelope["payload"], peak_rss_bytes=int(envelope["peak_rss_bytes"]))


def stop_reason_name(raw: Any) -> str:
    """Render egg's serialized `StopReason` the way Rust's `{:?}` does
    (`Saturated`, `NodeLimit(1000)`), which is what the analysis matches on."""
    if isinstance(raw, str):
        return raw
    variant, payload = next(iter(raw.items()))
    return f"{variant}({json.dumps(payload)})"


def attempt_summary(payload: Any) -> dict[str, Any]:
    """Flatten `attempt`'s `Result<ReachedRun, GuideError>` stdout payload.

    Unreached and panicked runs leave the egraph-shape fields at `None`.
    """
    empty: dict[str, Any] = dict.fromkeys(ATTEMPT_DTYPES)
    if "Ok" in payload:
        run = payload["Ok"]
        iterations = run["iterations"]
        return {
            **empty,
            "reached": True,
            "panic": False,
            "stop_reason": "goal_found",
            "iters": len(iterations),
            "nodes": run["nodes"],
            "classes": run["classes"],
            "total_applied": sum(sum(it["applied"].values()) for it in iterations),
            "total_time": sum(it["total_time"] for it in iterations),
            "memory": run["allocated"],
            "peak_live_heap": run["peak_allocated"],
        }
    err = payload["Err"]
    if isinstance(err, dict) and "Unreached" in err:
        unreached = err["Unreached"]
        return {
            **empty,
            "reached": False,
            "panic": False,
            "stop_reason": stop_reason_name(unreached["stop_reason"]),
            "memory": unreached["final_allocated"],
            "peak_live_heap": unreached["peak_allocated"],
        }
    return {**empty, "reached": False, "panic": True, "stop_reason": "panic"}


def rss_killed_summary() -> dict[str, Any]:
    """An `attempt_summary`-shaped row for a child SIGKILLed at its cgroup RSS cap.

    A killed child never printed its payload, so everything but the outcome
    markers stays `None`.
    """
    empty: dict[str, Any] = dict.fromkeys(ATTEMPT_DTYPES)
    return {**empty, "reached": False, "panic": False, "stop_reason": "rss_killed"}


def cli_flags(**values: str | float | bool | None) -> list[str]:
    flags: list[str] = []

    for key, value in values.items():
        if value is None or value is False:
            continue
        flags.append(f"--{key.replace('_', '-')}")
        if value is not True:
            flags.append(str(value))

    return flags


async def _run_limited(
    limit: asyncio.Semaphore, bar: tqdm, fn: Callable[[Any], Awaitable[Any]], item: Any
) -> Any:
    """Run if a slot is free and tick the bar."""
    async with limit:
        result = await fn(item)
    bar.update(1)
    return result


async def fan_out(
    jobs: int, fn: Callable[[Any], Awaitable[Any]], items: list, desc: str, unit: str = "job"
) -> list:
    """Run `fn` over `items`, `jobs` at a time, dropping the ones that returned None.

    Each item is its own task behind a semaphore, so a slow item holds up only
    itself. Results come back in `items` order rather than completion order, so
    a run's output does not depend on which item finished first.
    """
    limit = asyncio.Semaphore(jobs)
    with tqdm(total=len(items), desc=desc, unit=unit) as bar:
        # A task group rather than `gather`, so the first failure cancels the
        # items still queued on the semaphore instead of letting them start
        # more processes on the way down.
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(_run_limited(limit, bar, fn, item)) for item in items]

    return [result for task in tasks if (result := task.result()) is not None]


def uniform_sample_allocation(sizes: list[int], total_samples: int) -> list[tuple[int, int]]:
    if not sizes:
        return []

    size_count = len(sizes)
    base = total_samples // size_count
    remainder = total_samples % size_count

    return [(size, base + int(i < remainder)) for i, size in enumerate(sizes)]
