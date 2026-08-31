"""Shared helpers for the driver scripts: size parsing, subprocess-JSON
plumbing, binary checks, and eqsat CLI flag building."""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def eqsat_finished(stderr: str) -> bool:
    """Whether the eqsat run itself reached a stop reason."""
    return EQSAT_DONE_RE.search(stderr) is not None


class MemoryKilled(RuntimeError):
    """A capped child was SIGKILLed by its cgroup memory limit."""

    def __init__(self, what: str, stderr: str) -> None:
        super().__init__(f"{what} was killed at its RSS cap")
        self.stderr = stderr
        self.last_iter = last_success_iter(stderr)


def run_json_subprocess(
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
    proc = subprocess.run(
        prefix_rss_cap(cmd, rss_max_bytes) if rss_max_bytes else cmd,
        input=input,
        stdin=None if input is not None else subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if rss_max_bytes is not None and proc.returncode in OOM_RETURNCODES:
        raise MemoryKilled(what, proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed (code {proc.returncode}):\n{proc.stderr}")
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{what} returned non-JSON stdout: {e}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        ) from e
    return MeasuredJson(payload=envelope["payload"], peak_rss_bytes=int(envelope["peak_rss_bytes"]))


def stop_reason_name(raw: Any) -> str:
    """Render egg's serialized `StopReason` the way Rust's `{:?}` does
    (`Saturated`, `NodeLimit(1000)`), which is what the analysis matches on."""
    if isinstance(raw, str):
        return raw
    variant, payload = next(iter(raw.items()))
    return f"{variant}({json.dumps(payload)})"


VERIFY_FIELDS = (
    "reached",
    "panic",
    "stop_reason",
    "iters",
    "nodes",
    "classes",
    "total_applied",
    "total_time",
    "memory",
    "peak_live_heap",
)


def verify_summary(payload: Any) -> dict[str, Any]:
    """Flatten `verify`'s `Result<ReachedRun, GuideError>` stdout payload.

    Unreached and panicked runs leave the egraph-shape fields at `None`.
    """
    empty: dict[str, Any] = dict.fromkeys(VERIFY_FIELDS)
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


def eqsat_limits(cfg: dict) -> dict:
    """Extract the eqsat limits from a raw config dict (`problem_args.json`).
    `max_memory` is an optional absolute process
    live-heap ceiling (jemalloc `stats.allocated`), accepted as a human size
    string (e.g. `"1G"`) or a raw byte count, normalized to bytes. Rust compares
    it directly against the process live heap, with nothing subtracted out."""
    max_memory = cfg.get("max_memory")
    if isinstance(max_memory, str):
        max_memory = parse_size(max_memory)
    return {
        "max_iters": cfg["max_iters"],
        "max_nodes": cfg["max_nodes"],
        "max_time": cfg["max_time"],
        "max_memory": max_memory,
    }


def limit_flags(limits: dict) -> list[str]:
    """Turn an eqsat-limit dict into the `--max-*` CLI flags the Rust binaries
    take. Optional memory flags are added only when set."""
    flags = [
        "--max-iters",
        str(limits["max_iters"]),
        "--max-nodes",
        str(limits["max_nodes"]),
        "--max-time",
        str(limits["max_time"]),
    ]
    if limits.get("max_memory") is not None:
        flags += ["--max-memory", str(limits["max_memory"])]
    return flags


def with_max_iters(cmd: list[str], max_iters: int) -> list[str]:
    """Copy `cmd` with its `--max-iters` value replaced."""
    out = list(cmd)
    out[out.index("--max-iters") + 1] = str(max_iters)
    return out


def uniform_candidate_allocation(
    sizes: list[int],
    total_candidates: int,
) -> list[tuple[int, int]]:
    if not sizes:
        return []

    size_count = len(sizes)
    base = total_candidates // size_count
    remainder = total_candidates % size_count

    return [(size, base + int(i < remainder)) for i, size in enumerate(sizes)]
