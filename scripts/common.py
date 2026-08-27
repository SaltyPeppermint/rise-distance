"""Shared helpers for the driver scripts: size parsing, subprocess-JSON
plumbing, binary checks, and eqsat CLI flag building."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
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


def run_json_subprocess(
    cmd: list[str],
    *,
    what: str,
    rss_max_bytes: int | None = None,
    input: str | None = None,
    timeout: float | None = None,
) -> MeasuredJson:
    """Run a JSON child and return its Linux lifetime high-water RSS.

    Temporary files keep large JSON and log streams from blocking while
    ``wait4`` retains per-child ``ru_maxrss`` even under concurrent drivers.
    """
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdin_file,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file,
    ):
        if input is not None:
            stdin_file.write(input)
            stdin_file.seek(0)
        # print(f"------\n{' '.join(cmd)}\n-----")
        proc = subprocess.Popen(
            prefix_rss_cap(cmd, rss_max_bytes) if rss_max_bytes else cmd,
            stdin=stdin_file if input is not None else subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        started = time.monotonic()
        usage = None
        while True:
            waited, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if waited:
                proc.returncode = os.waitstatus_to_exitcode(status)
                break
            if timeout is not None and time.monotonic() - started > timeout:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                waited, status, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
                raise subprocess.TimeoutExpired(cmd, timeout)
            time.sleep(0.01)

        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed (code {proc.returncode}):\n{stderr}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{what} returned non-JSON stdout: {e}\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        ) from e
    # Linux reports ru_maxrss in KiB.
    return MeasuredJson(payload=payload, peak_rss_bytes=int(usage.ru_maxrss) * 1024)


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
        "predict_next_memory": cfg.get("predict_next_memory"),
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
