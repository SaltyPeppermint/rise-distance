"""Shared helpers for the driver scripts: size parsing, subprocess-JSON
plumbing, binary checks, and eqsat CLI flag building."""

import json
import subprocess
import sys
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


def run_json_subprocess(
    cmd: list[str],
    *,
    what: str,
    input: str | None = None,
    timeout: float | None = None,
) -> Any:
    """Run `cmd`, expecting a JSON payload on stdout.

    Raises RuntimeError with stdout/stderr attached on a nonzero exit or
    non-JSON stdout; `what` names the failing unit in those messages
    (e.g. "goal for seed '(+ a b)'").
    """
    proc = subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed (code {proc.returncode}):\n{proc.stderr}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{what} returned non-JSON stdout: {e}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        ) from e


#: Sentinel written as a term's measurement when `measure-size` fails to
#: produce a usable `{"peak", "iterations"}` object (nonzero exit, RLIMIT
#: kill, timeout, or unparseable output). `peak == -1` marks the failure.
MEASURE_FAILURE: dict = {"peak": -1, "iterations": []}


def measure_term(cmd: list[str], *, timeout: float) -> dict:
    """Run a `measure-size`-style command, returning the `{"peak", "iterations"}`
    JSON object parsed from the last line of stdout, or a fresh copy of
    `MEASURE_FAILURE` on any error (nonzero exit, timeout, unparseable output)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return dict(MEASURE_FAILURE)
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError, json.JSONDecodeError):
        return dict(MEASURE_FAILURE)


def eqsat_limits(cfg: dict) -> dict:
    """Extract the eqsat limits from a raw config dict (`generation_args.json`
    / `goal_args.json`). `max_memory` (RSS ceiling in bytes) is optional."""
    return {
        "max_iters": cfg["max_iters"],
        "max_nodes": cfg["max_nodes"],
        "max_time": cfg["max_time"],
        "max_memory": cfg.get("max_memory"),
        "backoff_scheduler": bool(cfg.get("backoff_scheduler", True)),
    }


def limit_flags(limits: dict) -> list[str]:
    """Turn an eqsat-limit dict into the `--max-*` CLI flags the Rust binaries
    take. `--max-memory` is added only when set; `--backoff-scheduler` is a
    presence flag, added only when true."""
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
    if limits["backoff_scheduler"]:
        flags.append("--backoff-scheduler")
    return flags


def language_eqsat_flags(cfg: dict) -> list[str]:
    """`--language` plus the eqsat limit flags, from a raw config dict."""
    return ["--language", str(cfg["language"]), *limit_flags(eqsat_limits(cfg))]
