# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "tqdm"]
# ///
"""Measure peak resident set size of running eqsat on each term in a JSON.

Spawns the `measure-size` Rust binary once per row. The binary reads its
own peak RSS from `/proc/self/status` (VmHWM) at exit and prints the
value (bytes) on stdout -> this matches what htop reports. A virtual-
memory cap is enforced inside the binary via RLIMIT_AS as a safety net
against runaways.

Writes the result back to the JSON in a `peak_memory_bytes` column.
On any failure (non-zero exit, RLIMIT kill, timeout, unparseable
output), the value is -1.

Example:
    cargo build --release --bin measure-size
    uv run scripts/measure_size.py --path output.json --max-memory 8G --max-time 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm


def parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = 1
    for suf, m in (("K", 1024), ("M", 1024**2), ("G", 1024**3), ("T", 1024**4)):
        if s.endswith(suf):
            mult = m
            s = s[:-1]
            break
    return int(float(s) * mult)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="JSON with a `term` column. Edited in place.",
    )
    parser.add_argument(
        "--binary", type=Path, default=Path("target/release/measure-size")
    )
    parser.add_argument(
        "--max-memory",
        type=parse_size,
        required=True,
        help="Per-term virtual-memory cap (e.g. 8G). Backstop only.",
    )
    parser.add_argument("--max-iters", type=int, default=11)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument(
        "--backoff-scheduler",
        action="store_true",
        help="Use egg's BackoffScheduler instead of the SimpleScheduler.",
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(
            f"Binary not found: {args.binary}. Build with `cargo build --release --bin measure-size`.",
            file=sys.stderr,
        )
        return 2

    df = pl.read_json(args.path)
    if "term" not in df.columns:
        print("JSON must contain a `term` column", file=sys.stderr)
        return 2

    timeout = max(1, int(args.max_time * 4) + 5)
    cmd_base = [
        str(args.binary),
        "--max-iters",
        str(args.max_iters),
        "--max-nodes",
        str(args.max_nodes),
        "--max-time",
        str(args.max_time),
        "--max-memory",
        str(args.max_memory),
    ]
    if args.backoff_scheduler:
        cmd_base.append("--backoff-scheduler")

    measurements: list[int] = []
    for term in tqdm(df["term"].to_list(), desc="terms"):
        try:
            proc = subprocess.run(
                [*cmd_base, "--term", term],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            peak = (
                int(proc.stdout.strip().splitlines()[-1])
                if proc.returncode == 0
                else -1
            )
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            peak = -1
        measurements.append(peak)

    df.with_columns(pl.Series("peak_memory_bytes", measurements)).write_json(args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
