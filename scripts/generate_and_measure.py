# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "tqdm"]
# ///
"""Generate random math terms and measure peak heap memory of eqsat on each.

Shells out to `target/release/generate` to produce a CSV of terms, then to
`target/release/measure-size` once per row to record peak heap usage in a
`peak_memory_bytes` column. Egg limits (`--max-iters`, `--max-nodes`,
`--max-time`, `--backoff-scheduler`) are forwarded to both binaries.

On any per-term failure (non-zero exit, RLIMIT kill, timeout, unparseable
output), `peak_memory_bytes` is -1.

Example:
    cargo build --release --bin generate --bin measure-size
    uv run scripts/generate_and_measure.py \\
        --path output.csv --total-samples 1000 --min-size 10 --max-size 50 \\
        --distribution uniform --seed 42 --max-memory 8G \\
        --max-iters 50 --max-nodes 100000 --max-time 10 --backoff-scheduler
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
    parser.add_argument("--path", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--generate-binary",
        type=Path,
        default=Path("target/release/generate"),
    )
    parser.add_argument(
        "--measure-binary",
        type=Path,
        default=Path("target/release/measure-size"),
    )

    # generate-only args
    parser.add_argument("--total-samples", type=int, required=True)
    parser.add_argument("--min-size", type=int, required=True)
    parser.add_argument("--max-size", type=int, required=True)
    parser.add_argument("--distribution", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--retry-limit", type=int, default=10000)
    parser.add_argument("--parallelism", type=int, default=None)

    # measure-only args
    parser.add_argument(
        "--max-memory",
        type=parse_size,
        required=True,
        help="Per-term virtual-memory cap (e.g. 8G). Backstop only.",
    )

    # shared egg args
    parser.add_argument("--max-iters", type=int, default=11)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument("--backoff-scheduler", action="store_true")

    args = parser.parse_args()

    for binary in (args.generate_binary, args.measure_binary):
        if not binary.exists():
            print(
                f"Binary not found: {binary}. "
                f"Build with `cargo build --release --bin generate --bin measure-size`.",
                file=sys.stderr,
            )
            return 2

    gen_cmd = [
        str(args.generate_binary),
        "--total-samples",
        str(args.total_samples),
        "--min-size",
        str(args.min_size),
        "--max-size",
        str(args.max_size),
        "--distribution",
        args.distribution,
        "--seed",
        str(args.seed),
        "--tolerance",
        str(args.tolerance),
        "--retry-limit",
        str(args.retry_limit),
        "--path",
        str(args.path),
        "--max-iters",
        str(args.max_iters),
        "--max-nodes",
        str(args.max_nodes),
        "--max-time",
        str(args.max_time),
    ]
    if args.parallelism is not None:
        gen_cmd += ["--parallelism", str(args.parallelism)]
    if args.backoff_scheduler:
        gen_cmd.append("--backoff-scheduler")

    print(f"Generating terms -> {args.path}", file=sys.stderr)
    gen = subprocess.run(gen_cmd)
    if gen.returncode != 0:
        print(f"generate failed (exit {gen.returncode})", file=sys.stderr)
        return gen.returncode

    df = pl.read_csv(args.path)
    if "term" not in df.columns:
        print("Generated CSV is missing a `term` column", file=sys.stderr)
        return 2

    timeout = max(1, int(args.max_time * 4) + 5)
    measure_base = [
        str(args.measure_binary),
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
        measure_base.append("--backoff-scheduler")

    measurements: list[int] = []
    for term in tqdm(df["term"].to_list(), desc="terms"):
        try:
            proc = subprocess.run(
                [*measure_base, "--term", term],
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

    df.with_columns(pl.Series("peak_memory_bytes", measurements)).write_csv(args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
