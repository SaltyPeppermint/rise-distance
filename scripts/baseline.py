"""Run the unguided baseline for a problem folder once.

The baseline only depends on the problems and the guide-replay budget, not on
any of the search flags, so a grid of `guided_search.py` runs can share one
through `--baseline` instead of each re-running it.

Example:
    cargo build --release --bin attempt
    uv run scripts/baseline.py data/problems/dusky-cramp \\
        --output data/baselines/dusky-cramp --max-iters 50 --max-rss 4G

An existing baseline in ``--output`` is kept if it matches and covers every
pair, and is a hard error otherwise.
"""

import asyncio
import json
import sys
from pathlib import Path

import polars as pl
from pydantic import Field
from pydantic_settings import CliApp

from common import (
    BaselineMismatch,
    Pair,
    check_baseline,
    cli_flags,
    exit_if_missing,
    fan_out,
    load_pairs,
    measure_attempt,
    problem_language,
)
from replay_args import ReplayArgs
from schemes import UNGUIDED_SCHEMA


class BaselineArgs(ReplayArgs):
    """`ReplayArgs` plus where the baseline goes."""

    output: Path = Field(description="Baseline folder.")


async def run_unguided_pair(
    args: BaselineArgs, base_flags: list[str], limit: asyncio.Semaphore, pair: Pair
) -> dict:
    """Run the pair-matched single-start baseline.

    A baseline killed at the RSS cap or by an uncaught panic becomes an
    ``out_of_memory``/``binary_panic`` failure row.
    """
    cmd = [
        str(args.attempt_bin),
        *base_flags,
        *cli_flags(start=pair.start, goal=pair.goal),
    ]
    attempt = await measure_attempt(
        cmd,
        what=f"unguided attempt for goal term {pair.goal!r}",
        max_rss_bytes=args.max_rss_bytes,
        limit=limit,
    )
    summary = attempt.summary
    return {
        "start": pair.start,
        "goal": pair.goal,
        "unguided_success": summary["reached"],
        "unguided_stop_reason": summary["stop_reason"],
        "unguided_panic": summary["panic"],
        "unguided_final_live_heap": summary["final_live_heap"],
        "unguided_peak_live_heap": summary["peak_live_heap"],
        "unguided_peak_rss": attempt.peak_rss,
    }


async def main(args: BaselineArgs) -> int:
    exit_if_missing(args.attempt_bin)
    out = args.output
    pairs = load_pairs(args.path, args.n_starts, args.n_goals)

    if (out / "config.json").is_file():
        try:
            check_baseline(out, args.baseline_key(), pairs)
        except BaselineMismatch as mismatch:
            print(f"{mismatch}; remove it or pass another --output", file=sys.stderr)
            return 1
        print(f"Baseline in {out} already covers all {len(pairs)} pair(s)", file=sys.stderr)
        return 0

    base_flags = args.base_flags(problem_language(args.path))
    limit = asyncio.Semaphore(args.jobs)
    rows = await fan_out(
        None,
        lambda pair: run_unguided_pair(args, base_flags, limit, pair),
        pairs,
        "unguided",
        unit="pair",
    )

    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=UNGUIDED_SCHEMA).write_parquet(out / "unguided.parquet")
    # Written last, so a baseline killed halfway is not mistaken for a finished one.
    config = {**args.model_dump(), "baseline_key": args.baseline_key()}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    reached = sum(row["unguided_success"] for row in rows)
    print(f"\nUnguided reached {reached}/{len(rows)} pair(s). Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    args = CliApp.run(BaselineArgs)
    raise SystemExit(asyncio.run(main(args)))
