"""Run the unguided baseline for a problem folder once.

The baseline only depends on the problems and the guide-replay budget, not on
any of the search flags, so a grid of `guided_search.py` runs can share one
through `baseline` instead of each re-running it. `experiment.py` drives it.

An existing baseline in `output` is kept if it matches and covers every
pair, and is a hard error otherwise.
"""

import json
from pathlib import Path

import polars as pl
from pydantic import Field

from common import (
    Pair,
    PrioritySlot,
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
    args: BaselineArgs, base_flags: list[str], limit: PrioritySlot, pair: Pair
) -> dict:
    """Run the pair-matched single-start baseline.

    A baseline killed at the RSS cap, by an uncaught panic, or with a too-long
    argument becomes an ``out_of_memory``/``binary_panic``/``arg_too_long``
    failure row.
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


async def compute_baseline(args: BaselineArgs, limit: PrioritySlot) -> str:
    """Compute the baseline into `output` and return a one-line summary.

    Raises `BaselineMismatch` if `output` holds a baseline that does not fit.
    """
    exit_if_missing(args.attempt_bin)
    out = args.output
    pairs = load_pairs(args.path, args.n_starts, args.n_goals)

    if (out / "config.json").is_file():
        check_baseline(out, args.baseline_key(), pairs)
        return f"Baseline in {out} already covers all {len(pairs)} pair(s)"

    base_flags = args.base_flags(problem_language(args.path))
    rows = await fan_out(
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
    return f"Unguided reached {reached}/{len(rows)} pair(s). Wrote {out}"
