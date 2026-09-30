"""Run the unguided baseline for a problem folder once.

The baseline only depends on the problems and the guide-replay budget, not on
any of the search flags, so a grid of `guided_search.py` runs can share one
through `--baseline` instead of each re-running it.

Example:
    cargo build --release --bin attempt
    uv run scripts/baseline.py data/problems/dusky-cramp --stop-iters 50 --max-rss 4G

The result lands in ``data/baselines/<problem folder>`` unless ``--output`` is
given. An existing baseline there is kept if it matches and covers every pair,
and is a hard error otherwise.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import polars as pl
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict

from common import (
    BinaryPanicked,
    MemoryKilled,
    Problem,
    attempt_summary,
    binary_panic_summary,
    cli_flags,
    exit_if_missing,
    fan_out,
    flatten_problems,
    out_of_memory_summary,
    parse_size,
    run_json_subprocess,
)
from schemes import UNGUIDED_SCHEMA


class BaselineArgs(BaseSettings):
    """The flags the unguided baseline depends on, shared with `guided_search.py`."""

    model_config = SettingsConfigDict(cli_kebab_case=True, cli_implicit_flags=True)

    # I/O
    path: CliPositionalArg[Path] = Field(
        description=("Problem folder with `the problems (both written by `generate_problems.py`).")
    )

    output: Path | None = Field(
        default=None,
        description=("Baseline folder. `data/baselines/<problem folder>` if omitted."),
    )

    attempt_bin: Path = Field(
        default=Path("target/release/attempt"), description="Path to the attempt binary."
    )

    # Guide-replay budget
    #
    # At least one must be given. Replay ends when the first configured budget
    # is exhausted; omitted budgets are effectively unlimited.
    stop_iters: int | None = Field(
        default=None, gt=0, description=("Guide-replay iteration budget.")
    )

    stop_nodes: int | None = Field(
        default=None, gt=0, description=("Guide-replay egraph-node budget.")
    )

    stop_time: float | None = Field(
        default=None, gt=0, description=("Guide-replay wall-clock budget in seconds.")
    )

    max_rss: str = Field(
        default="4G",
        description=(
            "Cap each `sample`/`attempt` process at this cgroup RSS limit, as a "
            "human size such as `4G`. A killed `sample` is retried, still "
            "capped, with `--max-iters` cut to the iterations that completed."
        ),
    )

    start_terms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only process the first N start terms in sorted order. All start terms are processed if omitted."
        ),
    )

    goal_terms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only use the first N goals per start term in file order. All goals are used if omitted."
        ),
    )

    jobs: int | None = Field(
        default=None,
        gt=0,
        description=("Maximum number of concurrent `sample`/`attempt` processes."),
    )

    @property
    def limits(self) -> dict[str, int | float]:
        """The configured guide-replay budgets, without the omitted ones."""
        budgets = {
            "max_iters": self.stop_iters,
            "max_nodes": self.stop_nodes,
            "max_time": self.stop_time,
        }
        return {key: value for key, value in budgets.items() if value is not None}

    def base_flags(self, language: str) -> list[str]:
        """Flags shared by every `attempt` process."""
        return cli_flags(language=language, **self.limits)

    def baseline_key(self) -> dict:
        """Everything a baseline row depends on besides its pair.

        Two runs with the same key can share a baseline.
        """
        return {
            "problems": str(self.path.resolve()),
            "stop_iters": self.stop_iters,
            "stop_nodes": self.stop_nodes,
            "stop_time": self.stop_time,
            "max_rss": parse_size(self.max_rss),
        }


async def run_unguided_pair(
    args: BaselineArgs, base_flags: list[str], limit: asyncio.Semaphore, pair: Problem
) -> dict:
    """Run the pair-matched single-start baseline.

    A baseline killed at the RSS cap or by an uncaught panic becomes an
    ``out_of_memory``/``binary_panic`` failure row.
    """
    cmd = [
        str(args.attempt_bin),
        *base_flags,
        *cli_flags(start_term=pair.start, goal_term=pair.goal),
    ]
    what = f"unguided attempt for goal term {pair.goal!r}"
    try:
        measured = await run_json_subprocess(
            cmd, what=what, rss_max_bytes=parse_size(args.max_rss), limit=limit
        )
        summary = attempt_summary(measured.payload)
        peak_rss_bytes = measured.peak_rss_bytes
    except MemoryKilled:
        summary, peak_rss_bytes = out_of_memory_summary(), None
    except BinaryPanicked as panicked:
        panicked.warn()
        summary, peak_rss_bytes = binary_panic_summary(), None
    return {
        "start_term": pair.start,
        "goal_term": pair.goal,
        "unguided_success": summary["reached"],
        "unguided_stop_reason": summary["stop_reason"],
        "unguided_panic": summary["panic"],
        "unguided_final_live_heap_bytes": summary["memory"],
        "unguided_peak_live_heap_bytes": summary["peak_live_heap"],
        "unguided_peak_rss_bytes": peak_rss_bytes,
    }


class BaselineMismatch(RuntimeError):
    """A stored baseline that was computed under other flags or misses pairs."""


def load_baseline(args: BaselineArgs, directory: Path, pairs: list[Problem]) -> pl.DataFrame:
    """The stored baseline rows for `pairs`, checked against `args`.

    Raises `BaselineMismatch` if there is no finished baseline in `directory`,
    or it was computed under a different `baseline_key` or does not cover every pair.
    """
    if not (directory / "config.json").is_file():
        raise BaselineMismatch(f"no finished baseline in {directory}; run `baseline.py` first")
    config = json.loads((directory / "config.json").read_text())
    expected = args.baseline_key()
    if config["baseline_key"] != expected:
        diff = {
            key: (config["baseline_key"].get(key), value)
            for key, value in expected.items()
            if config["baseline_key"].get(key) != value
        }
        raise BaselineMismatch(f"baseline in {directory} differs (stored, wanted): {diff}")

    wanted = pl.DataFrame(
        {"start_term": [p.start for p in pairs], "goal_term": [p.goal for p in pairs]},
        schema={"start_term": pl.String, "goal_term": pl.String},
    )
    rows = pl.read_parquet(directory / "unguided_results.parquet")
    missing = wanted.join(rows, on=["start_term", "goal_term"], how="anti")
    if len(missing):
        raise BaselineMismatch(f"baseline in {directory} misses {len(missing)} pair(s)")
    return wanted.join(rows, on=["start_term", "goal_term"], how="left", validate="1:1")


def default_baseline_dir(problems: Path) -> Path:
    """Where a problem folder's baseline lives unless told otherwise."""
    return Path("data/baselines") / problems.name


async def main(args: BaselineArgs) -> int:
    exit_if_missing(args.attempt_bin)
    out = args.output or default_baseline_dir(args.path)
    pairs = flatten_problems(args.path, args.start_terms, args.goal_terms)

    if (out / "config.json").is_file():
        try:
            load_baseline(args, out, pairs)
        except BaselineMismatch as mismatch:
            print(f"{mismatch}; remove it or pass another --output", file=sys.stderr)
            return 1
        print(f"Baseline in {out} already covers all {len(pairs)} pair(s)", file=sys.stderr)
        return 0

    cfg = json.loads((args.path / "problem_args.json").read_text())
    base_flags = args.base_flags(str(cfg["language"]))
    limit = asyncio.Semaphore(args.jobs or os.cpu_count() or 1)
    rows = await fan_out(
        None,
        lambda pair: run_unguided_pair(args, base_flags, limit, pair),
        pairs,
        "unguided",
        unit="pair",
    )

    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=UNGUIDED_SCHEMA).write_parquet(out / "unguided_results.parquet")
    # Written last, so a baseline killed halfway is not mistaken for a finished one.
    config = {**args.model_dump(), "baseline_key": args.baseline_key()}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    reached = sum(row["unguided_success"] for row in rows)
    print(f"\nUnguided reached {reached}/{len(rows)} pair(s). Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    args = CliApp.run(BaselineArgs)
    raise SystemExit(asyncio.run(main(args)))
