"""Load and summarize guided/unguided peak-memory experiments."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

REQUIRED_COMPARISON_COLUMNS = {
    "seed",
    "goal",
    "guided_success",
    "unguided_success",
    "guided_peak_rss_bytes",
    "unguided_peak_rss_bytes",
    "attempts_run",
    "success_attempt",
    "setup_status",
    "predictor_scope",
}
MEMORY_SUMMARY_SCHEMA = {
    "mode": pl.String,
    "n_paired_successes": pl.Int64,
    "guided_median_peak_mib": pl.Float64,
    "unguided_median_peak_mib": pl.Float64,
    "guided_p90_peak_mib": pl.Float64,
    "unguided_p90_peak_mib": pl.Float64,
    "median_peak_ratio": pl.Float64,
    "median_memory_saved_pct": pl.Float64,
    "guided_lower_peak_share": pl.Float64,
}


@dataclass(frozen=True)
class Run:
    directory: Path
    label: str
    config: dict


def _data_dir(subdir: str) -> Path:
    return Path(__file__).parent / ".." / "data" / subdir


def _run_dirs(pattern: str, subdir: str) -> list[Path]:
    base = _data_dir(subdir)
    if not base.is_dir():
        return []
    return sorted(
        (path for path in base.iterdir() if path.is_dir() and pattern in path.name),
        key=lambda path: path.stat().st_mtime,
    )


def _short_strategy(strategy: str) -> str:
    return strategy.removeprefix("no_replacement_").removeprefix("with_replacement_")


def _format_memory_limit(value: int | None) -> str:
    if value is None:
        return "unbounded"
    for unit, size in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= size and value % size == 0:
            return f"{value // size} {unit}"
    return f"{value} B"


def _run_label(directory: Path, config: dict) -> str:
    limits = config.get("effective_limits", {})
    memory = _format_memory_limit(limits.get("max_memory"))
    predictive = "on" if limits.get("predict_next_memory") is not None else "off"
    return (
        f"{_short_strategy(config['strategy'])} · k={config['k']} · "
        f"memory={memory} · predictive={predictive} · {directory.name}"
    )


def resolve_runs(patterns: Sequence[str]) -> list[Run]:
    """Resolve completed individual runs; empty selects every new-schema run."""
    candidates = (
        [path for path in _run_dirs("run.", "guided_search")]
        if not patterns
        else [
            matches[-1] for pattern in patterns if (matches := _run_dirs(pattern, "guided_search"))
        ]
    )
    if patterns and len(candidates) != len(patterns):
        found = {path.name for path in candidates}
        raise FileNotFoundError(f"Could not resolve all run patterns; found {sorted(found)}")

    runs = []
    for directory in dict.fromkeys(candidates):
        comparison = directory / "comparison.parquet"
        config_path = directory / "config.json"
        missing = [path.name for path in (comparison, config_path) if not path.is_file()]
        if missing:
            if patterns:
                raise ValueError(
                    f"{directory} is incomplete; missing final artifacts: {', '.join(missing)}"
                )
            continue
        config = json.loads(config_path.read_text())
        runs.append(Run(directory, _run_label(directory, config), config))
    if not runs:
        raise FileNotFoundError("No completed peak-memory guided-search runs")
    return runs


def _validate_comparison(frame: pl.DataFrame, source: Path) -> None:
    missing = REQUIRED_COMPARISON_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing comparison fields: {sorted(missing)}")


def load_comparisons(runs: Sequence[Run]) -> tuple[pl.DataFrame, dict]:
    """Stack one-row-per-pair comparison files from individual runs."""
    frames = []
    for run in runs:
        frame = pl.read_parquet(run.directory / "comparison.parquet")
        _validate_comparison(frame, run.directory)
        frames.append(
            frame.with_columns(
                pl.lit(run.label).alias("mode"),
                pl.lit(run.directory.name).alias("run"),
                pl.concat_str(["seed", "goal"], separator="│").alias("pair"),
            )
        )
    data = pl.concat(frames, how="diagonal_relaxed")
    scopes = sorted(data["predictor_scope"].drop_nulls().unique().to_list())
    meta = {
        "modes": [run.label for run in runs],
        "n_pairs": data.select("seed", "goal").unique().height,
        "subtitle": [
            f"{data.height} planned pair observations",
            f"predictor scope: {', '.join(scopes)}",
        ],
    }
    return data, meta


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return center - margin, center + margin


def _rate_rows(
    frame: pl.DataFrame,
    group_columns: Sequence[str],
    success_column: str,
) -> list[dict]:
    rows = []
    for keys, group in frame.group_by(*group_columns, maintain_order=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        total = len(group)
        successes = int(group[success_column].fill_null(False).sum())
        lower, upper = _wilson(successes, total)
        rows.append(
            {
                **dict(zip(group_columns, key_values, strict=True)),
                "successes": successes,
                "n": total,
                "success_rate": successes / total if total else None,
                "ci_low": lower,
                "ci_high": upper,
            }
        )
    return rows


def success_rates(frame: pl.DataFrame) -> pl.DataFrame:
    """Guided and unguided success rates with Wilson intervals."""
    rows = []
    for method, column in (
        ("guided", "guided_success"),
        ("unguided", "unguided_success"),
    ):
        for row in _rate_rows(frame, ["mode"], column):
            rows.append({**row, "method": method})
    return pl.DataFrame(rows).sort("mode", "method")


def outcome_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """Counts for the four paired success outcomes."""
    return (
        frame.with_columns(
            pl.when(pl.col("guided_success") & pl.col("unguided_success"))
            .then(pl.lit("both"))
            .when(pl.col("guided_success"))
            .then(pl.lit("guided only"))
            .when(pl.col("unguided_success"))
            .then(pl.lit("unguided only"))
            .otherwise(pl.lit("neither"))
            .alias("outcome")
        )
        .group_by("mode", "outcome")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum().over("mode")).alias("share"))
    )


def _stop_category(reason: pl.Expr) -> pl.Expr:
    """Collapse detailed egg stop strings into stable analysis categories."""
    return (
        pl.when(reason.is_null())
        .then(pl.lit("unknown"))
        .when(reason.str.starts_with("NodeLimit"))
        .then(pl.lit("node limit"))
        .when(reason.str.starts_with("MemoryLimit"))
        .then(pl.lit("memory limit"))
        .when(reason.str.starts_with("TimeLimit"))
        .then(pl.lit("time limit"))
        .when(reason.str.starts_with("IterationLimit"))
        .then(pl.lit("iteration limit"))
        .when(reason.str.starts_with('Other("predicted upcoming-iteration'))
        .then(pl.lit("predictive memory stop"))
        .when(reason == "Saturated")
        .then(pl.lit("saturated without goal"))
        .otherwise(pl.lit("other"))
    )


def failure_breakdown(frame: pl.DataFrame) -> pl.DataFrame:
    """Pair-level, mutually exclusive failure categories for both methods.

    Guided setup failures take precedence over any panic observed in the
    attempt workflow; panic then takes precedence over the terminal stop
    reason. Unguided runs have no guide-sampling setup stage.
    """
    guided = frame.filter(~pl.col("guided_success").fill_null(False)).select(
        "mode",
        pl.lit("guided").alias("method"),
        pl.when(pl.col("setup_status") != "ok")
        .then(pl.lit("setup failure"))
        .when(pl.col("guided_panic").fill_null(False))
        .then(pl.lit("panic"))
        .otherwise(_stop_category(pl.col("guided_stop_reason")))
        .alias("failure"),
    )
    unguided = frame.filter(~pl.col("unguided_success").fill_null(False)).select(
        "mode",
        pl.lit("unguided").alias("method"),
        pl.when(pl.col("unguided_panic").fill_null(False))
        .then(pl.lit("panic"))
        .otherwise(_stop_category(pl.col("unguided_stop_reason")))
        .alias("failure"),
    )
    planned = frame.group_by("mode").agg(pl.len().alias("planned_pairs"))
    return (
        pl.concat([guided, unguided])
        .group_by("mode", "method", "failure")
        .agg(pl.len().alias("count"))
        .with_columns(pl.col("count").sum().over("mode", "method").alias("method_failures"))
        .join(planned, on="mode", how="left")
        .with_columns(
            (pl.col("count") / pl.col("method_failures")).alias("share_of_failures"),
            (pl.col("count") / pl.col("planned_pairs")).alias("share_of_planned"),
        )
        .sort("mode", "method", "count", descending=[False, False, True])
    )


def paired_successes(frame: pl.DataFrame) -> pl.DataFrame:
    """Pairs reached by both methods with valid positive peak-RSS readings."""
    return (
        frame.filter(pl.col("guided_success") & pl.col("unguided_success"))
        .drop_nulls(["guided_peak_rss_bytes", "unguided_peak_rss_bytes"])
        .filter((pl.col("guided_peak_rss_bytes") > 0) & (pl.col("unguided_peak_rss_bytes") > 0))
        .with_columns(
            (pl.col("guided_peak_rss_bytes") / 2**20).alias("guided_peak_mib"),
            (pl.col("unguided_peak_rss_bytes") / 2**20).alias("unguided_peak_mib"),
            (pl.col("guided_peak_rss_bytes") / pl.col("unguided_peak_rss_bytes")).alias(
                "peak_ratio"
            ),
        )
        .with_columns(((1 - pl.col("peak_ratio")) * 100).alias("memory_saved_pct"))
    )


def memory_summary(paired: pl.DataFrame) -> pl.DataFrame:
    """Peak-RSS statistics conditional on both methods succeeding."""
    if paired.is_empty():
        return pl.DataFrame(schema=MEMORY_SUMMARY_SCHEMA)
    return (
        paired.group_by("mode", maintain_order=True)
        .agg(
            pl.len().alias("n_paired_successes"),
            pl.col("guided_peak_mib").median().alias("guided_median_peak_mib"),
            pl.col("unguided_peak_mib").median().alias("unguided_median_peak_mib"),
            pl.col("guided_peak_mib").quantile(0.9).alias("guided_p90_peak_mib"),
            pl.col("unguided_peak_mib").quantile(0.9).alias("unguided_p90_peak_mib"),
            pl.col("peak_ratio").median().alias("median_peak_ratio"),
            pl.col("memory_saved_pct").median().alias("median_memory_saved_pct"),
            (pl.col("peak_ratio") < 1).mean().alias("guided_lower_peak_share"),
        )
        .with_columns(pl.exclude("mode", "n_paired_successes").round(3))
    )


def success_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """One compact success and paired-memory row per mode."""
    rates = success_rates(frame).pivot(
        on="method",
        index="mode",
        values=["successes", "n", "success_rate", "ci_low", "ci_high"],
        separator="_",
    )
    memory = memory_summary(paired_successes(frame))
    return rates.join(memory, on="mode", how="left").sort("mode")


def resolve_grid(pattern: str = "") -> Path:
    matches = [
        directory
        for directory in _run_dirs(pattern, "guided_search_grid")
        if (directory / "grid_config.json").is_file()
    ]
    if not matches:
        raise FileNotFoundError(f"No guided-search grid matching {pattern!r}")
    return matches[-1]


def grid_prefix_budgets(grid_dir: Path) -> list[int]:
    config = json.loads((grid_dir / "grid_config.json").read_text())
    return [int(value) for value in config["prefix_budgets"]]


def _grid_mode(distribution: str, strategy: str) -> str:
    return f"{_short_strategy(strategy)} · {distribution}"


def load_grid(grid_dir: Path) -> tuple[pl.DataFrame, dict]:
    """Load completed grid cells using their pair-level comparison files."""
    config = json.loads((grid_dir / "grid_config.json").read_text())
    frames = []
    loaded = set()
    for path in sorted(grid_dir.glob("distribution.*/sampling_seed.*/*/comparison.parquet")):
        run_config = json.loads((path.parent / "config.json").read_text())
        distribution = str(run_config["size_distribution"])
        sampling_seed = int(run_config["sampling_seed"])
        strategy = str(run_config["strategy"])
        frame = pl.read_parquet(path)
        _validate_comparison(frame, path)
        frames.append(
            frame.with_columns(
                pl.lit(distribution).alias("distribution"),
                pl.lit(sampling_seed).alias("sampling_seed"),
                pl.lit(strategy).alias("strategy"),
                pl.lit(_grid_mode(distribution, strategy)).alias("mode"),
                pl.concat_str(["seed", "goal"], separator="│").alias("pair"),
            )
        )
        loaded.add((distribution, sampling_seed, strategy))
    if not frames:
        raise FileNotFoundError(f"No completed comparison cells under {grid_dir}")

    expected = {
        (distribution, int(seed), strategy)
        for distribution in config["distributions_expanded"]
        for seed in config["sampling_seeds_expanded"]
        for strategy in config["strategies_expanded"]
    }
    modes = [
        _grid_mode(distribution, strategy)
        for distribution in config["distributions_expanded"]
        for strategy in config["strategies_expanded"]
    ]
    data = pl.concat(frames, how="diagonal_relaxed")
    meta = {
        "modes": modes,
        "n_pairs": data.select("seed", "goal").unique().height,
        "max_attempts": int(config["attempts"]),
        "sampling_seeds": data["sampling_seed"].n_unique(),
        "missing_cells": len(expected - loaded),
        "subtitle": [
            f"{len(loaded)}/{len(expected)} grid cells",
            f"{data['sampling_seed'].n_unique()} sampling seeds",
        ],
        "grid_dir": grid_dir,
    }
    return data, meta


def grid_success_by_budget(grid_dir: Path, budgets: Sequence[int]) -> pl.DataFrame:
    """Success observations at cumulative attempt budgets, including setup failures."""
    frames = []
    for comparison_path in sorted(
        grid_dir.glob("distribution.*/sampling_seed.*/*/comparison.parquet")
    ):
        run_dir = comparison_path.parent
        config = json.loads((run_dir / "config.json").read_text())
        distribution = str(config["size_distribution"])
        sampling_seed = int(config["sampling_seed"])
        strategy = str(config["strategy"])
        base = pl.read_parquet(comparison_path, columns=["seed", "goal"])
        attempts = pl.read_parquet(run_dir / "results.parquet")
        for budget in budgets:
            reached = (
                attempts.filter(pl.col("attempt") < budget)
                .group_by("seed", "goal")
                .agg(pl.col("reached").any().alias("guided_success"))
            )
            frames.append(
                base.join(reached, on=["seed", "goal"], how="left").with_columns(
                    pl.col("guided_success").fill_null(False),
                    pl.lit(distribution).alias("distribution"),
                    pl.lit(sampling_seed).alias("sampling_seed"),
                    pl.lit(strategy).alias("strategy"),
                    pl.lit(_grid_mode(distribution, strategy)).alias("mode"),
                    pl.lit(int(budget)).alias("budget"),
                )
            )
    observations = pl.concat(frames, how="vertical")
    return pl.DataFrame(_rate_rows(observations, ["mode", "budget"], "guided_success")).sort(
        "mode", "budget"
    )


def grid_policy_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Full-budget success and paired peak-RSS statistics by grid policy."""
    return success_summary(frame)
