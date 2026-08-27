"""Load and summarize guided/unguided peak-memory experiments."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

REQUIRED_COMPARISON_COLUMNS = {
    "start_term",
    "goal_term",
    "guided_success",
    "unguided_success",
    "candidate_peak_rss_bytes",
    "verify_peak_rss_bytes",
    "guided_peak_rss_bytes",
    "unguided_peak_rss_bytes",
    "guided_peak_live_heap_bytes",
    "unguided_peak_live_heap_bytes",
    "attempts_run",
    "success_attempt",
    "setup_status",
}

MEMORY_METRICS = {
    "live_heap": {
        "label": "peak live heap",
        "guided_workflow": "guided_peak_live_heap_bytes",
        "unguided": "unguided_peak_live_heap_bytes",
        "guided_verification": "guided_peak_live_heap_bytes",
        "components": (
            ("guided workflow", "guided_peak_live_heap_bytes"),
            ("unguided verification", "unguided_peak_live_heap_bytes"),
        ),
    },
    "rss": {
        "label": "peak RSS",
        "guided_workflow": "guided_peak_rss_bytes",
        "unguided": "unguided_peak_rss_bytes",
        # The decisive leg (the one that reached, else the last tried).
        "guided_verification": "verify_peak_rss_bytes",
        "components": (
            ("candidate construction", "candidate_peak_rss_bytes"),
            ("guided verification (decisive leg)", "verify_peak_rss_bytes"),
            ("guided verification (max leg)", "verify_peak_rss_bytes_max"),
            ("guided workflow", "guided_peak_rss_bytes"),
            ("unguided verification", "unguided_peak_rss_bytes"),
        ),
    },
}
DEFAULT_MEMORY_METRIC = "rss"


def _metric(metric: str) -> dict:
    try:
        return MEMORY_METRICS[metric]
    except KeyError:
        raise ValueError(
            f"unknown memory metric {metric!r}; expected one of {sorted(MEMORY_METRICS)}"
        ) from None


MEMORY_SUMMARY_SCHEMA = {
    "mode": pl.String,
    "guided_peak_scope": pl.String,
    "memory_metric": pl.String,
    "n_paired_successes": pl.Int64,
    "guided_median_peak_mib": pl.Float64,
    "unguided_median_peak_mib": pl.Float64,
    "guided_p90_peak_mib": pl.Float64,
    "unguided_p90_peak_mib": pl.Float64,
    "median_peak_ratio": pl.Float64,
    "median_memory_saved_pct": pl.Float64,
    "guided_lower_peak_share": pl.Float64,
}
MEMORY_COMPONENT_SUMMARY_SCHEMA = {
    "mode": pl.String,
    "component": pl.String,
    "n": pl.Int64,
    "median_peak_mib": pl.Float64,
    "p90_peak_mib": pl.Float64,
    "memory_metric": pl.String,
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
    return f"{config['policy']} · memory={memory} · {directory.name}"


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
                pl.concat_str(["start_term", "goal_term"], separator="│").alias("pair"),
            )
        )
    data = pl.concat(frames, how="diagonal_relaxed")
    meta = {
        "modes": [run.label for run in runs],
        "n_pairs": data.select("start_term", "goal_term").unique().height,
        "subtitle": [f"{data.height} planned pair observations"],
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
    reason. Unguided runs have no guide-candidate construction stage.
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


def _paired_successes_for_peak(
    frame: pl.DataFrame,
    guided_peak_column: str,
    guided_peak_scope: str,
    unguided_peak_column: str,
    metric_label: str,
) -> pl.DataFrame:
    """Build a guided/unguided comparison for one explicitly named guided peak."""
    return (
        frame.filter(pl.col("guided_success") & pl.col("unguided_success"))
        .drop_nulls([guided_peak_column, unguided_peak_column])
        .filter((pl.col(guided_peak_column) > 0) & (pl.col(unguided_peak_column) > 0))
        .with_columns(
            pl.lit(guided_peak_scope).alias("guided_peak_scope"),
            pl.lit(metric_label).alias("memory_metric"),
            (pl.col(guided_peak_column) / 2**20).alias("guided_peak_mib"),
            (pl.col(unguided_peak_column) / 2**20).alias("unguided_peak_mib"),
            (pl.col(guided_peak_column) / pl.col(unguided_peak_column)).alias("peak_ratio"),
        )
        .with_columns(((1 - pl.col("peak_ratio")) * 100).alias("memory_saved_pct"))
    )


def paired_verification_successes(
    frame: pl.DataFrame, metric: str = DEFAULT_MEMORY_METRIC
) -> pl.DataFrame:
    """Paired successes comparing guided verification with unguided verification."""
    spec = _metric(metric)
    scope = (
        "guided verification"
        if spec["guided_verification"] != spec["guided_workflow"]
        else "guided workflow"
    )
    return _paired_successes_for_peak(
        frame, spec["guided_verification"], scope, spec["unguided"], spec["label"]
    )


def paired_workflow_successes(
    frame: pl.DataFrame, metric: str = DEFAULT_MEMORY_METRIC
) -> pl.DataFrame:
    """Paired successes comparing the complete guided workflow with unguided verification."""
    spec = _metric(metric)
    return _paired_successes_for_peak(
        frame, spec["guided_workflow"], "guided workflow", spec["unguided"], spec["label"]
    )


def paired_successes(frame: pl.DataFrame, metric: str = DEFAULT_MEMORY_METRIC) -> pl.DataFrame:
    """Complete-workflow comparison for grid analyses."""
    return paired_workflow_successes(frame, metric)


def memory_component_summary(
    frame: pl.DataFrame, metric: str = DEFAULT_MEMORY_METRIC
) -> pl.DataFrame:
    """Per-phase peak memory for the selected metric."""
    paired = frame.filter(pl.col("guided_success") & pl.col("unguided_success"))
    spec = _metric(metric)
    components = tuple(
        (component, column) for component, column in spec["components"] if column in frame.columns
    )
    summaries = []
    for component, column in components:
        valid = paired.drop_nulls(column).filter(pl.col(column) > 0)
        if valid.is_empty():
            summaries.append(
                paired.select("mode")
                .unique(maintain_order=True)
                .with_columns(
                    pl.lit(component).alias("component"),
                    pl.lit(0, dtype=pl.UInt32).alias("n"),
                    pl.lit(None, dtype=pl.Float64).alias("median_peak_mib"),
                    pl.lit(None, dtype=pl.Float64).alias("p90_peak_mib"),
                )
            )
            continue
        summaries.append(
            valid.group_by("mode", maintain_order=True).agg(
                pl.lit(component).first().alias("component"),
                pl.len().alias("n"),
                (pl.col(column) / 2**20).median().alias("median_peak_mib"),
                (pl.col(column) / 2**20).quantile(0.9).alias("p90_peak_mib"),
            )
        )
    if not summaries:
        return pl.DataFrame(schema=MEMORY_COMPONENT_SUMMARY_SCHEMA)
    return (
        pl.concat(summaries)
        .with_columns(
            pl.col("median_peak_mib", "p90_peak_mib").round(3),
            pl.lit(spec["label"]).alias("memory_metric"),
        )
        .sort("mode", "component")
    )


def memory_summary(paired: pl.DataFrame) -> pl.DataFrame:
    """Paired peak-memory statistics conditional on both methods succeeding."""
    if paired.is_empty():
        return pl.DataFrame(schema=MEMORY_SUMMARY_SCHEMA)
    if "guided_peak_scope" not in paired.columns:
        paired = paired.with_columns(pl.lit("guided workflow").alias("guided_peak_scope"))
    if "memory_metric" not in paired.columns:
        paired = paired.with_columns(pl.lit("unknown").alias("memory_metric"))
    return (
        paired.group_by("mode", "guided_peak_scope", "memory_metric", maintain_order=True)
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
        .with_columns(
            pl.exclude("mode", "guided_peak_scope", "memory_metric", "n_paired_successes").round(3)
        )
    )


def success_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """One compact success-only row per mode."""
    return (
        success_rates(frame)
        .pivot(
            on="method",
            index="mode",
            values=["successes", "n", "success_rate", "ci_low", "ci_high"],
            separator="_",
        )
        .sort("mode")
    )


def problem_pairs(pattern: str = "") -> pl.DataFrame:
    """Load the `problems.json` a run was built from, for provenance.

    Carries each pair's generation-time unguided measurement (`peak_rss_bytes`
    above `--min-rss` is why the pair was kept).
    """
    base = _data_dir("problems")
    matches = [
        directory
        for directory in _run_dirs(pattern, "problems")
        if (directory / "problems.json").is_file()
    ]
    if not matches:
        raise FileNotFoundError(f"No problem folder with problems.json under {base}")
    directory = matches[-1]
    return pl.DataFrame(json.loads((directory / "problems.json").read_text())).with_columns(
        pl.lit(directory.name).alias("problem_set")
    )
