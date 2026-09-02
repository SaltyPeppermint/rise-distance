"""Load and summarize guided peak-memory experiments against brute-force proof cost."""

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
    "attempts_run",
    "success_attempt",
    "setup_status",
}

BRUTE_COLUMNS = {"peak_rss_bytes": "brute_peak_rss_bytes"}

GUIDED_WORKFLOW_COLUMN = "guided_peak_rss_bytes"
GUIDED_VERIFICATION_COLUMN = "verify_peak_rss_bytes"
BRUTE_COLUMN = "brute_peak_rss_bytes"

PEAK_WIN_SCHEMA = {
    "mode": pl.String,
    "guided_peak_scope": pl.String,
    "n_guided_successes": pl.Int64,
    "n_below": pl.UInt32,
    "n_at_or_above": pl.UInt32,
    "share_below": pl.Float64,
    "median_peak_ratio": pl.Float64,
}

# Share of a bin's width left empty, so grouped bars separate into buckets.
BRUTE_COST_BIN_PAD = 0.14
# Ordered: the drawing slot of a grouped bar is the position in this tuple.
BRUTE_COST_OUTCOMES = ("guided failed", "guided proved, at or above", "guided proved, cheaper")


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
    # Absent in runs predating `--sampling-rss-max`.
    cap = config.get("sampling_rss_max")
    capped = f" · sampling cap={cap}" if cap else ""
    return f"{config['policy']} · memory={memory}{capped} · {directory.name}"


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


def _brute_baseline(run: Run) -> pl.DataFrame:
    """Per-pair brute-force proof cost from the problem set the run was built on."""
    directory = Path(__file__).parent / ".." / run.config["path"]
    problems = directory / "problems.json"
    if not problems.is_file():
        raise FileNotFoundError(
            f"{run.directory.name} was built on {run.config['path']}, which has no problems.json"
        )
    frame = pl.DataFrame(json.loads(problems.read_text()))
    missing = ({"start_term", "goal_term", "reached"} | set(BRUTE_COLUMNS)) - set(frame.columns)
    if missing:
        raise ValueError(f"{problems} is missing baseline fields: {sorted(missing)}")

    # A pair that never reached the goal has no proof cost, only a censored
    # lower bound, so it cannot serve as a baseline.
    unreached = frame.filter(~pl.col("reached").fill_null(False)).height
    if unreached:
        raise ValueError(
            f"{problems} has {unreached} pairs that never reached the goal; "
            "the brute-force baseline is only defined for proven pairs"
        )
    keys = frame.select("start_term", "goal_term")
    if keys.unique().height != keys.height:
        raise ValueError(f"{problems} repeats start/goal pairs; the baseline join would fan out")

    return frame.select(
        "start_term",
        "goal_term",
        *(pl.col(source).alias(target) for source, target in BRUTE_COLUMNS.items()),
        pl.lit(directory.name).alias("problem_set"),
    )


def load_comparisons(runs: Sequence[Run]) -> tuple[pl.DataFrame, dict]:
    """Stack one-row-per-pair comparison files, joining each run's brute-force baseline."""
    frames = []
    for run in runs:
        frame = pl.read_parquet(run.directory / "comparison.parquet")
        _validate_comparison(frame, run.directory)
        frame = frame.join(_brute_baseline(run), on=["start_term", "goal_term"], how="left")
        unmatched = frame.filter(pl.col("brute_peak_rss_bytes").is_null()).height
        if unmatched:
            raise ValueError(
                f"{run.directory.name} has {unmatched} pairs absent from {run.config['path']}; "
                "the run and its problem set disagree"
            )
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
        "problem_sets": data["problem_set"].unique().sort().to_list(),
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
        .when(reason == "rss_killed")
        .then(pl.lit("rss cap kill"))
        .when(reason == "Saturated")
        .then(pl.lit("saturated without goal"))
        .otherwise(pl.lit("other"))
    )


def failure_breakdown(frame: pl.DataFrame) -> pl.DataFrame:
    """Pair-level, mutually exclusive failure categories for both methods.

    Guided setup failures take precedence over any panic observed in the
    attempt workflow; panic then takes precedence over the terminal stop
    reason. Unguided runs have no guide-candidate construction stage.

    A "setup failure" is a non-ok ``setup_status``: the guide-candidate menu
    could not be built, so the pair never ran a `verify` leg. Since candidate
    construction runs once per start term, one failed start term marks all of
    its goal pairs, and the count is inflated relative to distinct events.

    The status is kept in the label (``setup failure: failed`` when the
    `candidates` subprocess returned an empty payload, ``setup failure:
    empty_pool`` when a menu was built but no attempt ran) rather than
    collapsed, so a new status shows up as its own row instead of silently
    joining an existing bucket.
    """
    guided = frame.filter(~pl.col("guided_success").fill_null(False)).select(
        "mode",
        pl.lit("guided").alias("method"),
        pl.when(pl.col("setup_status") != "ok")
        .then(pl.lit("setup failure: ") + pl.col("setup_status"))
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


def _guided_vs_brute(
    frame: pl.DataFrame, guided_peak_column: str, guided_peak_scope: str
) -> pl.DataFrame:
    """Compare one explicitly named guided peak with the brute-force proof cost.

    Conditioned on guided success only: a guided failure has no peak to
    compare, just the budget it exhausted. The brute-force baseline exists for
    every pair, so nothing is dropped on its account.
    """
    return (
        frame.filter(pl.col("guided_success"))
        .drop_nulls([guided_peak_column, BRUTE_COLUMN])
        .filter((pl.col(guided_peak_column) > 0) & (pl.col(BRUTE_COLUMN) > 0))
        .with_columns(
            pl.lit(guided_peak_scope).alias("guided_peak_scope"),
            (pl.col(guided_peak_column) / 2**20).alias("guided_peak_mib"),
            (pl.col(BRUTE_COLUMN) / 2**20).alias("brute_peak_mib"),
            (pl.col(guided_peak_column) / pl.col(BRUTE_COLUMN)).alias("peak_ratio"),
        )
        .with_columns(((1 - pl.col("peak_ratio")) * 100).alias("memory_saved_pct"))
    )


def verification_vs_brute(frame: pl.DataFrame) -> pl.DataFrame:
    """Guided verification versus the brute-force proof cost for the same pair."""
    return _guided_vs_brute(frame, GUIDED_VERIFICATION_COLUMN, "guided verification")


def workflow_vs_brute(frame: pl.DataFrame) -> pl.DataFrame:
    """The complete guided workflow versus the brute-force proof cost for the same pair."""
    return _guided_vs_brute(frame, GUIDED_WORKFLOW_COLUMN, "guided workflow")


def peak_win_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """How many guided successes peak below the brute-force proof, per guided scope.

    A ratio of exactly 1 is counted as `n_at_or_above`, so the two counts
    partition `n_guided_successes`.
    """
    counts = []
    for builder in (verification_vs_brute, workflow_vs_brute):
        comparison = builder(frame)
        if comparison.is_empty():
            continue
        counts.append(
            comparison.group_by("mode", "guided_peak_scope", maintain_order=True).agg(
                pl.len().alias("n_guided_successes"),
                (pl.col("peak_ratio") < 1).sum().alias("n_below"),
                (pl.col("peak_ratio") >= 1).sum().alias("n_at_or_above"),
                pl.col("peak_ratio").median().alias("median_peak_ratio"),
            )
        )
    if not counts:
        return pl.DataFrame(schema=PEAK_WIN_SCHEMA)
    return (
        pl.concat(counts)
        .with_columns(
            (pl.col("n_below") / pl.col("n_guided_successes")).round(3).alias("share_below"),
            pl.col("median_peak_ratio").round(3),
        )
        .sort("mode", "guided_peak_scope")
    )


def brute_cost_by_outcome(frame: pl.DataFrame, bins: int = 14) -> pl.DataFrame:
    """Brute-force proof cost of every planned pair, binned and split by guided outcome.

    Brute force is the brute force measurement taken from `problems.json`, with
    no memory limit at all.

    The three outcome exist for each bucket:
    - Not proven
    - Proven but more expensive
    - Proven and cheaper
    A success whose workflow peak was not recorded counts as at or above.
    """
    guided = pl.col(GUIDED_WORKFLOW_COLUMN)
    cheaper = (guided > 0) & (guided < pl.col(BRUTE_COLUMN))
    data = (
        frame.drop_nulls(BRUTE_COLUMN)
        .filter(pl.col(BRUTE_COLUMN) > 0)
        .select(
            "mode",
            pl.when(~pl.col("guided_success").fill_null(False))
            .then(pl.lit(BRUTE_COST_OUTCOMES[0]))
            .when(cheaper)
            .then(pl.lit(BRUTE_COST_OUTCOMES[2]))
            .otherwise(pl.lit(BRUTE_COST_OUTCOMES[1]))
            .alias("outcome"),
            (pl.col(BRUTE_COLUMN) / 2**20).alias("brute_peak_mib"),
        )
    )
    if data.is_empty():
        raise ValueError(f"no pair carries a positive {BRUTE_COLUMN}")

    span = data.select(
        pl.col("brute_peak_mib").log10().min().alias("low"),
        pl.col("brute_peak_mib").log10().max().alias("high"),
    ).row(0, named=True)
    low, high = span["low"], span["high"]
    # A single distinct cost leaves no range to divide; give it one unit bin.
    width = (high - low) / bins if high > low else 1.0
    # Bars run to the bin edge without it, so neighbouring buckets touch and
    # read as one group; this reserves a gap at each bucket's trailing edge.
    usable = 1 - BRUTE_COST_BIN_PAD
    slots = len(BRUTE_COST_OUTCOMES)
    # The slot is the fixed position in BRUTE_COST_OUTCOMES rather than a rank
    # among the outcomes present, so bars stay aligned across bins where one
    # outcome is empty.
    slot = pl.col("outcome").replace_strict({name: i for i, name in enumerate(BRUTE_COST_OUTCOMES)})
    groups = data.group_by("mode", "outcome").agg(
        pl.len().alias("group_n"),
        pl.col("brute_peak_mib").median().alias("group_median_mib"),
    )
    return (
        data.with_columns(
            ((pl.col("brute_peak_mib").log10() - low) / width)
            .floor()
            .clip(0, bins - 1)
            .cast(pl.Int32)
            .alias("bin")
        )
        .group_by("mode", "outcome", "bin")
        .agg(pl.len().alias("count"))
        .join(groups, on=["mode", "outcome"], how="left")
        .with_columns(
            (10 ** (low + pl.col("bin") * width)).alias("bin_start_mib"),
            (10 ** (low + (pl.col("bin") + 1) * width)).alias("bin_end_mib"),
            (10 ** (low + (pl.col("bin") + usable * slot / slots) * width)).alias("slot_start_mib"),
            (10 ** (low + (pl.col("bin") + usable * (slot + 1) / slots) * width)).alias(
                "slot_end_mib"
            ),
            (pl.col("count") / pl.col("group_n")).alias("share"),
            pl.col("count").sum().over("mode", "bin").alias("bucket_n"),
        )
        .with_columns((pl.col("count") / pl.col("bucket_n")).alias("bucket_share"))
        .sort("mode", "bin", "outcome")
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

    Carries each pair's brute-force proof measurement (`peak_rss_bytes` above
    `--min-rss` is why the pair was kept). `load_comparisons` already joins the
    columns the memory statistics need; this is for inspecting the rest.
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
