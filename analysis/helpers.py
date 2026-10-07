"""Load and summarize guided peak-memory experiments against brute-force proof cost."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).parent.parent

GUIDED_WORKFLOW_COLUMN = "guided_peak_rss"
BRUTE_COLUMN = "brute_peak_rss"

# Guided peaks comparable with brute force
GUIDED_PEAK_SCOPES = {
    "guided attempt": "attempt_peak_rss",
    "guided workflow": GUIDED_WORKFLOW_COLUMN,
}

# Pair columns describing how deep the search tree went, by display name.
DEPTH_MEASURES = {"success_depth": "first success", "deepest_attempt": "deepest attempt"}

# Ordered: the drawing slot of a grouped bar is the position in this tuple.
BRUTE_COST_OUTCOMES = ("guided failed", "guided proved, at or above", "guided proved, cheaper")


@dataclass(frozen=True)
class Run:
    directory: Path
    label: str
    config: dict


def _run_dirs(pattern: str) -> list[Path]:
    base = REPO_ROOT / "data" / "guided_search"
    if not base.is_dir():
        return []
    return sorted(
        (path for path in base.iterdir() if path.is_dir() and pattern in path.name),
        key=lambda path: path.stat().st_mtime,
    )


def _budget(value: float | None) -> str:
    return "∞" if value is None else f"{value:g}"


def _run_label(directory: Path, config: dict) -> str:
    frontier = "frontier" if config["frontier"] else "whole"
    full_union = "full_union" if config["full_union"] else "simple_union"
    run_name = directory.name.split("_")[0]
    return (
        f"run_{run_name} · {config['search_policy']} · "
        f"branching={config['branching']}\n"
        f"{config['sample_policy']} · {frontier} · {full_union}\n"
        f"attempts={_budget(config['max_attempts'])} · "
        f"time={_budget(config['max_pair_time'])} · "
        f"seed={config['seed']}"
    )

    # return (
    #     f"run_{run_name} · {config['search_policy']} · "
    #     f"branching={config['branching']}\n"
    #     f"{config['sample_policy']} · {frontier} · {full_union} · "
    #     f"size_steps={config['size_search_steps']}\n"
    #     f"cap={config['max_rss']} · attempts={_budget(config['max_attempts'])} · "
    #     f"time={_budget(config['max_pair_time'])} · "
    #     f"backoff={config['sampling_backoff']} · seed={config['seed']}"
    # )


def resolve_runs(patterns: Sequence[str]) -> tuple[list[Run], list[str]]:
    """Resolve run folders under `data/guided_search` by name substring."""
    directories = (
        _run_dirs("")
        if not patterns
        else [matches[-1] for pattern in patterns if (matches := _run_dirs(pattern))]
    )
    if patterns and len(directories) != len(patterns):
        found = {path.name for path in directories}
        raise FileNotFoundError(f"Could not resolve all run patterns; found {sorted(found)}")

    runs = []
    incomplete_runs = []
    for directory in dict.fromkeys(directories):
        pairs = directory / "pairs.parquet"
        config_path = directory / "config.json"
        absent = [path.name for path in (pairs, config_path) if not path.is_file()]
        if absent:
            if patterns:
                print(f"{directory} is incomplete; missing final artifacts: {', '.join(absent)}")
            incomplete_runs.append(str(directory))
            continue
        config = json.loads(config_path.read_text())
        runs.append(Run(directory, _run_label(directory, config), config))
    if not runs:
        raise FileNotFoundError("No completed guided-search runs")
    runs.sort(key=lambda run: int(run.directory.name.split("_")[0]))
    return runs, incomplete_runs


def _brute_baseline(run: Run) -> pl.DataFrame:
    """Per-pair brute-force proof cost from the problem set the run was built on."""
    directory = REPO_ROOT / run.config["path"]
    problems = directory / "problems.json"
    if not problems.is_file():
        raise FileNotFoundError(
            f"{run.directory.name} was built on {run.config['path']}, which has no problems.json"
        )
    frame = pl.DataFrame(json.loads(problems.read_text()))
    missing = {"start", "goal", "reached", "peak_rss"} - set(frame.columns)
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
    keys = frame.select("start", "goal")
    if keys.unique().height != keys.height:
        raise ValueError(f"{problems} repeats start/goal pairs; the baseline join would fan out")

    return frame.select("start", "goal", pl.col("peak_rss").alias(BRUTE_COLUMN))


def _unguided_baseline(run: Run) -> pl.DataFrame:
    """Per-pair unguided results from the `baseline.py` folder the run was checked against."""
    directory = REPO_ROOT / run.config["baseline"]
    results = directory / "unguided.parquet"
    if not results.is_file():
        raise FileNotFoundError(
            f"{run.directory.name} was checked against {run.config['baseline']}, "
            "which has no unguided.parquet"
        )
    return pl.read_parquet(results).with_columns(pl.lit(directory.name).alias("baseline"))


def load_comparisons(runs: Sequence[Run]) -> tuple[pl.DataFrame, dict]:
    """Stack one-row-per-pair results, joining each run's unguided and brute-force baselines."""
    frames = []
    for run in runs:
        frame = pl.read_parquet(run.directory / "pairs.parquet")
        # `guided_search.py` checked that the baseline covers every pair, so a
        # miss here means the baseline changed since.
        frame = frame.join(
            _unguided_baseline(run), on=["start", "goal"], how="left", validate="1:1"
        )
        unmatched = frame.filter(pl.col("baseline").is_null()).height
        if unmatched:
            raise ValueError(
                f"{run.directory.name} has {unmatched} pairs absent from {run.config['baseline']}; "
                "the baseline changed after the run"
            )
        frame = frame.join(_brute_baseline(run), on=["start", "goal"], how="left")
        unmatched = frame.filter(pl.col(BRUTE_COLUMN).is_null()).height
        if unmatched:
            raise ValueError(
                f"{run.directory.name} has {unmatched} pairs absent from {run.config['path']}; "
                "the run and its problem set disagree"
            )
        frames.append(
            frame.with_columns(
                pl.lit(run.label).alias("mode"),
                pl.lit(run.directory.name).alias("run"),
                pl.concat_str(["start", "goal"], separator="│").alias("pair"),
            )
        )
    data = pl.concat(frames, how="diagonal_relaxed")
    meta = {
        "modes": [run.label for run in runs],
        "baselines": data["baseline"].unique().sort().to_list(),
        "subtitle": [f"{data.height} planned pair observations"],
    }
    return data, meta


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
        rows.append(
            {
                **dict(zip(group_columns, key_values, strict=True)),
                "successes": successes,
                "n": total,
                "success_rate": successes / total if total else None,
            }
        )
    return rows


def success_rates(frame: pl.DataFrame) -> pl.DataFrame:
    """Guided success rates per mode and unguided ones per baseline."""
    guided = [
        {**row, "method": "guided"}
        for row in _rate_rows(frame, ["mode", "baseline"], "guided_success")
    ]
    pairs = frame.unique(["baseline", "start", "goal"], maintain_order=True)
    unguided = [
        {**row, "mode": None, "method": "unguided"}
        for row in _rate_rows(pairs, ["baseline"], "unguided_success")
    ]
    return pl.DataFrame([*guided, *unguided])


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
        .group_by("mode", "outcome", maintain_order=True)
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
        .when(reason == "out_of_memory")
        .then(pl.lit("attempts exhausted, all oomed"))
        .when(reason == "binary_panic")
        .then(pl.lit("binary panic"))
        .when(reason == "Saturated")
        .then(pl.lit("saturated without goal"))
        # A pair whose search never ran an attempt falls back to the search's
        # own stop reason, so those land here too.
        .when(reason == "time_exhausted")
        .then(pl.lit("pair time budget"))
        .when(reason == "attempt_budget_exhausted")
        .then(pl.lit("attempt budget"))
        .when(reason == "frontier_exhausted")
        .then(pl.lit("frontier exhausted"))
        .when(reason == "unstarted")
        .then(pl.lit("search never started"))
        .otherwise(pl.lit("other"))
    )


def _setup_category(status: pl.Expr) -> pl.Expr:
    """Name why a pair never got a guide menu, from its non-ok ``setup_status``.

    ``guide menu: out of memory``
        Every `sample` try, `--sampling-backoff` retries included, died at the
        RSS cap.
    ``guide menu: no novel terms``
        The child survived the cap and still printed an empty payload: no
        novel root terms below the size cap, so there was nothing to draw.
    ``guide menu: empty pool``
        The draw succeeded but returned no samples, or every sample it
        returned had already been attempted on this pair.
    ``guide menu: binary panic``
        The `sample` process died of an uncaught panic.

    An unrecognized status simply gets passed through.
    """
    return (
        pl.when(status == "out_of_memory")
        .then(pl.lit("guide menu: out of memory"))
        .when(status == "no_novel_terms")
        .then(pl.lit("guide menu: no novel terms"))
        .when(status == "empty_pool")
        .then(pl.lit("guide menu: empty pool"))
        .when(status == "binary_panic")
        .then(pl.lit("guide menu: binary panic"))
        .otherwise(pl.lit("guide menu: ") + status)
    )


def failure_breakdown(frame: pl.DataFrame) -> pl.DataFrame:
    """Pair-level, mutually exclusive failure categories."""
    guided = frame.filter(~pl.col("guided_success").fill_null(False)).select(
        "mode",
        pl.when(pl.col("setup_status") != "ok")
        .then(_setup_category(pl.col("setup_status")))
        .when(pl.col("guided_panic").fill_null(False))
        .then(pl.lit("panic"))
        .otherwise(_stop_category(pl.col("guided_stop_reason")))
        .alias("failure"),
    )
    # unguided = frame.filter(~pl.col("unguided_success").fill_null(False)).select(
    #     "mode",
    #     pl.lit("unguided").alias("method"),
    #     pl.when(pl.col("unguided_panic").fill_null(False))
    #     .then(pl.lit("panic"))
    #     .otherwise(_stop_category(pl.col("unguided_stop_reason")))
    #     .alias("failure"),
    # )
    planned = frame.group_by("mode", maintain_order=True).agg(pl.len().alias("planned_pairs"))
    return (
        # pl.concat([guided, unguided])
        guided.group_by("mode", "failure", maintain_order=True)
        .agg(pl.len().alias("count"))
        .with_columns(pl.col("count").sum().over("mode").alias("mode_failures"))
        .join(planned, on="mode", how="left")
        .with_columns(
            (pl.col("count") / pl.col("mode_failures")).alias("share_of_failures"),
            (pl.col("count") / pl.col("planned_pairs")).alias("share_of_planned"),
        )
    )


def saturation_rates(runs: Sequence[Run]) -> pl.DataFrame:
    """How often each run's proof attempts saturated."""
    rows = []
    for run in runs:
        path = run.directory / "attempts.parquet"
        frame = pl.read_parquet(path)
        hits = int(
            frame.select((pl.col("stop_reason") == "Saturated").fill_null(False).sum()).item()
        )
        rows.append(
            {
                "mode": run.label,
                "saturated": hits,
                "n": frame.height,
                "rate": hits / frame.height if frame.height else None,
            }
        )
    return pl.DataFrame(rows)


def guided_vs_brute(frame: pl.DataFrame, scope: str) -> pl.DataFrame:
    """One `GUIDED_PEAK_SCOPES` peak against the brute-force cost of the same pair.

    Conditioned on guided success only: a guided failure has no peak to
    compare, just the budget it exhausted.
    """
    column = GUIDED_PEAK_SCOPES[scope]
    return (
        frame.filter(pl.col("guided_success"))
        .drop_nulls([column, BRUTE_COLUMN])
        .filter((pl.col(column) > 0) & (pl.col(BRUTE_COLUMN) > 0))
        .with_columns(
            pl.lit(scope).alias("guided_peak_scope"),
            (pl.col(column) / 2**20).alias("guided_peak_mib"),
            (pl.col(BRUTE_COLUMN) / 2**20).alias("brute_peak_mib"),
            (pl.col(column) / pl.col(BRUTE_COLUMN)).alias("peak_ratio"),
        )
    )


def peak_win_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """How many guided successes peak below the brute-force proof, per guided scope.

    A ratio of exactly 1 is counted as `n_at_or_above`, so the two counts
    partition `n_guided_successes`.
    """
    counts = []
    for scope in GUIDED_PEAK_SCOPES:
        comparison = guided_vs_brute(frame, scope)
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
    return pl.concat(counts).with_columns(
        (pl.col("n_below") / pl.col("n_guided_successes")).round(3).alias("share_below"),
        pl.col("median_peak_ratio").round(3),
    )


def brute_cost_by_outcome(frame: pl.DataFrame, bins: int = 14) -> pl.DataFrame:
    """Brute-force proof cost of every planned pair, binned and split by guided outcome.
    Brute force is the brute force measurement taken from `problems.json`, with
    no memory limit at all.
    Up to three outcomes exist for each bucket:
    - Not proven
    - Proven but more expensive
    - Proven and cheaper
    An outcome no pair reaches is left out entirely rather than carrying an
    empty slot in every bucket.
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
    # Share of a bin's width the bars may fill. Without the gap left at each
    # bucket's trailing edge, neighbouring buckets touch and read as one group.
    usable = 1 - 0.14
    # An outcome with no pair anywhere in the frame gives up its slot, so the
    # remaining bars widen to fill the bucket instead of leaving a gap.
    # Presence is measured over the whole frame, not per mode, to keep the slot
    # widths the same in every row of the facet.
    present = set(data["outcome"].unique().to_list())
    ordered = [name for name in BRUTE_COST_OUTCOMES if name in present]
    slots = len(ordered)
    # The slot is the fixed position in `ordered` rather than a rank among the
    # outcomes present in the bucket, so bars stay aligned across bins where
    # one outcome is empty.
    slot = pl.col("outcome").replace_strict({name: i for i, name in enumerate(ordered)})
    groups = data.group_by("mode", "outcome", maintain_order=True).agg(pl.len().alias("group_n"))
    return (
        data.with_columns(
            ((pl.col("brute_peak_mib").log10() - low) / width)
            .floor()
            .clip(0, bins - 1)
            .cast(pl.Int32)
            .alias("bin")
        )
        .group_by("mode", "outcome", "bin", maintain_order=True)
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
    )


def pairwise_solved_diff(frame: pl.DataFrame) -> pl.DataFrame:
    """Per ordered mode pair, how many shared problems the row solves and the column does not.

    Restricted to the problems both modes planned, so the two directions of a
    cell share one denominator. The diagonal is dropped.
    """
    wide = frame.select(
        "pair", "mode", pl.col("guided_success").fill_null(False).alias("solved")
    ).pivot(on="mode", index="pair", values="solved")
    modes = frame["mode"].unique(maintain_order=True).to_list()
    rows = []
    for a in modes:
        for b in modes:
            if a == b:
                continue
            shared = wide.drop_nulls([a, b])
            rows.append(
                {
                    "row_mode": a,
                    "col_mode": b,
                    "n_shared": shared.height,
                    "only_row": int((shared[a] & ~shared[b]).sum()),
                    "only_col": int((~shared[a] & shared[b]).sum()),
                    "both": int((shared[a] & shared[b]).sum()),
                    "neither": int((~shared[a] & ~shared[b]).sum()),
                }
            )
    return pl.DataFrame(rows).with_columns(
        (pl.col("only_row") - pl.col("only_col")).alias("net"),
        (pl.col("only_row") / pl.col("n_shared")).alias("share_only_row"),
    )


def depth_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """Pairs per search-tree depth: where the first success sat, and how deep any attempt went.

    A pair without a success has no `success_depth`, one without any attempt
    no `deepest_attempt`; neither is counted for that measure.
    """
    return (
        frame.unpivot(
            index="mode",
            on=list(DEPTH_MEASURES),
            variable_name="measure",
            value_name="depth",
        )
        .drop_nulls("depth")
        .with_columns(pl.col("measure").replace_strict(DEPTH_MEASURES))
        .group_by("mode", "measure", "depth", maintain_order=True)
        .len("count")
    )


def success_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """One compact success-only row per mode, next to its baseline's unguided rate."""
    rates = success_rates(frame)
    values = ["successes", "n", "success_rate"]
    guided = rates.filter(pl.col("method") == "guided").select(
        "mode", "baseline", *(pl.col(v).alias(f"guided_{v}") for v in values)
    )
    unguided = rates.filter(pl.col("method") == "unguided").select(
        "baseline", *(pl.col(v).alias(f"unguided_{v}") for v in values)
    )
    return guided.join(unguided, on="baseline", how="left")
