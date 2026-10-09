"""Load and summarize guided peak-memory experiments against brute-force proof cost."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).parent.parent

BRUTE_COLUMN = "brute_peak_rss"

# Guided peaks comparable with brute force
GUIDED_PEAK_SCOPES = {
    "guided attempt": "attempt_peak_rss",
    "guided workflow": "guided_peak_rss",
}

# Pair columns describing how deep the search tree went, by display name.
DEPTH_MEASURES = {"success_depth": "first success", "deepest_attempt": "deepest attempt"}

# Ordered: the drawing slot of a grouped bar is the position in this tuple.
BRUTE_COST_OUTCOMES = ("guided failed", "guided proved")

# Whether a failed attempt's pair was proved by some attempt, in legend order.
PAIR_OUTCOMES = ("eventually proved", "never proved")


# `_settings` keys drawn as the grid's rows; the seed is pooled, and every
# other varying setting names the grid's columns.
STRATEGY = "strategy"
SEED = "seed"

# Grid cell of a run: seeds are replicates of one configuration, so every
# aggregate pools them and only `pairwise_solved_diff` keeps runs apart.
GROUP = [STRATEGY, "setting"]


@dataclass(frozen=True)
class Run:
    directory: Path
    label: str
    config: dict
    strategy: str
    setting: str
    # The setting and, if it varies, the seed: what tells the run apart within its strategy.
    short: str


def _run_dirs(folder: str, pattern: str) -> list[Path]:
    base = REPO_ROOT / "data" / folder
    if not base.is_dir():
        return []
    return sorted(
        (path for path in base.iterdir() if path.is_dir() and pattern in path.name),
        key=lambda path: path.stat().st_mtime,
    )


def _settings(config: dict) -> dict[str, str]:
    """Display value of every setting a run can differ in, the strategy first and the seed last."""
    budget = {
        key: "∞" if config[key] is None else f"{config[key]:g}"
        for key in ("max_attempts", "max_pair_time")
    }
    frontier = "frontier" if config["frontier"] else "whole"
    return {
        STRATEGY: f"{config['search_policy']} · {frontier}",
        "branching": f"branching={config['branching']}",
        "sampling": config["sample_policy"],
        "union": "full_union" if config["full_union"] else "simple_union",
        "attempts": f"attempts={budget['max_attempts']}",
        "time": f"time={budget['max_pair_time']}",
        SEED: f"seed={config['seed']}",
    }


def _varying(configs: Sequence[dict]) -> tuple[list[str], list[str]]:
    """`_settings` keys that differ between `configs`, and the values of those that do not."""
    settings = [_settings(config) for config in configs]
    keys = list(settings[0])
    varying = [key for key in keys if len({s[key] for s in settings}) > 1]
    constant = [settings[0][key] for key in keys if key not in varying]
    return varying, constant


def _resolve_runs(
    patterns: Sequence[str], folder: str
) -> tuple[list[Run], list[str], list[str], list[str]]:
    """Resolve run folders under `data/<folder>` by name substring.

    Also returns the incomplete run folders, and `_varying` over the complete ones.
    """
    directories = (
        _run_dirs(folder, "")
        if not patterns
        else [matches[-1] for pattern in patterns if (matches := _run_dirs(folder, pattern))]
    )
    if patterns and len(directories) != len(patterns):
        found = {path.name for path in directories}
        raise FileNotFoundError(f"Could not resolve all run patterns; found {sorted(found)}")

    complete = []
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
        complete.append((directory, json.loads(config_path.read_text())))
    if not complete:
        raise FileNotFoundError(f"No completed guided-search runs in data/{folder}")
    complete.sort(key=lambda run: int(run[0].name.split("_")[0]))

    # Labels name only what tells the selected runs apart; `load_comparisons`
    # puts the rest in the subtitle.
    varying, constant = _varying([config for _, config in complete])
    columns = [key for key in varying if key not in (STRATEGY, SEED)]
    runs = []
    for directory, config in complete:
        settings = _settings(config)
        setting = " · ".join(settings[key] for key in columns) or "all runs"
        name = f"run_{directory.name.split('_')[0]}"
        short = f"{setting} · {settings[SEED]}" if SEED in varying else setting
        label = f"{name} · {settings[STRATEGY]}\n{short}"
        runs.append(Run(directory, label, config, settings[STRATEGY], setting, short))
    return runs, incomplete_runs, varying, constant


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


def load_comparisons(
    patterns: Sequence[str], folder: str = "guided_search"
) -> tuple[list[Run], list[str], pl.DataFrame, dict]:
    """Stack the one-row-per-pair results of the runs `patterns` select in `data/<folder>`.

    Joins each run's unguided and brute-force baselines. Returns the runs, the
    incomplete run folders that were skipped, the stacked results, and the
    chart metadata.
    """
    runs, incomplete, varying, constant = _resolve_runs(patterns, folder)
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
                pl.lit(run.strategy).alias("strategy"),
                pl.lit(run.setting).alias("setting"),
                pl.lit(run.short).alias("short"),
                pl.lit(run.directory.name).alias("run"),
                pl.concat_str(["start", "goal"], separator="│").alias("pair"),
            )
        )
    data = pl.concat(frames, how="diagonal_relaxed")
    seeds = ", ".join(dict.fromkeys(str(run.config["seed"]) for run in runs))
    settings = list(dict.fromkeys(run.setting for run in runs))
    meta = {
        "strategies": list(dict.fromkeys(run.strategy for run in runs)),
        "settings": settings,
        # By setting, then by run, which puts a setting's seeds next to each other.
        "shorts": list(
            dict.fromkeys(
                run.short for run in sorted(runs, key=lambda run: settings.index(run.setting))
            )
        ),
        "subtitle": [
            s for s in (" · ".join(constant), f"{data.height} planned pair observations") if s
        ],
        # Appended to the subtitle of every chart that pools the seeds.
        "pooled": f"seeds {seeds} pooled" if SEED in varying else None,
    }
    return runs, incomplete, data, meta


def success_rates(frame: pl.DataFrame) -> pl.DataFrame:
    """Guided and unguided success rates per grid cell, next to the cell's baseline."""
    return pl.concat(
        frame.group_by(GROUP, maintain_order=True).agg(
            pl.col(f"{method}_success").sum().alias("successes"),
            pl.len().alias("n"),
            pl.lit(method).alias("method"),
            pl.first("baseline"),
        )
        for method in ("guided", "unguided")
    ).with_columns((pl.col("successes") / pl.col("n")).alias("success_rate"))


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
        .group_by(*GROUP, "outcome", maintain_order=True)
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum().over(GROUP)).alias("share"))
    )


# Display names of an attempt's `stop_reason`, keyed by egg's variant name; unknown ones are "other".
STOP_LABELS = {
    "unknown": "unknown",
    "NodeLimit": "node limit",
    "MemoryLimit": "memory limit",
    "TimeLimit": "time limit",
    "IterationLimit": "iteration limit",
    "Saturated": "saturated without goal",
    "out_of_memory": "out of memory",
    "binary_panic": "binary panic",
    "timeout": "hard timeout",
    "panic": "panic in eqsat",
    "arg_too_long": "guide too long for argv",
}


def _stop_category(reason: pl.Expr) -> pl.Expr:
    """Collapse detailed egg stop strings into stable analysis categories.

    egg's limits render with their value, e.g. `TimeLimit(300.08)`, so only the
    variant name before the parenthesis is looked up.
    """
    return (
        reason.fill_null("unknown")
        .str.extract(r"^(\w+)")
        .replace_strict(STOP_LABELS, default="other")
    )


# Display names of `search_stop_reason`; `reached` and unknown reasons pass through.
SEARCH_LABELS = {
    "time_exhausted": "pair time budget",
    "attempt_budget_exhausted": "attempt budget",
    "frontier_exhausted": "frontier exhausted",
    "unstarted": "search never started",
}


# Why a pair never got a guide menu, by its non-ok `setup_status`; unknown ones pass through.
SETUP_LABELS = {
    # Every `sample` try, `--sampling-backoff` retries included, died at the RSS cap.
    "out_of_memory": "out of memory",
    # The child survived the cap and still printed an empty payload: no novel
    # root terms below the size cap, so there was nothing to draw.
    "no_novel_terms": "no novel terms",
    # The draw succeeded but returned no samples, or every sample it returned
    # had already been attempted on this pair.
    "empty_pool": "empty pool",
    # The `sample` process died of an uncaught panic.
    "binary_panic": "binary panic",
    # The `sample` process outran `max_pair_time` plus the grace and was killed.
    "timeout": "hard timeout",
    # The start term exceeds the kernel's per-argument limit, so `sample` was never spawned.
    "arg_too_long": "start too long for argv",
}


def search_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    """How every pair's search ended, one mutually exclusive category per pair.

    A missing guide menu takes precedence; its search ends `frontier_exhausted`
    without ever running an attempt.
    """
    return (
        frame.select(
            *GROUP,
            pl.when(pl.col("setup_status") != "ok")
            .then("guide menu: " + pl.col("setup_status").replace(SETUP_LABELS))
            .otherwise(pl.col("search_stop_reason").replace(SEARCH_LABELS))
            .alias("outcome"),
        )
        .group_by(*GROUP, "outcome", maintain_order=True)
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum().over(GROUP)).alias("share"))
    )


def attempt_failures(runs: Sequence[Run]) -> pl.DataFrame:
    """Why each run's failed attempts stopped, split by whether their pair was eventually proved."""
    reached = pl.col("reached").fill_null(False)
    attempts = pl.concat(
        pl.read_parquet(run.directory / "attempts.parquet").with_columns(
            pl.lit(run.directory.name).alias("run"),
            pl.lit(run.strategy).alias("strategy"),
            pl.lit(run.setting).alias("setting"),
        )
        for run in runs
    )
    return (
        attempts.with_columns(
            pl.when(reached.any().over("run", "start", "goal"))
            .then(pl.lit(PAIR_OUTCOMES[0]))
            .otherwise(pl.lit(PAIR_OUTCOMES[1]))
            .alias("pair_outcome"),
        )
        .filter(~reached)
        .group_by(
            *GROUP,
            "pair_outcome",
            _stop_category(pl.col("stop_reason")).alias("failure"),
            maintain_order=True,
        )
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum().over(*GROUP, "pair_outcome")).alias("share")
        )
    )


def guided_vs_brute(frame: pl.DataFrame, scope: str) -> pl.DataFrame:
    """One `GUIDED_PEAK_SCOPES` peak against the brute-force cost of the same pair.

    Conditioned on guided success only: a guided failure has no peak to
    compare, just the budget it exhausted.
    """
    column = GUIDED_PEAK_SCOPES[scope]
    return frame.filter(pl.col("guided_success")).with_columns(
        pl.lit(scope).alias("guided_peak_scope"),
        (pl.col(column) / 2**20).alias("guided_peak_mib"),
        (pl.col(BRUTE_COLUMN) / 2**20).alias("brute_peak_mib"),
        (pl.col(column) / pl.col(BRUTE_COLUMN)).alias("peak_ratio"),
    )


def brute_cost_by_outcome(frame: pl.DataFrame, bins: int = 14) -> pl.DataFrame:
    """Brute-force proof cost of every planned pair, binned and split by guided outcome.

    Brute force is the brute force measurement taken from `problems.json`, with
    no memory limit at all.
    """
    data = frame.select(
        *GROUP,
        pl.when(pl.col("guided_success"))
        .then(pl.lit(BRUTE_COST_OUTCOMES[1]))
        .otherwise(pl.lit(BRUTE_COST_OUTCOMES[0]))
        .alias("outcome"),
        (pl.col(BRUTE_COLUMN) / 2**20).alias("brute_peak_mib"),
    )
    span = data.select(
        pl.col("brute_peak_mib").log10().min().alias("low"),
        pl.col("brute_peak_mib").log10().max().alias("high"),
    ).row(0, named=True)
    low = span["low"]
    width = (span["high"] - low) / bins
    # Share of a bin's width the bars may fill. Without the gap left at each
    # bucket's trailing edge, neighbouring buckets touch and read as one group.
    usable = 1 - 0.14
    slots = len(BRUTE_COST_OUTCOMES)
    # The slot is the fixed position in `BRUTE_COST_OUTCOMES` rather than a rank
    # among the outcomes present in the bucket, so bars stay aligned across bins
    # where one outcome is empty.
    slot = pl.col("outcome").replace_strict({name: i for i, name in enumerate(BRUTE_COST_OUTCOMES)})
    groups = data.group_by(*GROUP, "outcome", maintain_order=True).agg(pl.len().alias("group_n"))
    return (
        data.with_columns(
            ((pl.col("brute_peak_mib").log10() - low) / width)
            .floor()
            .clip(0, bins - 1)
            .cast(pl.Int32)
            .alias("bin")
        )
        .group_by(*GROUP, "outcome", "bin", maintain_order=True)
        .agg(pl.len().alias("count"))
        .join(groups, on=[*GROUP, "outcome"], how="left")
        .with_columns(
            (10 ** (low + pl.col("bin") * width)).alias("bin_start_mib"),
            (10 ** (low + (pl.col("bin") + 1) * width)).alias("bin_end_mib"),
            (10 ** (low + (pl.col("bin") + usable * slot / slots) * width)).alias("slot_start_mib"),
            (10 ** (low + (pl.col("bin") + usable * (slot + 1) / slots) * width)).alias(
                "slot_end_mib"
            ),
            (pl.col("count") / pl.col("group_n")).alias("share"),
            pl.col("count").sum().over(*GROUP, "bin").alias("bucket_n"),
        )
        .with_columns((pl.col("count") / pl.col("bucket_n")).alias("bucket_share"))
    )


def pairwise_solved_diff(frame: pl.DataFrame) -> pl.DataFrame:
    """Per ordered mode pair, how many problems the row solves and the column does not.

    Every run plans the same problems. The diagonal is dropped.
    """
    wide = frame.select("pair", "mode", pl.col("guided_success").alias("solved")).pivot(
        on="mode", index="pair", values="solved"
    )
    modes = frame["mode"].unique(maintain_order=True).to_list()
    rows = []
    for a in modes:
        for b in modes:
            if a == b:
                continue
            rows.append(
                {
                    "row_mode": a,
                    "col_mode": b,
                    "n_shared": wide.height,
                    "only_row": int((wide[a] & ~wide[b]).sum()),
                    "only_col": int((~wide[a] & wide[b]).sum()),
                    "both": int((wide[a] & wide[b]).sum()),
                    "neither": int((~wide[a] & ~wide[b]).sum()),
                }
            )
    runs = frame.select("mode", "strategy", "short").unique("mode", maintain_order=True)
    return (
        pl.DataFrame(rows)
        .with_columns(
            (pl.col("only_row") - pl.col("only_col")).alias("net"),
            (pl.col("n_shared") - pl.col("neither")).alias("combined"),
            (pl.col("only_row") / pl.col("n_shared")).alias("share_only_row"),
        )
        .join(runs.select(pl.all().name.prefix("row_")), on="row_mode", how="left")
        .join(runs.select(pl.all().name.prefix("col_")), on="col_mode", how="left")
    )


def depth_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """Pairs per search-tree depth: where the first success sat, and how deep any attempt went.

    A pair without a success has no `success_depth`, one without any attempt
    no `deepest_attempt`; neither is counted for that measure.
    """
    return (
        frame.unpivot(
            index=GROUP, on=list(DEPTH_MEASURES), variable_name="measure", value_name="depth"
        )
        .drop_nulls("depth")
        .with_columns(pl.col("measure").replace_strict(DEPTH_MEASURES))
        .group_by(*GROUP, "measure", "depth", maintain_order=True)
        .len("count")
    )


def success_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """One compact success row per grid cell, next to its baseline's unguided rate.

    Also the median ratio of guided to brute-force peak over the cell's
    successes, per `GUIDED_PEAK_SCOPES` scope.
    """
    rates = success_rates(frame)
    values = ["successes", "n", "success_rate"]
    guided = rates.filter(pl.col("method") == "guided").select(
        *GROUP, "baseline", *(pl.col(v).alias(f"guided_{v}") for v in values)
    )
    unguided = rates.filter(pl.col("method") == "unguided").select(
        *GROUP, *(pl.col(v).alias(f"unguided_{v}") for v in values)
    )
    summary = guided.join(unguided, on=GROUP, how="left")
    for scope in GUIDED_PEAK_SCOPES:
        ratio = (
            guided_vs_brute(frame, scope)
            .group_by(GROUP)
            .agg(
                pl.col("peak_ratio")
                .median()
                .round(3)
                .alias(f"median_{scope.split()[-1]}_peak_ratio")
            )
        )
        summary = summary.join(ratio, on=GROUP, how="left")
    return summary
