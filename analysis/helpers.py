"""Load and summarize guided peak-memory experiments against brute-force proof cost."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).parent.parent

BRUTE_COLUMN = "brute_peak_rss"

GUIDED_PEAK_SCOPES = {
    "guided attempt": "attempt_peak_rss",
    "guided workflow": "guided_peak_rss",
}

MEASURES = {
    "attempts": {"success_attempt": "first success", "attempts_run": "whole search"},
    "depth": {"success_depth": "first success", "deepest_attempt": "whole search"},
    "time": {"success_time": "first success", "search_time": "whole search"},
}
MEASURE_ORDER = ("first success", "whole search")

# Ordered: tuple position sets grouped-bar slots and legend order.
BRUTE_COST_OUTCOMES = ("guided failed", "guided proved")
PAIR_OUTCOMES = ("eventually proved", "never proved")

STRATEGY = "strategy"
SEED = "seed"
# Grid cell of a run: seeds are pooled as replicates.
GROUP = (STRATEGY, "setting")


@dataclass(frozen=True)
class Run:
    directory: Path
    label: str
    config: dict
    strategy: str
    setting: str
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
    """Display values of every setting a run can vary in (strategy first, seed last)."""
    budget = {
        key: "∞" if config[key] is None else f"{config[key]:g}"
        for key in ("max_attempts", "max_pair_time", "max_attempt_time")
    }
    frontier = "frontier" if config["frontier"] else "whole"
    return {
        STRATEGY: f"{config['search_policy']} · {frontier}",
        "tree_width": f"tree_width={config['tree_width']}",
        "sampling": config["sample_policy"],
        "backoff": f"backoff={config['sampling_backoff']}",
        "guides": f"guides={config['guides_per_attempt']}",
        "union": "full_union" if config["full_union"] else "simple_union",
        "attempts": f"attempts={budget['max_attempts']}",
        "time": f"time={budget['max_pair_time']}",
        "attempt_time": f"attempt_time={budget['max_attempt_time']}",
        SEED: f"seed={config['seed']}",
    }


def _matches(directory: Path, config: dict, where: Mapping[str, Any]) -> bool:
    """Whether `config` matches every `where` value (list/tuple allows any entry)."""
    for key, allowed in where.items():
        if key not in config:
            raise KeyError(f"{directory.name}/config.json has no {key!r} to filter on")
        if config[key] not in (allowed if isinstance(allowed, (list, tuple)) else (allowed,)):
            return False
    return True


def _resolve_runs(
    patterns: Sequence[str], folder: str, where: Mapping[str, Any]
) -> tuple[list[Run], list[str], list[str], list[str]]:
    """Resolve completed run folders under `data/<folder>`, filtered by `where`."""
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
        config = json.loads(config_path.read_text())
        if _matches(directory, config, where):
            complete.append((directory, config))
    if not complete:
        raise FileNotFoundError(
            f"No completed guided-search runs in data/{folder}"
            + (f" match {dict(where)}" if where else "")
        )
    complete.sort(key=lambda run: int(run[0].name.split("_")[0]))

    all_settings = [_settings(config) for _, config in complete]
    keys = list(all_settings[0])
    varying = [key for key in keys if len({s[key] for s in all_settings}) > 1]
    constant = [all_settings[0][key] for key in keys if key not in varying]
    columns = [key for key in varying if key not in (STRATEGY, SEED)]
    runs = []
    for (directory, config), settings in zip(complete, all_settings, strict=True):
        setting = " · ".join(settings[key] for key in columns) or "all runs"
        name = f"run_{directory.name.split('_')[0]}"
        short = f"{setting} · {settings[SEED]}" if SEED in varying else setting
        label = f"{name} · {settings[STRATEGY]}\n{short}"
        runs.append(Run(directory, label, config, settings[STRATEGY], setting, short))
    return runs, incomplete_runs, varying, constant


def _brute_baseline(run: Run) -> pl.DataFrame:
    """Per-pair brute-force proof cost from the problem set `run` was built on."""
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
    """Per-pair unguided results from the baseline folder `run` references."""
    directory = REPO_ROOT / run.config["baseline"]
    results = directory / "unguided.parquet"
    if not results.is_file():
        raise FileNotFoundError(
            f"{run.directory.name} was checked against {run.config['baseline']}, "
            "which has no unguided.parquet"
        )
    return pl.read_parquet(results).with_columns(pl.lit(directory.name).alias("baseline"))


def _load_run_table(runs: Sequence[Run], filename: str) -> pl.DataFrame:
    """Concatenate a per-run parquet table tagged with `run`, `strategy`, and `setting`."""
    return pl.concat(
        pl.read_parquet(run.directory / filename).with_columns(
            pl.lit(run.directory.name).alias("run"),
            pl.lit(run.strategy).alias("strategy"),
            pl.lit(run.setting).alias("setting"),
        )
        for run in runs
    )


def load_comparisons(
    patterns: Sequence[str],
    folder: str = "guided_search",
    where: Mapping[str, Any] | None = None,
) -> tuple[list[Run], list[str], pl.DataFrame, dict]:
    """Load and stack selected runs under `data/<folder>` with unguided and brute-force baselines."""
    runs, incomplete, varying, constant = _resolve_runs(patterns, folder, where or {})
    frames = []
    for run in runs:
        frame = pl.read_parquet(run.directory / "pairs.parquet")
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
                pl.when("guided_success").then("wall_time").alias("success_time"),
                pl.when(pl.col("attempts_run") > 0).then("wall_time").alias("search_time"),
            )
        )
    data = pl.concat(frames, how="diagonal_relaxed")
    seeds = ", ".join(dict.fromkeys(str(run.config["seed"]) for run in runs))
    settings = list(dict.fromkeys(run.setting for run in runs))
    cell_counts = data.group_by(GROUP).agg(
        pl.len().alias("tested"),
        pl.col("pair").n_unique().alias("unique"),
    )
    tested = _count_span(cell_counts["tested"])
    unique = _count_span(cell_counts["unique"])
    meta = {
        "strategies": list(dict.fromkeys(run.strategy for run in runs)),
        "settings": settings,
        # Group each setting's seeds together.
        "shorts": list(
            dict.fromkeys(
                run.short for run in sorted(runs, key=lambda run: settings.index(run.setting))
            )
        ),
        "subtitle": [
            s for s in (" · ".join(constant), f"{tested} goals tested ({unique} unique)") if s
        ],
        "pooled": f"seeds {seeds} pooled" if SEED in varying else None,
    }
    return runs, incomplete, data, meta


def _count_span(series: pl.Series) -> str:
    """Single integer when all categories match, otherwise `min–max`."""
    lo, hi = int(series.min()), int(series.max())  # ty: ignore[invalid-argument-type]
    return str(lo) if lo == hi else f"{lo}–{hi}"


def outcome_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """Counts and shares for the four paired success outcomes."""
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
    "budget_exhausted": "pair time budget",
    "panic": "panic in eqsat",
    "arg_too_long": "guide too long for argv",
}


def _stop_category(reason: pl.Expr) -> pl.Expr:
    """Map egg stop strings (e.g. `TimeLimit(300.08)`) to display categories."""
    return (
        reason.fill_null("unknown")
        .str.extract(r"^(\w+)")
        .replace_strict(STOP_LABELS, default="other")
    )


SEARCH_LABELS = {
    "time_exhausted": "pair time budget",
    "attempt_budget_exhausted": "attempt budget",
    "frontier_exhausted": "frontier exhausted",
    "unstarted": "search never started",
}

SETUP_LABELS = {
    "out_of_memory": "out of memory",
    "no_novel_terms": "no novel terms",
    "empty_pool": "empty pool",
    "binary_panic": "binary panic",
    "timeout": "hard timeout",
    "budget_exhausted": "pair time budget",
    "arg_too_long": "start too long for argv",
}


def search_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    """Mutually exclusive search outcome per pair (setup failures take precedence)."""
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
    """Failed attempts by stop reason, split by whether their pair was eventually proved."""
    reached = pl.col("reached").fill_null(False)
    return (
        _load_run_table(runs, "attempts.parquet")
        .with_columns(
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
    """Guided peak RSS against brute-force proof cost for successful pairs."""
    column = GUIDED_PEAK_SCOPES[scope]
    return frame.filter(pl.col("guided_success")).with_columns(
        pl.lit(scope).alias("guided_peak_scope"),
        (pl.col(column) / 2**20).alias("guided_peak_mib"),
        (pl.col(BRUTE_COLUMN) / 2**20).alias("brute_peak_mib"),
        (pl.col(column) / pl.col(BRUTE_COLUMN)).alias("peak_ratio"),
    )


def _brute_cost_by_outcome(frame: pl.DataFrame, bins: int = 14) -> pl.DataFrame:
    """Log-binned brute-force proof cost per grid cell, split by guided outcome."""
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
    usable = 0.86  # Leave a gap between adjacent log bins.
    slots = len(BRUTE_COST_OUTCOMES)
    slot = pl.col("outcome").replace_strict({name: i for i, name in enumerate(BRUTE_COST_OUTCOMES)})
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
        .with_columns(
            (10 ** (low + pl.col("bin") * width)).alias("bin_start_mib"),
            (10 ** (low + (pl.col("bin") + 1) * width)).alias("bin_end_mib"),
            (10 ** (low + (pl.col("bin") + usable * slot / slots) * width)).alias("slot_start_mib"),
            (10 ** (low + (pl.col("bin") + usable * (slot + 1) / slots) * width)).alias(
                "slot_end_mib"
            ),
            (pl.col("count") / pl.col("count").sum().over(*GROUP, "outcome")).alias("share"),
            pl.col("count").sum().over(*GROUP, "bin").alias("bucket_n"),
        )
        .with_columns((pl.col("count") / pl.col("bucket_n")).alias("bucket_share"))
    )


def peak_memory(
    frame: pl.DataFrame, scope: str = "guided workflow", bins: int = 14
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Guided-vs-brute peak RSS on proved pairs and brute-force cost buckets by outcome."""
    return guided_vs_brute(frame, scope), _brute_cost_by_outcome(frame, bins=bins)


def pairwise_solved_diff(frame: pl.DataFrame) -> pl.DataFrame:
    """Per ordered mode pair (excluding diagonal), solved-set differences and unions."""
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
            ((pl.col("n_shared") - pl.col("neither")) / pl.col("n_shared")).alias("share_combined"),
        )
        .join(runs.select(pl.all().name.prefix("row_")), on="row_mode", how="left")
        .join(runs.select(pl.all().name.prefix("col_")), on="col_mode", how="left")
    )


def pow2_bucket_label(bucket: int) -> str:
    """Display range of a power-of-two bucket: 1, 2, 3–4, 5–8, …"""
    if bucket < 2:
        return str(bucket + 1)
    return f"{2 ** (bucket - 1) + 1}–{2**bucket}"


def _pow2_counts(frame: pl.DataFrame, value: str, group_cols: Sequence[str]) -> pl.DataFrame:
    """Count positive rows of `value` in power-of-two buckets (2^(k-1), 2^k]."""
    positive = frame.filter(pl.col(value) > 0)
    top = int(positive[value].max() or 1)  # ty: ignore[invalid-argument-type]
    # Exact ceil(log2 n) without float rounding.
    bucket = pl.sum_horizontal(pl.col(value) > 2**j for j in range(top.bit_length())).cast(pl.Int64)
    return (
        positive.group_by(*group_cols, bucket.alias("bucket"), maintain_order=True)
        .len("count")
        .with_columns(
            pl.col("bucket").map_elements(pow2_bucket_label, return_dtype=pl.String).alias("range")
        )
    )


def search_effort(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Power-of-two bucket counts per grid cell for attempts, depth, and wall time."""
    result = {}
    for name, measures in MEASURES.items():
        values = frame.unpivot(
            index=GROUP, on=list(measures), variable_name="measure", value_name=name
        ).with_columns(pl.col("measure").replace_strict(measures))
        result[name] = _pow2_counts(values, name, (*GROUP, "measure"))
    return result


def sampling_overhead(runs: Sequence[Run]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Non-cached `sample` retry buckets and share of wall time lost to RSS-killed tries."""
    expansions = _load_run_table(runs, "expansions.parquet").filter(~pl.col("cached"))
    tries = _pow2_counts(expansions, "sample_tries", GROUP)
    killed = (
        expansions.group_by(GROUP, maintain_order=True)
        .agg(
            pl.col("sample_killed_time").sum().alias("killed_time"),
            pl.col("wall_time").sum().alias("sample_time"),
            (pl.col("sample_tries") > 1).sum().alias("retried"),
            pl.len().alias("draws"),
        )
        .with_columns((pl.col("killed_time") / pl.col("sample_time")).alias("killed_share"))
    )
    return tries, killed


def success_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Compact success and median peak-ratio summary per grid cell."""
    summary = frame.group_by(GROUP, maintain_order=True).agg(
        pl.first("baseline"),
        pl.col("guided_success").sum().alias("guided_successes"),
        pl.len().alias("guided_n"),
        pl.col("guided_success").mean().alias("guided_success_rate"),
        pl.col("unguided_success").sum().alias("unguided_successes"),
        pl.len().alias("unguided_n"),
        pl.col("unguided_success").mean().alias("unguided_success_rate"),
    )
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
