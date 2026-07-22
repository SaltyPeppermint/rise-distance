"""Data loading and tidy-frame shaping for the guide-selection analysis.

`guided_search.py` writes one `results.parquet` per run (a run == one strategy/config).
This module loads a set of runs into a single **tidy long DataFrame** — one row
per leg, tagged with its display `mode`, with the per-seed unguided baseline and
the metric/baseline `ratio` already attached. The Altair specs in `plots.py`
consume that frame directly, so the notebook stays a handful of declarative
calls.

Every run now uses a single guide-set size `k` (see `guided_search.py`), so `k`
is a per-run constant, not a swept axis. Each mode's display name is derived
from its replay-versus-baseline limit differences, attempt count, and `k`; the
plots put the individual seed/goal *pairs* on the x-axis instead.

Column glossary for the tidy frame (`load_runs`):

    mode         derived display name of the run (overlaid series)
    seed, goal   the eqsat seed term and the goal term
    pair         "<seed>│<goal>", the per-experiment identifier used as x
    k            number of guides (constant within a run)
    reached      bool: did this leg reach the goal
    stop_reason  "empty_pool" when the candidate pool was empty
    iters, nodes, classes, nodes_per_class, total_time, memory
                 leg cost (nodes/classes/time/memory include the guide-egraph
                 overhead; nodes/classes are the true final egraph size read off
                 the rebuilt egraph after the leg, not an iteration-boundary
                 snapshot; memory is jemalloc live-heap bytes (stats.allocated),
                 folded as max(leg, guide))
    base_nodes, base_classes, base_iters, base_total_time, base_nodes_per_class,
    base_memory  that seed's full unguided eqsat cost (base_memory = live-heap
                 bytes sampled right after the unguided run)
    base_stop_reason
                 why that unguided run ended, including the limit measurement
                 carried by egg (for example ``NodeLimit(112290)``)
    guide_nodes, guide_classes, guide_time, guide_memory
                 the per-seed guide replay's cost (from `sample.rs`)
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Run:
    """A resolved guided-search run with an analysis-derived display label."""

    pattern: str
    directory: Path
    label: str
    k: int
    attempts: int
    strategy: str
    limits: dict[str, dict[str, float]]


def find_latest_run(pattern: str = "run", subdir: str = "guided_search") -> Path:
    """Return the newest run directory matching `pattern` under `data/<subdir>`."""
    data_dir = Path(__file__).parent / ".." / "data" / subdir
    candidates = sorted(
        (f for f in data_dir.iterdir() if f.is_dir() and pattern in f.name),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No '{pattern}' directories in {data_dir.resolve()}")
    return candidates[-1]


def _load_leg_frame(run_dir: Path) -> pl.DataFrame:
    """One run's `results.parquet`, with guide-egraph overhead folded into cost.

    nodes/classes/memory become ``max(leg, guide)`` and total_time gains
    guide_time, so each leg's cost includes the guide phase it built on. For
    memory (jemalloc live-heap bytes, a point-in-time reading) the max is the
    heavier of the two phases' footprints. Null cost columns (unreached /
    empty-pool / panic legs) stay null. Empty-pool legs (k=0, no real guide set)
    are dropped.
    """
    df = pl.read_parquet(run_dir / "results.parquet").filter(pl.col("k") > 0)
    return df.with_columns(
        pl.when(pl.col("nodes").is_not_null())
        .then(pl.max_horizontal("nodes", "guide_nodes"))
        .alias("nodes"),
        pl.when(pl.col("classes").is_not_null())
        .then(pl.max_horizontal("classes", "guide_classes"))
        .alias("classes"),
        pl.when(pl.col("total_time").is_not_null())
        .then(pl.col("total_time") + pl.col("guide_time"))
        .alias("total_time"),
        pl.when(pl.col("memory").is_not_null())
        .then(pl.max_horizontal("memory", "guide_memory"))
        .alias("memory"),
    ).with_columns((pl.col("nodes") / pl.col("classes")).alias("nodes_per_class"))


def _baseline_frame(run_dir: Path) -> pl.DataFrame:
    """Per-seed unguided full-eqsat baseline for a run, as a joinable frame.

    The run's `config.json` points (via `path`) at the seed-terms folder, whose
    `goal_terms.json` carries a per-seed `goal_egraph` block: full eqsat on the
    seed with no guides. Every goal under a seed shares that seed's baseline, so
    a guided leg can be compared against its own seed's unguided cost. `Err`
    seeds are skipped. Columns: seed, base_<metric> for each METRIC.
    """
    config = json.loads((run_dir / "config.json").read_text())
    repo_root = Path(__file__).parent / ".."
    terms = json.loads((repo_root / config["path"] / "goal_terms.json").read_text())

    rows = []
    for _size, inner in terms:
        for seed, payload in inner.items():
            if "Ok" not in payload:
                continue
            ok = payload["Ok"]
            e = ok["goal_egraph"]
            rows.append(
                {
                    "seed": seed,
                    "base_iters": e["iters"],
                    "base_nodes": e["nodes"],
                    "base_classes": e["classes"],
                    "base_nodes_per_class": e["nodes"] / e["classes"],
                    "base_total_time": e["time"],
                    # Live-heap bytes (jemalloc stats.allocated): on the payload,
                    # not inside goal_egraph.
                    "base_memory": ok["base_memory"],
                    "base_stop_reason": ok["stop_reason"],
                }
            )
    if not rows:
        raise ValueError(f"No Ok seeds with a goal_egraph baseline in {config['path']}")
    return pl.DataFrame(rows)


def load_runs(runs: Sequence[Run]) -> tuple[pl.DataFrame, dict]:
    """Load and stack the given runs into one tidy long DataFrame.

    `runs` comes from `resolve_runs`, which derives each display label from the
    replay/baseline limit difference, attempt count, and guide-set size `k`.
    Returns ``(df, meta)`` where `df` is the tidy frame described in the module
    docstring and `meta` carries plot-annotation info::

        meta = {
            "modes":    [derived display names, in order],
            "k":        {mode -> its single k},
            "n_goals":  int,                   # max distinct goals across modes
            "n_trials": int,                   # configured `attempts` (max)
        }
    """
    frames, modes, k_by_mode = [], [], {}
    n_goals = n_trials = 0
    for run in runs:
        run_dir = run.directory
        config = json.loads((run_dir / "config.json").read_text())

        legs = _load_leg_frame(run_dir)
        ks = legs["k"].unique().to_list()
        if len(ks) != 1:
            raise ValueError(f"{run_dir.name}: expected a single k, found {sorted(ks)}")
        k = ks[0]
        if k != run.k:
            raise ValueError(f"{run_dir.name}: resolved k={run.k}, found k={k}")
        label = run.label

        base = _baseline_frame(run_dir)
        df = legs.join(base, on="seed", how="left").with_columns(
            pl.lit(label).alias("mode"),
            pl.concat_str(["seed", "goal"], separator="│").alias("pair"),
        )
        frames.append(df)
        modes.append(label)
        k_by_mode[label] = k
        n_goals = max(n_goals, df["goal"].n_unique())
        n_trials = max(n_trials, config["attempts"])

        n_reached = int(df["reached"].sum())
        print(
            f"{label}: {run_dir.name} strategy={config['strategy']}: "
            f"{len(df)} rows, {n_reached} reached"
        )

    df = pl.concat(frames, how="diagonal_relaxed")
    meta = {
        "modes": modes,
        "k": k_by_mode,
        "n_goals": n_goals,
        "n_trials": n_trials,
    }
    print(f"\nk per mode: {k_by_mode}   (n_goals={n_goals}, n_trials={n_trials})")
    return df, meta


# Dimensions that egg actually bounds with a `--max-*` (or `--stop-*`) limit, and
# the eqsat-config key carrying that ceiling. `classes` has no limit (egg bounds
# nodes, not classes), so it never gets a limit line.
LIMIT_KEYS = {
    "iters": "max_iters",
    "nodes": "max_nodes",
    "total_time": "max_time",
    "memory": "max_memory",
}

LIMIT_LABELS = {
    "iters": "iters",
    "nodes": "nodes",
    "total_time": "time",
    "memory": "memory",
}


def _compact_number(value: float) -> str:
    """Format a config value compactly without hiding meaningful precision."""
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        scaled = value / scale
        if value >= scale and scaled.is_integer():
            return f"{int(scaled)}{suffix}"
    return f"{value:g}"


def _format_limit(metric: str, value: float | None) -> str:
    if value is None:
        return "unlimited"
    if metric == "memory":
        for scale, suffix in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
            scaled = value / scale
            if value >= scale and scaled.is_integer():
                return f"{int(scaled)}{suffix}"
    formatted = _compact_number(float(value))
    return f"{formatted}s" if metric == "total_time" else formatted


def _derived_label(
    limits: dict[str, dict[str, float]], attempts: int, k: int
) -> str:
    """Describe only replay limits that differ from the unguided baseline."""
    replay = limits["leg"]
    baseline = limits["baseline"]
    differences = [
        f"{LIMIT_LABELS[metric]} {_format_limit(metric, replay.get(metric))} vs "
        f"{_format_limit(metric, baseline.get(metric))}"
        for metric in LIMIT_KEYS
        if replay.get(metric) != baseline.get(metric)
    ]
    limit_part = ", ".join(differences) if differences else "baseline limits"
    return f"{limit_part} · attempts={_compact_number(float(attempts))} · k={k}"


def resolve_runs(patterns: Sequence[str]) -> list[Run]:
    """Resolve run patterns and derive concise, collision-safe plot labels.

    Labels show guide/leg limits versus the unguided baseline, but only for
    dimensions whose effective limits differ. Strategy and directory provenance
    are appended only when the analytical labels would otherwise collide.
    """
    runs = []
    for pattern in patterns:
        run_dir = find_latest_run(pattern)
        config = json.loads((run_dir / "config.json").read_text())
        legs = pl.read_parquet(run_dir / "results.parquet").filter(pl.col("k") > 0)
        ks = legs["k"].unique().to_list()
        if len(ks) != 1:
            raise ValueError(f"{run_dir.name}: expected a single k, found {sorted(ks)}")
        k = ks[0]
        limits = _governing_limits(run_dir)
        runs.append(
            Run(
                pattern=pattern,
                directory=run_dir,
                label=_derived_label(limits, config["attempts"], k),
                k=k,
                attempts=config["attempts"],
                strategy=config["strategy"],
                limits=limits,
            )
        )

    labels = [run.label for run in runs]
    for label in set(labels):
        indexes = [i for i, candidate in enumerate(labels) if candidate == label]
        if len(indexes) < 2:
            continue
        strategy_labels = [f"{runs[i].label} · {runs[i].strategy}" for i in indexes]
        unique_by_strategy = len(set(strategy_labels)) == len(strategy_labels)
        for i, strategy_label in zip(indexes, strategy_labels, strict=True):
            suffix = (
                strategy_label
                if unique_by_strategy
                else f"{strategy_label} · {runs[i].directory.name}"
            )
            runs[i] = replace(runs[i], label=suffix)
    return runs


def _governing_limits(run_dir: Path) -> dict[str, dict[str, float]]:
    """The eqsat limits each egraph family actually ran under, per run.

    All three read the search-phase eqsat config from the seed folder's
    `goal_args.json`. The **guide** replay and the **leg** search both run under
    the `--stop-*` replay budget the guided_search run set (in `config.json`,
    each given dimension overriding its search-phase value — see
    `guided_search.replay_limits`); the **baseline** (unguided `goal.rs`) run uses
    the plain search-phase limits. Returns ``{egraph -> {dimension -> limit}}``,
    dropping dimensions whose limit is unset (e.g. no `--max-memory`).
    """
    config = json.loads((run_dir / "config.json").read_text())
    goal_args = json.loads(
        (Path(__file__).parent / ".." / config["path"] / "goal_args.json").read_text()
    )
    base = {dim: goal_args.get(key) for dim, key in LIMIT_KEYS.items()}

    # Guide replay and legs: search-phase limits with each given --stop-*
    # overriding its dimension (see guided_search.replay_limits).
    stop = {
        "iters": config.get("stop_iters"),
        "nodes": config.get("stop_nodes"),
        "total_time": config.get("stop_time"),
        "memory": config.get("stop_memory"),
    }
    replay = {dim: (stop[dim] if stop[dim] is not None else base[dim]) for dim in base}

    limits = {"guide": dict(replay), "leg": dict(replay), "baseline": dict(base)}
    return {
        egraph: {dim: lim for dim, lim in dims.items() if lim is not None}
        for egraph, dims in limits.items()
    }


def _limits_by_mode(runs: Sequence[Run]) -> dict[str, dict[str, dict[str, float]]]:
    """``mode -> {egraph -> {dimension -> limit}}`` for the given runs.

    Uses the same resolved labels and limits as `load_runs`, so its keys line up
    exactly with the tidy frame's `mode` column.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for run in runs:
        out[run.label] = run.limits
    return out


# Which `_governing_limits` egraph slot each plotted series ran under. The guide
# replay and the legs share the `--stop-*` replay budget; the baseline is the
# unguided full run under the plain search-phase limits.
SERIES_EGRAPH = {"leg": "leg", "baseline": "baseline", "guide": "guide"}


def series_limit_frame(
    runs: Sequence[Run],
    metrics: Sequence[str],
    series: Sequence[str] = ("leg", "baseline", "guide"),
) -> pl.DataFrame:
    """Tidy ``(mode, series, metric, limit)`` frame of the ceiling each plotted
    series ran under, for the limited `metrics`.

    The guide replay and the legs share the `--stop-*` replay budget, while the
    baseline ran under the plain search-phase limits, so the three series can have
    different ceilings — hence a limit per (series, metric), not one shared line.
    `abs_pair_strip` draws a dashed rule per series, colored to match its points.
    `metrics` without a limit (e.g. `classes`, which egg doesn't bound) are
    dropped, so those facets get no line.
    """
    limits_by_mode = _limits_by_mode(runs)
    rows = [
        {"mode": mode, "series": s, "metric": m, "limit": float(lims[SERIES_EGRAPH[s]][m])}
        for mode, lims in limits_by_mode.items()
        for s in series
        for m in metrics
        if m in LIMIT_KEYS and m in lims.get(SERIES_EGRAPH[s], {})
    ]
    return pl.DataFrame(
        rows,
        schema={"mode": pl.String, "series": pl.String, "metric": pl.String, "limit": pl.Float64},
    )


def goal_reach(df: pl.DataFrame) -> pl.DataFrame:
    """Per (mode, goal) reachability over its attempts: reach_rate and empty_pool_rate.

    With a single k per run, this aggregates over the attempt loop for each
    seed/goal pair. `cond_reach_rate` excludes empty-pool legs from the
    denominator: given we could sample k guides, did we reach the goal?
    """
    empty = pl.col("stop_reason") == "empty_pool"
    return (
        df.group_by("mode", "seed", "goal", "pair")
        .agg(
            pl.col("reached").mean().alias("reach_rate"),
            (pl.col("reached").sum() / empty.not_().sum()).alias("cond_reach_rate"),
            empty.mean().alias("empty_pool_rate"),
        )
        .sort("mode", "goal")
    )
