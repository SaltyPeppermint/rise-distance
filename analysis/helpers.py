"""Load guided-search runs into frames used by the analysis plots."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Run:
    pattern: str
    directory: Path
    label: str
    k: int
    attempts: int
    strategy: str
    limits: dict[str, dict[str, float]]


def _list_runs(pattern: str = "", subdir: str = "guided_search") -> list[Path]:
    """Run directories matching `pattern` under `data/<subdir>`, oldest first."""
    data_dir = Path(__file__).parent / ".." / "data" / subdir
    return sorted(
        (f for f in data_dir.iterdir() if f.is_dir() and pattern in f.name),
        key=lambda p: p.stat().st_mtime,
    )


def _load_leg_frame(run_dir: Path) -> pl.DataFrame:
    """Load legs, folding guide-egraph overhead into their costs."""
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
    """Load each seed's unguided full-eqsat baseline."""
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
                    # base_memory sits beside goal_egraph in the payload.
                    "base_memory": ok["base_memory"],
                    "base_stop_reason": ok["stop_reason"],
                }
            )
    if not rows:
        raise ValueError(f"No Ok seeds with a goal_egraph baseline in {config['path']}")
    return pl.DataFrame(rows)


def load_runs(runs: Sequence[Run]) -> tuple[pl.DataFrame, dict]:
    """Stack runs into a leg frame and return it with plot metadata."""
    frames, modes, k_by_mode, strategy_by_mode = [], [], {}, {}
    n_goals = n_trials = 0
    for run in runs:
        run_dir = run.directory
        config = json.loads((run_dir / "config.json").read_text())

        legs = _load_leg_frame(run_dir)
        strategies = legs["strategy"].unique().to_list()
        if strategies != [run.strategy]:
            raise ValueError(
                f"{run_dir.name}: config strategy={run.strategy!r}, "
                f"result strategies={sorted(strategies)!r}"
            )
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
        strategy_by_mode[label] = run.strategy
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
        "strategy": strategy_by_mode,
        "n_goals": n_goals,
        "n_trials": n_trials,
        "defaults": _render_fields(_partition_fields(runs)[0]),
    }
    print(
        f"\nk per mode: {k_by_mode}\n"
        f"strategy per mode: {strategy_by_mode}   "
        f"(n_goals={n_goals}, n_trials={n_trials})"
    )
    return df, meta


# Eqsat-config ceilings. Classes are not bounded separately.
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


def _run_fields(limits: dict[str, dict[str, float]], attempts: int, k: int, strategy: str) -> dict:
    """Format the fields used to compare and label runs."""
    replay = limits["leg"]
    baseline = limits["baseline"]
    fields = {}
    for metric in LIMIT_KEYS:
        if replay.get(metric) != baseline.get(metric):
            fields[LIMIT_LABELS[metric]] = (
                f"{_format_limit(metric, replay.get(metric))} vs "
                f"{_format_limit(metric, baseline.get(metric))}"
            )
        else:
            fields[LIMIT_LABELS[metric]] = "baseline"
    fields["attempts"] = _compact_number(float(attempts))
    fields["k"] = str(k)
    fields["strategy"] = strategy
    return fields


def _render_fields(fields: dict) -> str:
    """Render run fields as a plot-label fragment."""
    parts = []
    for name, value in fields.items():
        if name in LIMIT_LABELS.values():
            if value != "baseline":
                parts.append(f"{name} {value}")
        else:
            parts.append(f"{name}={value}")
    return " · ".join(parts) if parts else "baseline limits"


def resolve_runs(patterns: Sequence[str]) -> list[Run]:
    """Resolve run patterns and derive collision-safe plot labels.

    With no patterns, resolve every `run.*` directory instead.
    """
    if not patterns:
        patterns = [d.name for d in _list_runs("run.")]

    runs = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = _list_runs(pattern)
        if not matches:
            raise FileNotFoundError(f"No run directory matching {pattern!r}")
        run_dir = matches[-1]
        if run_dir in seen:
            continue
        seen.add(run_dir)
        config = json.loads((run_dir / "config.json").read_text())
        legs = pl.read_parquet(run_dir / "results.parquet").filter(pl.col("k") > 0)
        ks = legs["k"].unique().to_list()
        if len(ks) != 1:
            raise ValueError(f"{run_dir.name}: expected a single k, found {sorted(ks)}")
        k = ks[0]
        runs.append(
            Run(
                pattern=pattern,
                directory=run_dir,
                label="",
                k=k,
                attempts=config["attempts"],
                strategy=config["strategy"],
                limits=_governing_limits(run_dir),
            )
        )

    _, differing = _partition_fields(runs)
    for i, fields in enumerate(differing):
        runs[i] = replace(runs[i], label=_render_fields(fields))

    labels = [run.label for run in runs]
    for label in set(labels):
        indexes = [i for i, candidate in enumerate(labels) if candidate == label]
        if len(indexes) < 2:
            continue
        for i in indexes:
            runs[i] = replace(runs[i], label=f"{runs[i].label} · {runs[i].directory.name}")
    return runs


def _partition_fields(runs: Sequence[Run]) -> tuple[dict, list[dict]]:
    """Split run fields into shared defaults and per-run differences."""
    all_fields = [_run_fields(run.limits, run.attempts, run.k, run.strategy) for run in runs]
    if not all_fields:
        return {}, []
    shared_names = {name for name in all_fields[0] if len({f[name] for f in all_fields}) == 1}
    shared = {name: all_fields[0][name] for name in all_fields[0] if name in shared_names}
    differing = [
        {name: v for name, v in fields.items() if name not in shared_names} for fields in all_fields
    ]
    return shared, differing


def _governing_limits(run_dir: Path) -> dict[str, dict[str, float]]:
    """Resolve guide, leg, and baseline limits for one run."""
    config = json.loads((run_dir / "config.json").read_text())
    goal_args = json.loads(
        (Path(__file__).parent / ".." / config["path"] / "goal_args.json").read_text()
    )
    base = {dim: goal_args.get(key) for dim, key in LIMIT_KEYS.items()}

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


# Limit family used by each plotted series.
SERIES_EGRAPH = {"leg": "leg", "baseline": "baseline", "guide": "guide"}


def series_limit_frame(
    runs: Sequence[Run],
    metrics: Sequence[str],
    series: Sequence[str] = ("leg", "baseline", "guide"),
) -> pl.DataFrame:
    """Return ``(mode, series, metric, limit)`` rows for plotted ceilings.

    Series may have different ceilings; unbounded metrics are omitted.
    """
    rows = [
        {"mode": mode, "series": s, "metric": m, "limit": float(lims[SERIES_EGRAPH[s]][m])}
        for mode, lims in ((run.label, run.limits) for run in runs)
        for s in series
        for m in metrics
        if m in LIMIT_KEYS and m in lims.get(SERIES_EGRAPH[s], {})
    ]
    return pl.DataFrame(
        rows,
        schema={"mode": pl.String, "series": pl.String, "metric": pl.String, "limit": pl.Float64},
    )


def goal_reach(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate reach and empty-pool rates by seed/goal pair."""
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
