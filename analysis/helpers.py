"""Data loading and tidy-frame shaping for the guide-selection analysis.

`guided_search.py` writes one `results.parquet` per run (a run == one strategy/config).
This module loads a set of runs into a single **tidy long DataFrame** — one row
per leg, tagged with its display `mode`, with the per-seed unguided baseline and
the metric/baseline `ratio` already attached. The Altair specs in `plots.py`
consume that frame directly, so the notebook stays a handful of declarative
calls.

Column glossary for the tidy frame (`load_runs`):

    mode         display name of the run (overlaid series)
    seed, goal   the eqsat seed term and the goal term
    k            number of guides
    reached      bool: did this leg reach the goal
    stop_reason  "empty_pool" when the candidate pool was empty
    iters, nodes, classes, nodes_per_class, total_time, memory
                 leg cost (nodes/classes/time/memory include the guide-egraph
                 overhead; memory is peak process RSS, folded as max(leg, guide))
    base_nodes, base_classes, base_iters, base_total_time, base_nodes_per_class,
    base_memory  that seed's full unguided eqsat cost (base_memory = whole-run
                 peak RSS)
    r_nodes, r_classes, r_iters, r_total_time, r_nodes_per_class, r_memory
                 metric / baseline (paired per seed); null where the metric is null
"""

import json
from pathlib import Path

import polars as pl

# Cost metrics carried through the plots. `nodes_per_class` is computed but not
# plotted by default; add it to a plot's metric list to surface it.
METRICS = ["iters", "nodes", "classes", "nodes_per_class", "total_time", "memory"]

# smallest_* strategies emit a single k=1 point per goal (no k sweep). The plots
# draw them as standalone markers rather than lines/bands.
SINGLE_POINT_STRATEGIES = {"smallest_overall", "smallest_novel"}


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
    guide_time, so each leg's cost includes the guide phase it built on. Null
    cost columns (unreached / empty-pool / panic legs) stay null. Empty-pool legs
    (k=0, no real guide set) are dropped.
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
                    # Whole-run peak RSS: on the payload, not inside goal_egraph.
                    "base_memory": ok["base_memory"],
                }
            )
    if not rows:
        raise ValueError(f"No Ok seeds with a goal_egraph baseline in {config['path']}")
    return pl.DataFrame(rows)


def load_runs(runs: list[tuple[str, str]]) -> tuple[pl.DataFrame, dict]:
    """Load and stack the given runs into one tidy long DataFrame.

    `runs` is a list of ``(display_name, run-dir pattern)``. Returns
    ``(df, meta)`` where `df` is the tidy frame described in the module docstring
    and `meta` carries plot-annotation info::

        meta = {
            "modes":    [display names, in order],
            "single":   {mode -> bool},        # single-point strategy?
            "k_values": [sorted union of k across multi-point modes],
            "n_goals":  int,                   # from the first multi-point mode
            "n_trials": int,                   # configured `attempts`
        }
    """
    frames, modes, single = [], [], {}
    n_goals = n_trials = None
    for name, pattern in runs:
        run_dir = find_latest_run(pattern)
        config = json.loads((run_dir / "config.json").read_text())
        is_single = config["strategy"] in SINGLE_POINT_STRATEGIES

        legs = _load_leg_frame(run_dir)
        base = _baseline_frame(run_dir)
        df = legs.join(base, on="seed", how="left").with_columns(
            pl.lit(name).alias("mode"),
            *[
                pl.when(pl.col(f"base_{m}") > 0)
                .then(pl.col(m) / pl.col(f"base_{m}"))
                .alias(f"r_{m}")
                for m in METRICS
            ],
        )
        frames.append(df)
        modes.append(name)
        single[name] = is_single

        n_reached = int(df["reached"].sum())
        print(
            f"{name}: {run_dir.name} strategy={config['strategy']}: {len(df)} rows, "
            f"k={sorted(df['k'].unique().to_list())}, {n_reached} reached"
        )
        if not is_single and n_goals is None:
            n_goals = df["goal"].n_unique()
            n_trials = config["attempts"]

    if n_goals is None:
        raise ValueError("no multi-point run found to derive (n_goals, n_trials)")

    df = pl.concat(frames, how="diagonal_relaxed")
    k_values = sorted(
        df.filter(~pl.col("mode").is_in([m for m, s in single.items() if s]))["k"]
        .unique()
        .to_list()
    )
    meta = {
        "modes": modes,
        "single": single,
        "k_values": k_values,
        "n_goals": n_goals,
        "n_trials": n_trials,
    }
    print(f"\nk values: {k_values}   (n_goals={n_goals}, n_trials={n_trials})")
    return df, meta


def goal_reach(df: pl.DataFrame) -> pl.DataFrame:
    """Per (mode, goal, k) reachability: reach_rate and empty_pool_rate.

    `cond_reach_rate` excludes empty-pool legs from the denominator: given we
    could sample k guides, did we reach the goal?
    """
    empty = pl.col("stop_reason") == "empty_pool"
    return (
        df.group_by("mode", "goal", "k")
        .agg(
            pl.col("reached").mean().alias("reach_rate"),
            (pl.col("reached").sum() / empty.not_().sum()).alias("cond_reach_rate"),
            empty.mean().alias("empty_pool_rate"),
        )
        .sort("mode", "goal", "k")
    )
