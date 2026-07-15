import json
from pathlib import Path

import numpy as np
import polars as pl


def find_latest_run(pattern: str = "run", subdir: str = "guide_driver") -> Path:
    """Return the newest run directory matching `pattern` under `data/<subdir>`.

    `subdir` defaults to `guide_driver`, where `driver.py` writes.
    """
    data_dir = Path(__file__).parent / ".." / "data" / subdir
    candidates = sorted(
        [f for f in data_dir.iterdir() if f.is_dir() and pattern in f.name],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No '{pattern}' directories in {data_dir.resolve()}")
    return candidates[-1]


def load_driver_run(run_dir: Path, adjust_guide_overhead: bool = True) -> pl.DataFrame:
    """Load a `driver.py` run's `results.parquet` into a flat DataFrame.

    The parquet is one row per leg with columns::

        seed, goal, strategy, k, attempt, reached, gave_up, iters,
        nodes, classes, total_applied, total_time, stop_reason, panic,
        guide_nodes, guide_classes, guide_time

    With `adjust_guide_overhead`, each leg's cost includes the guide-phase
    replay it built on:

      - nodes   = max(leg_nodes, guide_nodes)
      - classes = max(leg_classes, guide_classes)
      - total_time += guide_time

    Null cost columns (unreached, empty-pool, or panic legs) stay null.
    """
    df = pl.read_parquet(run_dir / "results.parquet")
    if not adjust_guide_overhead:
        return df

    return df.with_columns(
        pl.when(pl.col("nodes").is_not_null())
        .then(pl.max_horizontal(pl.col("nodes"), pl.col("guide_nodes")))
        .otherwise(None)
        .alias("nodes"),
        pl.when(pl.col("classes").is_not_null())
        .then(pl.max_horizontal(pl.col("classes"), pl.col("guide_classes")))
        .otherwise(None)
        .alias("classes"),
        pl.when(pl.col("total_time").is_not_null())
        .then(pl.col("total_time") + pl.col("guide_time"))
        .otherwise(None)
        .alias("total_time"),
    )


def load_baseline_per_seed(run_dir: Path) -> dict[str, dict[str, float]]:
    """Load the unguided full-eqsat baseline for each seed of a `driver.py` run.

    The run's `config.json` points (via `path`) at the seed-terms folder, whose
    `terms.json` carries a per-seed `goal_egraph` block: the full eqsat on the
    seed with no guides. This keeps the per-seed grain, so each guided leg can be
    compared against its own seed's unguided cost (all goals under a seed share
    that seed's baseline). `Err` seeds are skipped.

    Returns ``{seed_term: {nodes, total_time, iters, classes, nodes_per_class}}``.
    The seed key matches the `seed` column of `load_driver_run`.
    """
    config = json.loads((run_dir / "config.json").read_text())
    # config's `path` is repo-relative; resolve it against the repo root.
    repo_root = Path(__file__).parent / ".."
    terms = json.loads((repo_root / config["path"] / "terms.json").read_text())

    per_seed: dict[str, dict[str, float]] = {}
    for _size, inner in terms:
        for seed, payload in inner.items():
            if "Ok" not in payload:
                continue
            e = payload["Ok"]["goal_egraph"]
            per_seed[seed] = {
                "nodes": e["nodes"],
                "total_time": e["time"],
                "iters": e["iters"],
                "classes": e["classes"],
                "nodes_per_class": e["nodes"] / e["classes"],
            }
    if not per_seed:
        raise ValueError(f"No Ok seeds with a goal_egraph baseline in {config['path']}")
    return per_seed


def load_baseline(run_dir: Path) -> dict[str, float]:
    """Load the unguided full-eqsat baseline for a `driver.py` run, averaged over
    seeds.

    Scalar wrapper over `load_baseline_per_seed`: averages each metric over the
    measured seeds. For baseline comparisons, prefer the per-seed version, since
    averaging the guided cost and its reference separately drops the seed-level
    pairing.

    Returns a dict with keys: nodes, total_time, iters, classes, nodes_per_class.
    """
    per_seed = load_baseline_per_seed(run_dir)
    metrics = next(iter(per_seed.values())).keys()
    n = len(per_seed)
    return {m: sum(b[m] for b in per_seed.values()) / n for m in metrics}


def compute_goal_reach(df: pl.DataFrame) -> pl.DataFrame:
    """Per-goal x k reachability aggregation.

    `driver.py` re-samples on failure rather than emitting sampling-failure
    trials, so the only "couldn't sample" case is an empty candidate pool
    (`stop_reason == "empty_pool"`). `cond_reach_rate` excludes those legs from
    the denominator: given we could sample k guides, did we reach the goal?
    """
    empty_pool = pl.col("stop_reason") == "empty_pool"
    return (
        df.group_by("goal", "k")
        .agg(
            pl.col("reached").mean().alias("reach_rate"),
            (pl.col("reached").sum() / empty_pool.not_().sum()).alias("cond_reach_rate"),
            empty_pool.mean().alias("empty_pool_rate"),
        )
        .sort("goal", "k")
    )


def compute_goal_reach_matrix(df: pl.DataFrame, k_values: list[int]) -> tuple[list, np.ndarray]:
    """Build a (goals × k) reachability matrix sorted by avg reachability (hardest first).

    Returns (goal_order, matrix) where matrix[i, j] is the reach_rate for
    goal_order[i] at k_values[j], or NaN if missing.
    """
    goal_reach = compute_goal_reach(df)
    goal_order = (
        goal_reach.group_by("goal")
        .agg(pl.col("reach_rate").mean().alias("avg_reach"))
        .sort("avg_reach")["goal"]
        .to_list()
    )
    goal_idx = {g: i for i, g in enumerate(goal_order)}
    matrix = np.full((len(goal_order), len(k_values)), np.nan)
    for row in goal_reach.iter_rows(named=True):
        gi = goal_idx[row["goal"]]
        ki = k_values.index(row["k"])
        matrix[gi, ki] = row["reach_rate"]
    return goal_order, matrix


def compute_best_ratio_per_goal(
    df: pl.DataFrame,
    per_seed_baseline: dict[str, dict[str, float]],
    metrics: list[str],
) -> pl.DataFrame:
    """Per-goal best-guide cost relative to that goal's own seed baseline.

    For a given goal at a given k, how much did the best of all sampled guides
    improve over the unguided baseline? For every reached leg we form
    ``ratio = metric / seed_baseline[metric]`` (joined on `seed`), then per
    (goal, k) keep the minimum ratio: the single best guide for that goal.
    ``ratio < 1`` means the best guide beats unguided; ``1`` is break-even.

    Returns a long DataFrame with one row per (goal, k, metric) and columns
    ``goal, k, metric, best_ratio``. The plot handles cross-goal summarisation
    (median + IQR). Legs whose seed lacks an Ok baseline, or whose metric is
    null, are dropped before the per-goal min, so a goal's best comes only from
    guides that reached it.
    """
    parts = []
    for m in metrics:
        base = pl.DataFrame(
            {
                "seed": list(per_seed_baseline.keys()),
                "_base": [b[m] for b in per_seed_baseline.values()],
            }
        )
        part = (
            df.join(base, on="seed", how="inner")
            .filter(pl.col(m).is_not_null() & (pl.col("_base") > 0))
            .with_columns((pl.col(m) / pl.col("_base")).alias("ratio"))
            .group_by("goal", "k")
            .agg(pl.col("ratio").min().alias("best_ratio"))
            .with_columns(pl.lit(m).alias("metric"))
        )
        parts.append(part)
    if not parts:
        return pl.DataFrame(
            schema={"goal": pl.Utf8, "k": pl.Int64, "metric": pl.Utf8, "best_ratio": pl.Float64}
        )
    return pl.concat(parts).select("goal", "k", "metric", "best_ratio")
