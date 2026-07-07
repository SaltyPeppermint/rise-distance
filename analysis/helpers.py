import json
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


SCATTER_MAX_POINTS = 20_000


def plot_predictor_vs_iters(
    data: pl.DataFrame,
    predictor: str,
    target: str = "iters",
    title_suffix: str = "",
    ax: plt.Axes | None = None,  # pyright: ignore[reportPrivateImportUsage]
    max_points: int = SCATTER_MAX_POINTS,
):
    """Scatter plot of predictor vs iterations, with reached/timed-out markers and OLS line.

    Downsamples to `max_points` for rendering speed; OLS line uses full data.
    """
    if ax is None:
        _, ax = plt.subplots()

    reached = data.filter(pl.col("reached")).drop_nulls(subset=[target])
    timed_out = data.filter(~pl.col("reached"))

    x_r = reached[predictor].to_numpy().astype(float)
    y_r = reached[target].to_numpy().astype(float)

    # Downsample for plotting only
    n_r = len(x_r)
    if n_r > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_r, max_points, replace=False)
        x_r_plot, y_r_plot = x_r[idx], y_r[idx]
    else:
        x_r_plot, y_r_plot = x_r, y_r

    ax.scatter(
        x_r_plot,
        y_r_plot,
        c="steelblue",
        s=10,
        alpha=0.4,
        label=f"reached (n={n_r})",
        rasterized=True,
    )

    # Timed-out points (plotted at max_iters + 1)
    n_t = len(timed_out)
    if n_t > 0:
        x_t = timed_out[predictor].to_numpy().astype(float)
        timeout_y = y_r.max() + 1 if len(y_r) > 0 else 1
        if n_t > max_points:
            rng = np.random.default_rng(1)
            idx = rng.choice(n_t, max_points, replace=False)
            x_t_plot = x_t[idx]
        else:
            x_t_plot = x_t
        ax.scatter(
            x_t_plot,
            np.full_like(x_t_plot, timeout_y),
            c="red",
            marker="x",  # type: ignore[arg-type]
            s=10,
            alpha=0.3,
            label=f"timed out (n={n_t})",
            rasterized=True,
        )
        ax.axhline(timeout_y, color="red", ls="--", alpha=0.3, lw=1)

    # OLS regression line (full data, closed-form)
    if len(x_r) >= 2:
        x_mean = x_r.mean()
        y_mean = y_r.mean()
        x_c = x_r - x_mean
        ss_xx = x_c @ x_c
        slope = (x_c @ (y_r - y_mean)) / ss_xx if ss_xx > 1e-12 else 0.0
        intercept = y_mean - slope * x_mean
        resid = y_r - (slope * x_r + intercept)
        ss_tot = (y_r - y_mean) @ (y_r - y_mean)
        r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 1e-12 else 0.0
        x_line = np.linspace(x_r.min(), x_r.max(), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            c="green",
            lw=2,
            alpha=0.7,
            label=f"OLS (R²={r2:.3f})",
        )

    ax.set_xlabel(predictor)
    ax.set_ylabel(target)
    ax.set_title(f"{predictor} vs {target}{title_suffix}")
    ax.legend(fontsize=8, loc="best")
    return ax


def find_latest_run(pattern: str = "run", subdir: str = "guide_driver") -> Path:
    """Find the most recently modified run directory matching `pattern` under
    `data/<subdir>` (defaults to `guide_driver`, where `driver.py` writes)."""
    data_dir = Path(__file__).parent / ".." / "data" / subdir
    candidates = sorted(
        [f for f in data_dir.iterdir() if f.is_dir() and pattern in f.name],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No '{pattern}' directories in {data_dir.resolve()}")
    return candidates[-1]


STOP_REASON_CATEGORIES = ("Saturated", "IterationLimit", "NodeLimit", "TimeLimit", "Other")


def stop_reason_category(s: str | None) -> str | None:
    """Map a StopReason debug string (e.g. 'IterationLimit(30)', 'Saturated') to its variant name."""
    if s is None:
        return None
    for cat in STOP_REASON_CATEGORIES:
        if s.startswith(cat):
            return cat
    return "Other"


def load_driver_run(run_dir: Path, adjust_guide_overhead: bool = True) -> pl.DataFrame:
    """Load a `driver.py` run's `results.parquet` into a flat DataFrame.

    The parquet is already one row per leg with columns::

        seed, goal, strategy, k, attempt, reached, gave_up, iters,
        nodes, classes, total_applied, total_time, stop_reason, panic,
        guide_nodes, guide_classes, guide_time

    When `adjust_guide_overhead` is set, the per-leg cost is folded together with
    the guide-phase replay it built on (mirrors the old `load_top_k`):

      - nodes   = max(leg_nodes, guide_nodes)
      - classes = max(leg_classes, guide_classes)
      - total_time += guide_time

    Null cost columns (unreached / empty-pool / panic legs) are left null.
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


def load_baseline(run_dir: Path) -> dict[str, float]:
    """Load the unguided full-eqsat baseline for a `driver.py` run.

    The run's `config.json` points (via `path`) at the seed-terms folder, whose
    `terms.json` carries a per-seed `goal_egraph` block — the full eqsat on the
    seed with no guides. We average those metrics over the successfully-measured
    seeds (`Ok` payloads; `Err` seeds are skipped), mirroring the old
    `metadata.json` baseline.

    Returns a dict with keys: nodes, total_time, iters, classes, nodes_per_class.
    """
    config = json.loads((run_dir / "config.json").read_text())
    # config's `path` is repo-relative; resolve it against the repo root (the
    # parent of this file's `analysis/` directory).
    repo_root = Path(__file__).parent / ".."
    terms = json.loads((repo_root / config["path"] / "terms.json").read_text())

    egraphs = [
        payload["Ok"]["goal_egraph"]
        for _size, inner in terms
        for payload in inner.values()
        if "Ok" in payload
    ]
    if not egraphs:
        raise ValueError(f"No Ok seeds with a goal_egraph baseline in {config['path']}")

    n = len(egraphs)
    node_avg = sum(e["nodes"] for e in egraphs) / n
    classes_avg = sum(e["classes"] for e in egraphs) / n
    return {
        "nodes": node_avg,
        "total_time": sum(e["time"] for e in egraphs) / n,
        "iters": sum(e["iters"] for e in egraphs) / n,
        "classes": classes_avg,
        "nodes_per_class": node_avg / classes_avg,
    }


def compute_goal_reach(df: pl.DataFrame) -> pl.DataFrame:
    """Per-goal×k reachability aggregation.

    `driver.py` re-samples on failure rather than emitting sampling-failure
    trials, so the only "couldn't sample" case is an empty candidate pool
    (`stop_reason == "empty_pool"`). `cond_reach_rate` excludes those legs from
    the denominator: "given we could sample k guides, did we reach the goal?"
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


def compute_war_agg(df: pl.DataFrame, baseline: dict[str, float], metrics: list[str]) -> list[dict]:
    """Compute WAR (baseline - guided) rows per k for each metric.

    Returns a list of dicts with keys: k, plus for each metric
    '{metric}_war' (mean) and 'best_{metric}_war' (per-seed best, then averaged).
    """
    agg = (
        df.group_by("k")
        .agg([pl.col(m).mean().alias(f"mean_{m}") for m in metrics])
        .sort("k")
        .drop_nulls([f"mean_{m}" for m in metrics])
    )
    best_agg = (
        df.group_by("k", "seed")
        .agg([pl.col(m).min().alias(f"seed_best_{m}") for m in metrics])
        .group_by("k")
        .agg([pl.col(f"seed_best_{m}").mean().alias(f"best_{m}") for m in metrics])
        .sort("k")
        .drop_nulls([f"best_{m}" for m in metrics])
    )
    rows = []
    for row, best_row in zip(agg.iter_rows(named=True), best_agg.iter_rows(named=True)):
        entry: dict = {"k": row["k"]}
        for m in metrics:
            entry[f"{m}_war"] = baseline[m] - row[f"mean_{m}"]
            entry[f"best_{m}_war"] = baseline[m] - best_row[f"best_{m}"]
        rows.append(entry)
    return rows


def load_guide_eval(path: Path) -> pl.DataFrame:
    """Load a guide-eval CSV file into a DataFrame.

    Returns a DataFrame with columns:
        goal, guide, zs_distance, structural_overlap, structural_zs_sum,
        zs_rank, structural_rank, reached, iterations_to_reach
    """
    df = pl.read_csv(path)
    # Derive 'reached' from iterations_to_reach (null = not reached)
    df = df.with_columns(pl.col("iterations_to_reach").is_not_null().alias("reached"))
    return df
