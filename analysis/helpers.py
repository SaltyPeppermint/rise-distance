from pathlib import Path
import json

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# For kendalltau on large datasets, subsample to this size (O(n²) algorithm).
KENDALL_MAX_N = 50_000


def compute_correlations(
    data: pl.DataFrame, predictors: list[str], target: str
) -> pl.DataFrame:
    """Compute Spearman rho, Kendall tau, and OLS R² for each predictor vs target.

    OLS stats are computed via closed-form formulas (no sklearn overhead).
    Kendall tau is subsampled when n > KENDALL_MAX_N to avoid O(n²) blowup.
    """
    reached = data.filter(pl.col("reached")).drop_nulls(subset=[target])
    if len(reached) < 3:
        return pl.DataFrame()

    # Pull all needed columns once
    arrays: dict[str, np.ndarray] = {}
    for c in [target] + [p for p in predictors if p in reached.columns]:
        arrays[c] = reached[c].to_numpy().astype(float)

    y_full = arrays[target]
    n = len(y_full)

    rows = []
    for pred in predictors:
        if pred not in arrays:
            continue
        x = arrays[pred]

        # ── Spearman (scipy, fast O(n log n)) ──
        rho, rho_p = spearmanr(x, y_full)

        # ── Kendall tau (subsample if large) ──
        if n > KENDALL_MAX_N:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, KENDALL_MAX_N, replace=False)
            tau, tau_p = kendalltau(x[idx], y_full[idx])
        else:
            tau, tau_p = kendalltau(x, y_full)

        # ── OLS via closed-form (avoids sklearn object overhead) ──
        x_mean = x.mean()
        y_mean = y_full.mean()
        x_centered = x - x_mean
        ss_xx = x_centered @ x_centered
        ss_xy = x_centered @ (y_full - y_mean)
        slope = ss_xy / ss_xx if ss_xx > 1e-12 else 0.0
        intercept = y_mean - slope * x_mean
        residuals = y_full - (slope * x + intercept)
        ss_res = residuals @ residuals
        ss_tot = (y_full - y_mean) @ (y_full - y_mean)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        mse = ss_res / max(n - 2, 1)
        se_slope = np.sqrt(mse / ss_xx) if ss_xx > 1e-12 else 0.0

        rows.append(
            {
                "predictor": pred,
                "n": n,
                "spearman_rho": round(float(rho), 4),  # pyright: ignore[reportArgumentType]
                "spearman_p": round(float(rho_p), 6),  # pyright: ignore[reportArgumentType]
                "kendall_tau": round(float(tau), 4),  # pyright: ignore[reportArgumentType]
                "kendall_p": round(float(tau_p), 6),  # pyright: ignore[reportArgumentType]
                "R²": round(float(r2), 4),
                "slope": round(float(slope), 4),
                "SE_slope": round(float(se_slope), 4),
                "intercept": round(float(intercept), 4),
            }
        )
    return pl.DataFrame(rows)


SCATTER_MAX_POINTS = 20_000


def plot_predictor_vs_iters(
    data: pl.DataFrame,
    predictor: str,
    target: str = "iterations_to_reach",
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


def find_latest_run(pattern: str) -> Path:
    """Find the most recently modified run directory matching `pattern` in DATA_DIR."""
    data_dir = Path(__file__).parent / "../data/guide_eval"
    candidates = sorted(
        [f for f in data_dir.iterdir() if f.is_dir() and pattern in f.name],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No '{pattern}' directories in {data_dir.resolve()}")
    return candidates[-1]


def parse_replacement_summary(raw: list[dict], strategy_name: str) -> list[dict]:
    """Parse a no/with_replacement_top_k_summary.json into flat rows.

    Format: [{seed, goal, entries_per_k: {k: [{'Ok': {...}} | {'Err': ...}]}}]

    Each row has: goal, strategy, k, iters, nodes, classes, total_applied,
    total_time, not_enough_samples, unreached, panic_while_sample.
    """
    rows = []
    for entry in raw:
        goal = entry["goal"]
        for k_str, trials in entry["entries_per_k"].items():
            k = int(k_str)
            for trial in trials:
                if "Ok" in trial:
                    t = trial["Ok"]
                    rows.append(
                        {
                            "goal": goal,
                            "strategy": strategy_name,
                            "k": k,
                            "iters": t["iters"],
                            "nodes": t["nodes"],
                            "classes": t["classes"],
                            "total_applied": t["total_applied"],
                            "total_time": t["total_time"],
                            "not_enough_samples": False,
                            "unreached": False,
                            "panic_while_sample": False,
                        }
                    )
                else:
                    err = trial.get("Err", "")
                    rows.append(
                        {
                            "goal": goal,
                            "strategy": strategy_name,
                            "k": k,
                            "iters": None,
                            "nodes": None,
                            "classes": None,
                            "total_applied": None,
                            "total_time": None,
                            "not_enough_samples": err == "NotEnoughSamples",
                            "unreached": err == "Unreached",
                            "panic_while_sample": err == "PanicWhileSample",
                        }
                    )
    return rows


def load_top_k(
    run_dir: Path, strategy_name: str, with_replacement: bool = False
) -> pl.DataFrame:
    """Load trial rows from a run directory, preferring parquet over JSON summary.

    Pass ``with_replacement=True`` to load the with-replacement top-k data
    instead of the default no-replacement variant.

    Adjusts `nodes` and `total_time` to account for the guide egraph overhead:
      - nodes = max(trial_nodes, guide_egraph_nodes)
      - total_time += guide_eqsat_time
    """
    with open(run_dir / "stats.json", encoding="utf-8") as f:
        run_stats = json.load(f)[0]
    guide_nodes = run_stats["guide_egraph_nodes"]
    guide_time = run_stats["guide_eqsat_time"]

    prefix = "with_replacement" if with_replacement else "no_replacement"
    parquet_path = run_dir / f"{prefix}_top_k_summary.parquet"
    if parquet_path.exists():
        df = pl.read_parquet(parquet_path).with_columns(
            pl.lit(strategy_name).alias("strategy")
        )
    else:
        summary_path = run_dir / f"{prefix}_top_k_summary.json"
        with open(summary_path, encoding="utf-8") as f:
            raw = json.load(f)
        df = pl.DataFrame(parse_replacement_summary(raw, strategy_name))

    return df.with_columns(
        pl.when(pl.col("nodes").is_not_null())
        .then(pl.max_horizontal(pl.col("nodes"), pl.lit(guide_nodes)))
        .otherwise(None)
        .alias("nodes"),
        pl.when(pl.col("total_time").is_not_null())
        .then(pl.col("total_time") + guide_time)
        .otherwise(None)
        .alias("total_time"),
    )


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
