import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression


def compute_correlations(
    data: pl.DataFrame, predictors: list[str], target: str
) -> pl.DataFrame:
    """Compute Spearman rho, Kendall tau, and OLS R² for each predictor vs target."""
    reached = data.filter(pl.col("reached")).drop_nulls(subset=[target])
    rows = []
    for pred in predictors:
        if pred not in reached.columns:
            continue
        col = reached[pred].drop_nulls()
        tgt = reached[target].drop_nulls()
        # Align lengths
        n = min(len(col), len(tgt))
        if n < 3:
            continue
        x = col[:n].to_numpy().astype(float)
        y = tgt[:n].to_numpy().astype(float)

        rho, rho_p = spearmanr(x, y)
        tau, tau_p = kendalltau(x, y)

        # OLS
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        r2 = reg.score(x.reshape(-1, 1), y)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        # Standard error of slope
        y_pred = reg.predict(x.reshape(-1, 1))
        residuals = y - y_pred
        mse = np.sum(residuals**2) / max(n - 2, 1)
        ss_xx = np.sum((x - x.mean()) ** 2)
        se_slope = np.sqrt(mse / ss_xx) if ss_xx > 1e-12 else 0.0

        rows.append(
            {
                "predictor": pred,
                "n": n,
                "spearman_rho": round(rho, 4),
                "spearman_p": round(rho_p, 6),
                "kendall_tau": round(tau, 4),
                "kendall_p": round(tau_p, 6),
                "R²": round(r2, 4),
                "slope": round(slope, 4),
                "SE_slope": round(se_slope, 4),
                "intercept": round(intercept, 4),
            }
        )
    return pl.DataFrame(rows)


def plot_predictor_vs_iters(
    data: pl.DataFrame,
    predictor: str,
    target: str = "iterations_to_reach",
    title_suffix: str = "",
    ax: plt.Axes | None = None,
):
    """Scatter plot of predictor vs iterations, with reached/timed-out markers and OLS line."""
    if ax is None:
        _, ax = plt.subplots()

    reached = data.filter(pl.col("reached")).drop_nulls(subset=[target])
    timed_out = data.filter(~pl.col("reached"))

    # Reached points
    x_r = reached[predictor].to_numpy().astype(float)
    y_r = reached[target].to_numpy().astype(float)
    ax.scatter(
        x_r, y_r, c="steelblue", s=20, alpha=0.6, label=f"reached (n={len(x_r)})"
    )

    # Timed-out points (plotted at max_iters + 1)
    if len(timed_out) > 0:
        x_t = timed_out[predictor].to_numpy().astype(float)
        timeout_y = y_r.max() + 1 if len(y_r) > 0 else 1
        ax.scatter(
            x_t,
            np.full_like(x_t, timeout_y),
            c="red",
            marker="x",
            s=20,
            alpha=0.5,
            label=f"timed out (n={len(x_t)})",
        )
        ax.axhline(timeout_y, color="red", ls="--", alpha=0.3, lw=1)

    # OLS regression line (reached only)
    if len(x_r) >= 2:
        reg = LinearRegression().fit(x_r.reshape(-1, 1), y_r)
        x_line = np.linspace(x_r.min(), x_r.max(), 100)
        ax.plot(
            x_line,
            reg.predict(x_line.reshape(-1, 1)),
            c="green",
            lw=2,
            alpha=0.7,
            label=f"OLS (R²={reg.score(x_r.reshape(-1, 1), y_r):.3f})",
        )

    ax.set_xlabel(predictor)
    ax.set_ylabel(target)
    ax.set_title(f"{predictor} vs {target}{title_suffix}")
    ax.legend(fontsize=8)
    return ax
