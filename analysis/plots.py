import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import polars as pl

_PLOTS_DIR = Path(__file__).parent / "plots"


def configure_k_axis(ax: Axes, k_values: list[int]) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(k_values)


def finish_fig(
    fig: Figure,
    title: str,
    ds_name: str,
    fontsize: float = 14,
    n: int | None = None,
    tight_layout: bool = True,
) -> None:
    if title is not None:
        full_title = (
            f"{title} [{ds_name}] (n={n} per k)" if n is not None else f"{title} [{ds_name}]"
        )
        fig.suptitle(full_title, fontsize=fontsize)
    if tight_layout:
        fig.tight_layout()
    _PLOTS_DIR.mkdir(exist_ok=True)
    slug = f"{title}-{ds_name}".replace(" ", "_").replace("/", "-")
    fig.savefig(_PLOTS_DIR / f"{slug}.png", bbox_inches="tight")
    plt.show()


def annotate_k_points(ax: Axes, xs: list, ys: list, ks: list, scale: float = 1.0) -> None:
    for k_val, x, y in zip(ks, xs, ys):
        ax.annotate(
            f"k={k_val}",
            (x, y * scale),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )


def plot_mean_with_best_dotted(
    ax: Axes,
    ks,
    values,
    best_values,
    style: dict,
    label: str,
    best_label: str,
) -> None:
    ax.plot(ks, values, label=label, **style)
    ax.plot(
        ks,
        best_values,
        color=style["color"],
        linestyle="dotted",
        linewidth=1.5,
        label=best_label,
    )


def split_modes(
    sampling_modes: list[tuple[str, str, bool]],
) -> tuple[list[tuple[int, str, str, bool]], list[tuple[int, str, str, bool]]]:
    multi = [(i, n, p, s) for i, (n, p, s) in enumerate(sampling_modes) if not s]
    single = [(i, n, p, s) for i, (n, p, s) in enumerate(sampling_modes) if s]
    return multi, single


def plot_metric_boxplots(
    df: pl.DataFrame,
    label: str,
    metric: str,
    ylabel: str,
    k_values: list[int],
    strat_names: list[str],
    palette: Sequence[Any],
    n: int | None = None,
) -> None:
    n_k = len(k_values)
    n_cols = min(3, n_k)
    n_rows = math.ceil(n_k / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False, sharey=True
    )
    for ki, k in enumerate(k_values):
        ax = axes[ki // n_cols][ki % n_cols]
        box_data, tick_labels, colors = [], [], []
        for i, strat in enumerate(strat_names):
            vals = df.filter((pl.col("k") == k) & (pl.col("strategy") == strat))[
                metric
            ].drop_nulls()
            if len(vals) > 0:
                box_data.append(vals)
                tick_labels.append(strat)
                colors.append(palette[i % len(palette)])
        if box_data:
            bp = ax.boxplot(box_data, tick_labels=tick_labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for mean in bp["means"]:
                mean.set_color("red")
        ax.set_title(f"k = {k}")
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel if ki % n_cols == 0 else "")
    for ki in range(n_k, n_rows * n_cols):
        axes[ki // n_cols][ki % n_cols].set_visible(False)
    finish_fig(fig, f"{ylabel} by strategy at each k", label, fontsize=13, n=n)
