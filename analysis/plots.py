from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import polars as pl


def configure_k_axis(ax: Axes, k_values: list[int]) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(k_values)


def finish_fig(fig: Figure, title: str | None = None, fontsize: float = 14) -> None:
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    plt.show()


def plot_metric_boxplots(
    df: pl.DataFrame,
    label: str,
    metric: str,
    ylabel: str,
    k_values: list[int],
    strat_names: list[str],
    palette: Sequence[Any],
) -> None:
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(4 * n_k, 5), squeeze=False, sharey=True)
    for ki, k in enumerate(k_values):
        ax = axes[0][ki]
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
        ax.set_ylabel(ylabel if ki == 0 else "")
    finish_fig(fig, f"{ylabel} by strategy at each k [{label}]", fontsize=13)
