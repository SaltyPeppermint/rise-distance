import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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
    n_goals: int | None = None,
    n_trials: int | None = None,
    tight_layout: bool = True,
) -> None:
    if title is not None:
        if n_goals is not None and n_trials is not None:
            full_title = f"{title} [{ds_name}]\n({n_goals} goals × {n_trials} trials per k)"
        else:
            full_title = f"{title} [{ds_name}]"
        fig.suptitle(full_title, fontsize=fontsize)
    if tight_layout:
        fig.tight_layout()
    _PLOTS_DIR.mkdir(exist_ok=True)
    slug = f"{title}-{ds_name}".replace(" ", "_").replace("/", "-")
    fig.savefig(_PLOTS_DIR / f"{slug}.png", bbox_inches="tight")
    plt.show()


def split_modes(
    sampling_modes: list[tuple[str, Path, bool]],
) -> tuple[list[tuple[int, str, Path, bool]], list[tuple[int, str, Path, bool]]]:
    multi = [(i, n, p, s) for i, (n, p, s) in enumerate(sampling_modes) if not s]
    single = [(i, n, p, s) for i, (n, p, s) in enumerate(sampling_modes) if s]
    return multi, single


def plot_reachability_summary(
    mode_dfs: dict[str, pl.DataFrame],
    sampling_modes: list[tuple[str, Path, bool]],
    label: str,
    k_values: list[int],
    palette: Sequence[Any],
    mode_style,
    compute_goal_reach,
    n_goals: int | None = None,
    n_trials: int | None = None,
) -> None:
    """Single reachability figure with three panels:

    - left:   reach rate vs k (fraction of legs reaching the goal)
    - middle: fraction of goals with >=1 successful trial vs k (coverage)
    - right:  CDF of per-goal reachability, per (mode, k)

    `mode_style` and `compute_goal_reach` are passed in to avoid a circular
    import back into the notebook / helpers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_rate, ax_cov, ax_cdf = axes

    for i, (mode_name, _run_dir, is_single) in enumerate(sampling_modes):
        style = mode_style(i, is_single)
        df = mode_dfs[mode_name]

        # left: leg-level reach rate vs k
        reach = df.group_by("k").agg(pl.col("reached").mean().alias("reach_rate")).sort("k")
        ax_rate.plot(reach["k"], reach["reach_rate"] * 100, label=mode_name, **style)

        # middle: fraction of goals with >=1 success vs k
        any_success = df.group_by("goal", "k").agg(pl.col("reached").any().alias("any_success"))
        cov = (
            any_success.group_by("k")
            .agg(pl.col("any_success").mean().alias("frac_goals"))
            .sort("k")
        )
        ax_cov.plot(cov["k"], cov["frac_goals"] * 100, label=mode_name, **style)

        # right: CDF of per-goal reachability per k
        goal_reach = compute_goal_reach(df)
        mode_k_values = sorted(df["k"].unique().to_list())
        for j, k in enumerate(mode_k_values):
            reach_at_k = goal_reach.filter(pl.col("k") == k).sort("reach_rate")["reach_rate"]
            ax_cdf.plot(
                reach_at_k,
                np.linspace(0, 1, len(reach_at_k)),
                label=f"{mode_name} k={k}",
                color=style["color"],
                linestyle=["-", "--", ":", "-."][j % 4],
            )

    ax_rate.set_xlabel("k (number of guides)")
    ax_rate.set_ylabel("Reachability rate (%)")
    ax_rate.set_title("Leg reachability rate vs k")
    configure_k_axis(ax_rate, k_values)
    ax_rate.set_ylim(bottom=0)
    ax_rate.legend(fontsize=8)

    ax_cov.set_xlabel("k (number of guides)")
    ax_cov.set_ylabel("Goals with >= 1 successful trial (%)")
    ax_cov.set_title("Goal coverage vs k")
    configure_k_axis(ax_cov, k_values)
    ax_cov.set_ylim(-5, 105)
    ax_cov.legend(fontsize=8)

    ax_cdf.set_xlabel("Per-goal reachability rate")
    ax_cdf.set_ylabel("Fraction of goals (CDF)")
    ax_cdf.set_title("CDF of per-goal reachability by k")
    ax_cdf.legend(fontsize=7)

    finish_fig(fig, "Reachability summary", label, n_goals=n_goals, n_trials=n_trials)


def plot_cost_boxplots(
    df: pl.DataFrame,
    label: str,
    metrics: list[tuple[str, str]],
    k_values: list[int],
    strat_names: list[str],
    palette: Sequence[Any],
    n_goals: int | None = None,
    n_trials: int | None = None,
    group_col: str = "strategy",
) -> None:
    """One figure of cost-distribution box plots: one row per (metric, ylabel)
    in `metrics`, one column per k.
    """
    n_rows = len(metrics)
    n_cols = len(k_values)
    # constrained layout (not tight_layout): tight_layout NaNs out the axis
    # geometry on this sharey grid under mpl 3.11, which then crashes savefig.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 5 * n_rows),
        squeeze=False,
        sharey="row",
        layout="constrained",
    )
    for ri, (metric, ylabel) in enumerate(metrics):
        for ki, k in enumerate(k_values):
            ax = axes[ri][ki]
            box_data, tick_labels, colors = [], [], []
            for i, strat in enumerate(strat_names):
                vals = df.filter((pl.col("k") == k) & (pl.col(group_col) == strat))[
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
            if ri == 0:
                ax.set_title(f"k = {k}")
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            ax.set_ylabel(ylabel if ki == 0 else "")
    finish_fig(
        fig,
        "Cost distribution by strategy at each k",
        label,
        fontsize=13,
        tight_layout=False,
        n_goals=n_goals,
        n_trials=n_trials,
    )


def _normalize_by_seed_baseline(
    df: pl.DataFrame,
    per_seed_baseline: dict[str, dict[str, float]],
    metric: str,
) -> pl.DataFrame:
    """Attach ``ratio = metric / seed_baseline[metric]`` to each reached leg.

    The join is on `seed`: every goal under a seed shares that seed's unguided
    baseline, so this pairs each guided leg with its own reference before any
    aggregation. Legs whose seed has no Ok baseline, or whose metric is null
    (unreached or empty-pool), are dropped.
    """
    base = pl.DataFrame(
        {
            "seed": list(per_seed_baseline.keys()),
            "_base": [b[metric] for b in per_seed_baseline.values()],
        }
    )
    return (
        df.join(base, on="seed", how="inner")
        .filter(pl.col(metric).is_not_null() & (pl.col("_base") > 0))
        .with_columns((pl.col(metric) / pl.col("_base")).alias("ratio"))
    )


def plot_normalized_vs_baseline(
    mode_dfs: dict[str, pl.DataFrame],
    sampling_modes: list[tuple[str, Path, bool]],
    per_seed_baselines: dict[str, dict[str, dict[str, float]]],
    label: str,
    k_values: list[int],
    metrics: list[str],
    mode_style,
    show_per_seed: bool = True,
    n_goals: int | None = None,
    n_trials: int | None = None,
) -> None:
    """Guided cost relative to each seed's own unguided baseline, vs k.

    For every reached leg we divide its metric by that seed's baseline, so the
    reference is a horizontal line at ``ratio = 1.0`` (guides break even); below
    1 means cheaper than unguided. Per k we draw the median ratio with a 25th-75th
    percentile band (ratios are heavy-tailed, so the median suits them better
    than a mean). With `show_per_seed`, faint per-seed median curves sit behind
    the aggregate to show the seed-to-seed spread.

    `per_seed_baselines[mode_name]` is the `load_baseline_per_seed` dict for that
    mode's run.
    """
    n_cols = min(2, len(metrics))
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = list(axes.flat)

    for metric, ax in zip(metrics, axes_flat):
        for i, (mode_name, _run_dir, is_single) in enumerate(sampling_modes):
            style = mode_style(i, is_single)
            color = style["color"]
            df = mode_dfs[mode_name].filter(pl.col("reached"))
            norm = _normalize_by_seed_baseline(df, per_seed_baselines[mode_name], metric)
            if norm.height == 0:
                continue

            if show_per_seed and not is_single:
                per_seed = (
                    norm.group_by("seed", "k").agg(pl.col("ratio").median().alias("m")).sort("k")
                )
                for _seed, sdf in per_seed.group_by("seed"):
                    ax.plot(
                        sdf["k"],
                        sdf["m"],
                        color=color,
                        alpha=0.12,
                        linewidth=0.6,
                        zorder=1,
                    )

            agg = (
                norm.group_by("k")
                .agg(
                    pl.col("ratio").median().alias("median"),
                    pl.col("ratio").quantile(0.25).alias("q25"),
                    pl.col("ratio").quantile(0.75).alias("q75"),
                )
                .sort("k")
            )
            ks = agg["k"]
            if is_single:
                ax.plot(ks, agg["median"], label=mode_name, **style)
            else:
                ax.fill_between(ks, agg["q25"], agg["q75"], alpha=0.2, color=color, zorder=3)
                # bump zorder so the line sits above the band and per-seed curves.
                ax.plot(ks, agg["median"], label=mode_name, **{**style, "zorder": 4})

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="baseline")
        ax.set_xlabel("k (number of guides)")
        ax.set_ylabel(f"{metric} / seed baseline")
        ax.set_title(f"{metric} relative to unguided (median, IQR band)")
        ax.legend(fontsize=7)
        configure_k_axis(ax, k_values)

    for ax in axes_flat[len(metrics) :]:
        ax.set_visible(False)

    finish_fig(
        fig,
        "Cost relative to per-seed baseline",
        label,
        n_goals=n_goals,
        n_trials=n_trials,
    )


def plot_best_ratio_per_goal(
    mode_dfs: dict[str, pl.DataFrame],
    sampling_modes: list[tuple[str, Path, bool]],
    per_seed_baselines: dict[str, dict[str, dict[str, float]]],
    label: str,
    k_values: list[int],
    metrics: list[str],
    mode_style,
    compute_best_ratio_per_goal,
    show_per_goal: bool = False,
    n_goals: int | None = None,
    n_trials: int | None = None,
) -> None:
    """Best-guide improvement over baseline, per goal, vs k.

    For each goal we take the best sampled guide at each k (the minimum of
    ``metric / seed_baseline`` over all guides that reached that goal), then
    summarise those per-goal bests across goals with the median and a 25th-75th
    percentile band. ``ratio = 1.0`` (dashed line) is break-even with unguided;
    below 1 means the best guide is cheaper. This is per-goal (not per-seed) and
    unitless.

    `compute_best_ratio_per_goal` is passed in (the `helpers` function) to avoid
    a circular import. With `show_per_goal`, faint per-goal points sit behind the
    aggregate to show the spread.
    """
    n_cols = min(3, len(metrics))
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = list(axes.flat)

    for metric, ax in zip(metrics, axes_flat):
        for i, (mode_name, _run_dir, is_single) in enumerate(sampling_modes):
            style = mode_style(i, is_single)
            color = style["color"]
            df = mode_dfs[mode_name].filter(pl.col("reached"))
            best = compute_best_ratio_per_goal(df, per_seed_baselines[mode_name], [metric]).filter(
                pl.col("metric") == metric
            )
            if best.height == 0:
                continue

            if show_per_goal and not is_single:
                ax.scatter(
                    best["k"],
                    best["best_ratio"],
                    color=color,
                    alpha=0.1,
                    s=6,
                    zorder=1,
                )

            agg = (
                best.group_by("k")
                .agg(
                    pl.col("best_ratio").median().alias("median"),
                    pl.col("best_ratio").quantile(0.25).alias("q25"),
                    pl.col("best_ratio").quantile(0.75).alias("q75"),
                )
                .sort("k")
            )
            ks = agg["k"]
            if is_single:
                ax.plot(ks, agg["median"], label=mode_name, **style)
            else:
                ax.fill_between(ks, agg["q25"], agg["q75"], alpha=0.2, color=color, zorder=3)
                ax.plot(ks, agg["median"], label=mode_name, **{**style, "zorder": 4})

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="baseline")
        ax.set_xlabel("k (number of guides)")
        ax.set_ylabel(f"best {metric} / seed baseline")
        ax.set_title(f"best guide's {metric} relative to unguided\n(median over goals, IQR band)")
        ax.legend(fontsize=7)
        configure_k_axis(ax, k_values)

    for ax in axes_flat[len(metrics) :]:
        ax.set_visible(False)

    finish_fig(
        fig,
        "Best-guide improvement over baseline (per goal)",
        label,
        n_goals=n_goals,
        n_trials=n_trials,
    )
