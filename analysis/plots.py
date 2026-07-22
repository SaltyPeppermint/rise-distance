"""Altair chart specs for the guide-selection analysis.

Everything consumes the tidy long frame from `helpers.load_runs` (one row per
leg, tagged with `mode`, carrying `r_<metric>` = metric / seed-baseline). Each
run now uses a single guide-set size `k`, so `k` is no longer an axis — the
recurring shape is *one point per seed/goal pair, pairs sorted by value along
x with no per-pair labels, one facet per metric, one color per mode*.
`pair_strip` builds that once and the notebook calls it with a different value
column each time; ratio plots carry a dashed break-even line at 1.0.

Categorical colors use the validated 8-hue order from the dataviz palette
(fixed, never cycled) so a mode keeps its color as runs are added/removed.
"""

from collections.abc import Sequence

import altair as alt
import polars as pl

# Validated categorical hues (dataviz reference palette, light-surface steps),
# in fixed order. Modes are assigned slots by their position in `modes`.
PALETTE = [
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Applied via `alt.themes` in the notebook; keeps grids recessive and sizing sane.
THEME = {
    "config": {
        "view": {"continuousWidth": 320, "continuousHeight": 260, "strokeOpacity": 0},
        "axis": {
            "grid": True,
            "gridColor": "#e8e8e6",
            "domainColor": "#c9c9c6",
            "tickColor": "#c9c9c6",
        },
        "axisX": {"labelAngle": 0},
        "legend": {"orient": "top", "titleFontSize": 11, "labelFontSize": 11},
        "title": {"fontSize": 13, "anchor": "start"},
        "range": {"category": PALETTE},
    }
}


def _color(modes: Sequence[str]) -> alt.Color:
    """Stable mode->hue mapping (domain fixes slot order across runs)."""
    return alt.Color(
        "mode:N",
        scale=alt.Scale(domain=list(modes), range=PALETTE[: len(modes)]),
        legend=alt.Legend(title=None),
    )


def _pair_x() -> alt.X:
    """Rank-ordered pair axis: each pair a slot, no labels (hundreds of pairs)."""
    return alt.X(
        "rank:Q",
        title="seed/goal pair (sorted by value)",
        axis=alt.Axis(labels=False, ticks=False),
    )


def _title(title: str, meta: dict) -> alt.TitleParams:
    return alt.TitleParams(
        title,
        subtitle=f"{meta['n_goals']} goals × {meta['n_trials']} attempts",
        subtitleColor="#7a7a77",
    )


def pair_strip(
    df: pl.DataFrame,
    value: str,
    meta: dict,
    *,
    title: str,
    y_title: str,
    metrics: Sequence[str],
    breakeven: bool = True,
    group_by: Sequence[str] = ("seed", "goal"),
    group_reduce: str = "median",
    columns: int = 3,
) -> alt.Chart:
    """One point per seed/goal pair, pairs sorted by `value` along x, per metric.

    Metric facets wrap after `columns` per row (default 3). Within each
    ``(mode, metric)`` facet the pairs are ranked by their `value` and that rank
    is the x position, so the strip reads as a sorted curve; per-pair labels are
    hidden (there can be hundreds).

    `df` is long with columns `mode`, `metric`, `value`, and the pair keys
    (`seed`, `goal`). Callers that start from the wide tidy frame melt the
    relevant `r_<metric>`/cost columns into (`metric`, `value`) first — see the
    notebook.

    `group_by` collapses each pair's attempts into a single point per
    ``(mode, metric, *group_by)`` using `group_reduce` (default: median over the
    reached attempts). ``group_by=["goal"]`` with ``group_reduce="min"`` gives
    "best guide per goal".

    Ratios are heavy-tailed, so per-pair points reveal the spread that a single
    band would hide.
    """
    modes = [m for m in meta["modes"] if m in df["mode"].unique().to_list()]
    per_pair = (
        df.drop_nulls(value)
        .group_by("mode", "metric", *group_by)
        .agg(getattr(pl.col(value), group_reduce)().alias("v"))
        # Rank pairs by value within each (mode, metric) facet -> sorted x.
        .with_columns((pl.col("v").rank("ordinal").over("mode", "metric") - 1).alias("rank"))
    )

    x = _pair_x()
    points = (
        alt.Chart(per_pair)
        .mark_circle(size=18, opacity=0.55)
        .encode(
            x=x,
            y=alt.Y("v:Q", title=y_title),
            color=_color(modes),
            tooltip=[
                "mode:N",
                "metric:N",
                *[f"{g}:N" for g in group_by],
                alt.Tooltip("v:Q", format=".3f"),
            ],
        )
    )
    layers = [points]
    if breakeven:
        layers.append(
            alt.Chart(per_pair)
            .mark_rule(strokeDash=[4, 4], color="#555", opacity=0.7)
            .encode(y=alt.datum(1.0))
        )

    return (
        alt.layer(*layers)
        .facet(facet=alt.Facet("metric:N", title=None, sort=list(metrics)), columns=columns)
        .properties(title=_title(title, meta))
        .resolve_scale(y="independent", x="independent")
    )


def reachability(df: pl.DataFrame, gr: pl.DataFrame, meta: dict) -> alt.VConcatChart:
    """Two panels: per-pair reach rate (sorted strip) and a per-mode summary bar.

    `gr` is `helpers.goal_reach(df)`, one row per (mode, seed, goal) carrying
    `reach_rate` = fraction of that pair's attempts that reached. The strip ranks
    pairs by reach_rate within each mode (hardest at left) so the shape of the
    reachability distribution is visible; the summary bar shows overall leg reach
    rate and goal coverage per mode.
    """
    modes = meta["modes"]
    color = _color(modes)

    # Per-pair reach rate, ranked within mode -> sorted strip over pairs.
    strip_df = gr.with_columns(
        (pl.col("reach_rate").rank("ordinal").over("mode") - 1).alias("rank")
    )
    strip = (
        alt.Chart(strip_df)
        .mark_circle(size=16, opacity=0.6)
        .encode(
            x=_pair_x(),
            y=alt.Y("reach_rate:Q", title="reach rate over attempts", scale=alt.Scale(domain=[0, 1])),
            color=color,
            tooltip=[
                "mode:N",
                "seed:N",
                "goal:N",
                alt.Tooltip("reach_rate:Q", format=".0%"),
            ],
        )
        .properties(title="Per-pair reach rate (sorted)")
    )

    # Per-mode summary: overall leg reach rate and goal coverage (≥1 success).
    summ = (
        pl.concat(
            [
                df.group_by("mode")
                .agg((pl.col("reached").mean() * 100).alias("pct"))
                .with_columns(pl.lit("leg reach rate").alias("kind")),
                gr.group_by("mode")
                .agg(((pl.col("reach_rate") > 0).mean() * 100).alias("pct"))
                .with_columns(pl.lit("goal coverage").alias("kind")),
            ],
            how="vertical",
        )
    )
    summary = (
        alt.Chart(summ)
        .mark_bar()
        .encode(
            x=alt.X("pct:Q", title="%", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("mode:N", title=None, sort=list(modes)),
            color=color,
            tooltip=["mode:N", "kind:N", alt.Tooltip("pct:Q", format=".1f")],
        )
        .properties(height=alt.Step(18))
        .facet(row=alt.Row("kind:N", title=None))
        .properties(title="Reachability summary by mode")
    )

    return alt.vconcat(strip, summary).properties(title=_title("Reachability summary", meta))


def cost_boxplots(df: pl.DataFrame, meta: dict, metrics: Sequence[str]) -> alt.Chart:
    """Box plots of reached-leg cost by mode, one facet per metric."""
    modes = meta["modes"]
    long = (
        df.filter(pl.col("reached"))
        .unpivot(index=["mode"], on=list(metrics), variable_name="metric", value_name="v")
        .drop_nulls("v")
    )
    return (
        alt.Chart(long)
        .mark_boxplot()
        .encode(
            x=alt.X("mode:N", title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("v:Q", title=None),
            color=_color(modes),
        )
        .properties(width=140, height=180)
        .facet(column=alt.Column("metric:N", sort=list(metrics), title=None))
        .resolve_scale(y="independent")
        .properties(title=_title("Cost distribution by mode", meta))
    )


def reach_heatmap(gr: pl.DataFrame, meta: dict) -> alt.Chart:
    """Goal × mode reachability heatmap (hardest goals at top).

    Goals are ranked by their mean reach_rate across modes, and that rank is the
    y position, so each mode column shares the same goal ordering. Hundreds of
    goals in a few hundred pixels means each row is sub-pixel, so an *ordinal* y
    (one axis band per goal) both stripes the image and is slow; we instead give
    each cell an explicit ``gy → gy+1`` quantitative band via `y`/`y2` so the
    rects tile seamlessly, and drop the rect stroke. The y scale is reversed so
    rank 1 (hardest) sits at the top.
    """
    order = (
        gr.group_by("goal")
        .agg(pl.col("reach_rate").mean().alias("avg"))
        .with_columns(pl.col("avg").rank("ordinal").alias("gy"))
    )
    plot = gr.join(order.select("goal", "gy"), on="goal").with_columns(
        (pl.col("gy") - 1).alias("gy0"), pl.col("gy").alias("gy1")
    )
    n_goals = plot.select(pl.col("gy").max()).item()
    return (
        alt.Chart(plot)
        .mark_rect(stroke=None, strokeWidth=0)
        .encode(
            x=alt.X("mode:N", title=None, sort=list(meta["modes"]), axis=alt.Axis(labelAngle=-40)),
            y=alt.Y(
                "gy0:Q",
                title="goal (sorted by reachability, hardest at top)",
                scale=alt.Scale(domain=[0, n_goals], reverse=True, nice=False),
                axis=alt.Axis(labels=False, ticks=False, grid=False),
            ),
            y2="gy1:Q",
            color=alt.Color(
                "reach_rate:Q",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                legend=alt.Legend(title="reach rate", format="%"),
            ),
            tooltip=["mode:N", "goal:N", alt.Tooltip("reach_rate:Q", format=".0%")],
        )
        .properties(width=alt.Step(46), height=460)
        .properties(title=_title("Goal-level reachability heatmap", meta))
    )
