"""Altair chart specs for the guide-selection analysis.

Everything consumes the tidy long frame from `helpers.load_runs` (one row per
leg, tagged with `mode`). Each run now uses a single guide-set size `k`, so `k`
is no longer an axis — the recurring shape is *one point per seed/goal pair,
pairs sorted by value along x with no per-pair labels, one facet per metric*.
`abs_pair_strip` builds that in absolute native units, overlaying each pair's
unguided baseline as a second series.

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


# Fixed hues for the series overlay, so each keeps its color independent of how
# many modes are present.
SERIES_COLORS = {"leg": PALETTE[0], "baseline": PALETTE[5], "guide": PALETTE[1]}  # blue/orange/green


def abs_pair_strip(
    df: pl.DataFrame,
    meta: dict,
    *,
    title: str,
    y_title: str,
    metrics: Sequence[str],
    limits: pl.DataFrame | None = None,
    group_by: Sequence[str] = ("seed", "goal"),
    group_reduce: str = "median",
    log_y: bool = True,
) -> alt.Chart:
    """Absolute-unit sibling of `pair_strip`: each series' cost vs the baseline.

    `df` is long with columns `mode`, `metric`, `series` (``"leg"`` /
    ``"baseline"`` / ``"guide"``), the pair keys, and `value` in native units —
    build it with the notebook's `absolute_long`. Each pair contributes one point
    per series per metric; pairs are ranked by their **leg** value (shared across
    series) so the strip reads as a sorted leg curve with the other series drawn
    at the same x.

    Color is the series. Mode is the facet **row** and metric the **column**, so
    overlaying several runs stacks them as rows without the two encodings fighting
    over the color channel. Every cell has an independent y (per-seed scale varies
    wildly across metrics and runs).

    `limits` is an optional ``(mode, series, metric, limit)`` frame (from
    `helpers.series_limit_frame`); when given, each facet gets a dashed rule *per
    series*, colored to match its points, at the ceiling that series ran under —
    the guide/leg replay budget and the baseline's search-phase limit can differ.
    Metrics absent from the frame — e.g. `classes`, which egg doesn't bound — get
    no line.

    `y` is log-scaled by default (`log_y`): absolute costs mix per-seed scale and
    span orders of magnitude — the very spread the ratio plots normalized away —
    so a linear axis squashes everything.
    """
    per_pair = (
        df.drop_nulls("value")
        .filter(pl.col("value") > 0 if log_y else pl.lit(True))
        .group_by("mode", "metric", "series", *group_by)
        .agg(getattr(pl.col("value"), group_reduce)().alias("v"))
    )
    # Rank pairs by the LEG value within each (mode, metric); share that rank with
    # the pair's other series so all series line up on x.
    leg_rank = (
        per_pair.filter(pl.col("series") == "leg")
        .with_columns((pl.col("v").rank("ordinal").over("mode", "metric") - 1).alias("rank"))
        .select("mode", "metric", *group_by, "rank")
    )
    per_pair = per_pair.join(leg_rank, on=["mode", "metric", *group_by], how="left")

    scale = alt.Scale(type="log") if log_y else alt.Scale()
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=list(SERIES_COLORS), range=list(SERIES_COLORS.values())),
        legend=alt.Legend(title=None),
    )
    modes = [m for m in meta["modes"] if m in per_pair["mode"].unique().to_list()]

    # A layered facet needs one shared top-level dataset, so points and limit
    # rules ride in a single frame tagged by `kind` and each mark filters to its
    # own rows. Limit rows carry only (mode, series, metric, limit); their `y` is
    # `limit`, the points' `y` is `v`.
    points_rows = per_pair.with_columns(pl.lit("point").alias("kind"))
    if limits is not None:
        drawn = per_pair.select("mode", "series", "metric").unique()
        limit_rows = limits.join(drawn, on=["mode", "series", "metric"], how="inner").with_columns(
            pl.lit("limit").alias("kind")
        )
        data = pl.concat([points_rows, limit_rows], how="diagonal_relaxed")
    else:
        data = points_rows

    base = alt.Chart(data)
    points = (
        base.transform_filter(alt.datum.kind == "point")
        .mark_circle(size=18, opacity=0.55)
        .encode(
            x=_pair_x(),
            y=alt.Y("v:Q", title=y_title, scale=scale),
            color=color,
            tooltip=[
                "mode:N",
                "series:N",
                "metric:N",
                *[f"{g}:N" for g in group_by],
                alt.Tooltip("v:Q", format=".3s"),
            ],
        )
    )
    layers = [points]
    if limits is not None:
        layers.append(
            base.transform_filter(alt.datum.kind == "limit")
            .mark_rule(strokeDash=[4, 4], opacity=0.8)
            .encode(y=alt.Y("limit:Q", scale=scale), color=color)
        )

    return (
        alt.layer(*layers)
        .properties(width=260, height=200)
        .facet(
            row=alt.Row("mode:N", title=None, sort=modes),
            column=alt.Column("metric:N", title=None, sort=list(metrics)),
        )
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
            y=alt.Y(
                "reach_rate:Q", title="reach rate over attempts", scale=alt.Scale(domain=[0, 1])
            ),
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
    summ = pl.concat(
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
