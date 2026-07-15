"""Altair chart specs for the guide-selection analysis.

Everything consumes the tidy long frame from `helpers.load_runs` (one row per
leg, tagged with `mode`, carrying `r_<metric>` = metric / seed-baseline). The
recurring shape is *median + 25-75 IQR band vs k, one facet per metric, one
color per mode* — `band_vs_k` builds that once and the notebook calls it with a
different value column each time. `k` is drawn on a log axis; ratio plots carry a
dashed break-even line at 1.0.

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


def _log_k(k_values: Sequence[int]) -> alt.X:
    return alt.X(
        "k:Q",
        scale=alt.Scale(type="log", base=10, nice=False),
        axis=alt.Axis(values=list(k_values), title="k (number of guides)"),
    )


def _title(title: str, meta: dict) -> alt.TitleParams:
    return alt.TitleParams(
        title,
        subtitle=f"{meta['n_goals']} goals × {meta['n_trials']} trials per k",
        subtitleColor="#7a7a77",
    )


def band_vs_k(
    df: pl.DataFrame,
    value: str,
    meta: dict,
    *,
    title: str,
    y_title: str,
    metrics: Sequence[str],
    breakeven: bool = True,
    group_by: Sequence[str] = (),
    group_reduce: str = "min",
    overlay_by: str | None = None,
) -> alt.Chart:
    """Median line + 25-75 IQR band of `value` vs k, one column per metric.

    `df` is long with columns `mode`, `k`, `metric`, and `value`. Callers that
    start from the wide tidy frame melt the relevant `r_<metric>`/cost columns
    into (`metric`, `value`) first — see the notebook.

    `group_by` pre-aggregates within each ``(mode, metric, k, *group_by)`` using
    `group_reduce` before the cross-group median+IQR — e.g. ``group_by=["goal"]``
    with ``group_reduce="min"`` gives "best guide per goal, summarised over
    goals". Without it, the median+IQR runs over the raw rows.

    `overlay_by` (e.g. ``"seed"`` or ``"goal"``) draws one faint median curve per
    group of that column behind the band, to show the spread. It is independent
    of `group_by`: the band is unchanged, the overlay just adds context.

    Ratios are heavy-tailed, so median+IQR beats mean±sd. Single-point modes
    (only k=1) fall out as a lone point, which reads fine on the log axis.
    """
    modes = [m for m in meta["modes"] if m in df["mode"].unique().to_list()]
    clean = df.drop_nulls(value)
    agg = (
        clean.group_by("mode", "metric", "k", *group_by).agg(
            getattr(pl.col(value), group_reduce)().alias("m")
        )
        if group_by
        else clean.with_columns(pl.col(value).alias("m"))
    )
    stats = (
        agg.group_by("mode", "metric", "k")
        .agg(
            pl.col("m").median().alias("med"),
            pl.col("m").quantile(0.25).alias("q25"),
            pl.col("m").quantile(0.75).alias("q75"),
        )
        .sort("k")
    )

    # A faceted layered chart must share ONE dataset across layers, so we union
    # the aggregate rows (_kind="stat") with the faint per-group overlay rows
    # (_kind="raw") into `data` and let each mark filter to its kind. Overlay
    # rows are the per-`overlay_by` median, keyed by `_gid` (mode + group id) so
    # each faint line is one group; this is independent of the band's grouping.
    stat_rows = stats.with_columns(
        pl.lit("stat").alias("_kind"), pl.lit(None, pl.Utf8).alias("_gid")
    )
    show_overlay = overlay_by is not None
    if show_overlay:
        # Match the band's per-group statistic when the overlay groups by the
        # same column (e.g. best_ratio = per-goal min); otherwise show medians.
        ov_reduce = group_reduce if overlay_by in group_by else "median"
        raw_rows = (
            clean.group_by("mode", "metric", "k", overlay_by)
            .agg(getattr(pl.col(value), ov_reduce)().alias("med"))
            .with_columns(
                pl.lit(None, pl.Float64).alias("q25"),
                pl.lit(None, pl.Float64).alias("q75"),
                pl.lit("raw").alias("_kind"),
                pl.concat_str(["mode", overlay_by], separator="│").alias("_gid"),
            )
            .select(stat_rows.columns)
            .sort("k")
        )
        data = pl.concat([raw_rows, stat_rows], how="vertical")
    else:
        data = stat_rows

    x = _log_k(meta["k_values"])
    layers = []
    if show_overlay:
        layers.append(
            alt.Chart()
            .transform_filter(alt.datum._kind == "raw")
            .mark_line(opacity=0.12, strokeWidth=0.6)
            .encode(x=x, y=alt.Y("med:Q", title=y_title), color=_color(modes), detail="_gid:N")
        )
    stat = alt.Chart().transform_filter(alt.datum._kind == "stat")
    band = stat.mark_area(opacity=0.18).encode(
        x=x, y=alt.Y("q25:Q", title=y_title), y2="q75:Q", color=_color(modes)
    )
    line = stat.mark_line(point=alt.OverlayMarkDef(size=45)).encode(
        x=x,
        y=alt.Y("med:Q", title=y_title),
        color=_color(modes),
        tooltip=["mode:N", "metric:N", "k:Q", alt.Tooltip("med:Q", format=".3f")],
    )
    layers += [band, line]
    if breakeven:
        layers.append(
            alt.Chart()
            .mark_rule(strokeDash=[4, 4], color="#555", opacity=0.7)
            .encode(y=alt.datum(1.0))
        )

    return (
        alt.layer(*layers, data=data)
        .facet(column=alt.Column("metric:N", title=None, sort=list(metrics)))
        .properties(title=_title(title, meta))
        .resolve_scale(y="independent")
    )


def reachability(df: pl.DataFrame, gr: pl.DataFrame, meta: dict) -> alt.VConcatChart:
    """Three panels: leg reach rate vs k, goal coverage vs k, CDF of per-goal reach.

    `gr` is `helpers.goal_reach(df)`. Panels share the mode color legend.
    """
    modes = meta["modes"]
    color = _color(modes)
    x = _log_k(meta["k_values"])

    leg = df.group_by("mode", "k").agg((pl.col("reached").mean() * 100).alias("rate")).sort("k")
    rate = (
        alt.Chart(leg)
        .mark_line(point=True)
        .encode(
            x=x,
            y=alt.Y("rate:Q", title="reach rate (%)", scale=alt.Scale(domainMin=0)),
            color=color,
            tooltip=["mode:N", "k:Q", alt.Tooltip("rate:Q", format=".1f")],
        )
        .properties(title="Leg reachability rate vs k")
    )

    cov = (
        gr.group_by("mode", "k")
        .agg((pl.col("reach_rate") > 0).mean().mul(100).alias("cov"))
        .sort("k")
    )
    coverage = (
        alt.Chart(cov)
        .mark_line(point=True)
        .encode(
            x=x,
            y=alt.Y("cov:Q", title="goals with ≥1 success (%)", scale=alt.Scale(domain=[0, 100])),
            color=color,
            tooltip=["mode:N", "k:Q", alt.Tooltip("cov:Q", format=".1f")],
        )
        .properties(title="Goal coverage vs k")
    )

    # CDF of per-goal reach_rate, per (mode, k): rank goals, normalize to [0,1].
    cdf_df = gr.with_columns(
        (pl.col("reach_rate").rank("ordinal").over("mode", "k") - 1).alias("rank"),
        pl.len().over("mode", "k").alias("n"),
    ).with_columns((pl.col("rank") / (pl.col("n") - 1).clip(lower_bound=1)).alias("frac"))
    cdf = (
        alt.Chart(cdf_df)
        .mark_line()
        .encode(
            x=alt.X("reach_rate:Q", title="per-goal reachability"),
            y=alt.Y("frac:Q", title="fraction of goals (CDF)"),
            color=color,
            detail="k:N",
            strokeDash=alt.StrokeDash("k:N", legend=alt.Legend(title="k")),
        )
        .properties(title="CDF of per-goal reachability by k")
    )

    return alt.vconcat(rate | coverage, cdf).properties(title=_title("Reachability summary", meta))


def cost_boxplots(df: pl.DataFrame, meta: dict, metrics: Sequence[str]) -> alt.Chart:
    """Box plots of reached-leg cost by mode, faceted metric (row) × k (column)."""
    modes = meta["modes"]
    long = (
        df.filter(pl.col("reached"))
        .unpivot(index=["mode", "k"], on=list(metrics), variable_name="metric", value_name="v")
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
        .facet(row=alt.Row("metric:N", sort=list(metrics)), column=alt.Column("k:N", title="k"))
        .resolve_scale(y="independent")
        .properties(title=_title("Cost distribution by mode at each k", meta))
    )


def reach_heatmap(gr: pl.DataFrame, meta: dict) -> alt.Chart:
    """Goal × k reachability heatmap, faceted by mode (hardest goals at top).

    Goals are ranked per mode by mean reach_rate, and that rank is the y position.
    Hundreds of goals in a few hundred pixels means each row is sub-pixel, so an
    *ordinal* y (one axis band per goal) both stripes the image and is slow; we
    instead give each cell an explicit ``gy → gy+1`` quantitative band via `y`/`y2`
    so the rects tile seamlessly, and drop the rect stroke. The y scale is
    reversed so rank 1 (hardest) sits at the top.
    """
    order = (
        gr.group_by("mode", "goal")
        .agg(pl.col("reach_rate").mean().alias("avg"))
        .with_columns((pl.col("avg").rank("ordinal").over("mode")).alias("gy"))
    )
    plot = gr.join(order.select("mode", "goal", "gy"), on=["mode", "goal"]).with_columns(
        (pl.col("gy") - 1).alias("gy0"), pl.col("gy").alias("gy1")
    )
    n_goals = plot.select(pl.col("gy").max()).item()
    return (
        alt.Chart(plot)
        .mark_rect(stroke=None, strokeWidth=0)
        .encode(
            x=alt.X("k:O", title="k (number of guides)"),
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
            tooltip=["mode:N", "goal:N", "k:O", alt.Tooltip("reach_rate:Q", format=".0%")],
        )
        .properties(width=180, height=420)
        .facet(column=alt.Column("mode:N", title=None))
        .properties(title=_title("Goal-level reachability heatmap", meta))
    )
