"""Altair charts for the guided-search frames from ``helpers``."""

from collections.abc import Sequence

import altair as alt
import polars as pl

# Modes take colors in this fixed order.
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
    subtitle = [f"{meta['n_goals']} goals × {meta['n_trials']} attempts"]
    if meta.get("defaults"):
        subtitle.append(meta["defaults"])
    return alt.TitleParams(
        title,
        subtitle=subtitle,
        subtitleColor="#7a7a77",
    )


SERIES_COLORS = {
    "leg": PALETTE[0],
    "baseline": PALETTE[5],
    "guide": PALETTE[1],
}  # blue/orange/green


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
    """Plot leg, guide, and baseline costs in native units.

    Input is long-form with ``mode``, ``metric``, ``series``, pair keys, and
    ``value``. Pairs share the rank of their leg value. Optional per-series
    limits become matching dashed rules; unbounded metrics have no rule.
    """
    per_pair = (
        df.drop_nulls("value")
        .filter(pl.col("value") > 0 if log_y else pl.lit(True))
        .group_by("mode", "metric", "series", *group_by)
        .agg(getattr(pl.col("value"), group_reduce)().alias("v"))
    )
    # Give all series the rank of their pair's leg value.
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

    # Layered facets need one dataset, so tag point and limit rows by kind.
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


def rel_cost_ecdf(
    df: pl.DataFrame,
    meta: dict,
    *,
    title: str,
    y_title: str,
    metrics: Sequence[str],
    group_by: Sequence[str] = ("seed", "goal"),
    group_reduce: str = "median",
) -> alt.Chart:
    """Plot ECDFs of leg and guide cost relative to the unguided baseline.

    Each curve gives the cumulative share of seed/goal pairs at or below a
    relative-cost ratio. The baseline becomes a dashed ``x = 1`` rule; values
    left of it are cheaper than unguided. The x scale is logarithmic.
    """
    # Pivot to divide each series by its pair's baseline, then return to long form.
    wide = (
        df.drop_nulls("value")
        .pivot(on="series", index=["mode", "metric", *group_by], values="value")
        .filter(pl.col("baseline") > 0)
    )
    ratios = wide.select(
        "mode",
        "metric",
        *group_by,
        *[(pl.col(s) / pl.col("baseline")).alias(s) for s in ("leg", "guide") if s in wide.columns],
    ).unpivot(
        index=["mode", "metric", *group_by],
        variable_name="series",
        value_name="value",
    )

    per_pair = (
        ratios.drop_nulls("value")
        .filter(pl.col("value") > 0)
        .group_by("mode", "metric", "series", *group_by)
        .agg(getattr(pl.col("value"), group_reduce)().alias("v"))
        .with_columns(
            (
                pl.col("v").rank("max").over("mode", "metric", "series")
                / pl.len().over("mode", "metric", "series")
            ).alias("cdf")
        )
    )

    color = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=["leg", "guide"],
            range=[SERIES_COLORS["leg"], SERIES_COLORS["guide"]],
        ),
        legend=alt.Legend(title=None),
    )
    modes = [m for m in meta["modes"] if m in per_pair["mode"].unique().to_list()]

    # A data-free rule draws in every facet without creating null-keyed facets.
    curves = (
        alt.Chart()
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=alt.X("v:Q", title="cost / baseline (log)", scale=alt.Scale(type="log")),
            y=alt.Y(
                "cdf:Q",
                title=y_title,
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=color,
            order=alt.Order("v:Q"),
            tooltip=[
                "mode:N",
                "series:N",
                "metric:N",
                *[f"{g}:N" for g in group_by],
                alt.Tooltip("v:Q", format=".3g"),
                alt.Tooltip("cdf:Q", title="cumulative share", format=".1%"),
            ],
        )
    )
    ref = (
        alt.Chart()
        .mark_rule(strokeDash=[4, 4], color="#7a7a77", opacity=0.8)
        .encode(x=alt.X("datum:Q", scale=alt.Scale(type="log")))
        .transform_calculate(datum="1")
    )

    return (
        alt.layer(curves, ref)
        .properties(width=260, height=200)
        .facet(
            row=alt.Row("mode:N", title=None, sort=modes),
            column=alt.Column("metric:N", title=None, sort=list(metrics)),
            data=per_pair,
        )
        .properties(title=_title(title, meta))
        .resolve_scale(x="independent")
    )


def baseline_vs_soft_limits(
    df: pl.DataFrame,
    meta: dict,
    *,
    metrics: Sequence[str],
    limits: pl.DataFrame,
) -> alt.Chart:
    """Compare each pair's unguided baseline with guided-leg soft limits.

    Never-reached and reached pairs form separate sorted blocks. When a baseline
    stopped on the plotted metric, its stop-reason value replaces the final
    measurement so it remains comparable with the limit.
    """
    base_cols = [f"base_{metric}" for metric in metrics]
    leg_stop_kind = (
        pl.when(pl.col("reached"))
        .then(pl.lit("reached"))
        .when(pl.col("panic"))
        .then(pl.lit("panic"))
        .when(pl.col("stop_reason") == "Saturated")
        .then(pl.lit("saturated"))
        .when(pl.col("stop_reason").str.starts_with("NodeLimit"))
        .then(pl.lit("node limit"))
        .when(pl.col("stop_reason").str.starts_with("TimeLimit"))
        .then(pl.lit("time limit"))
        .when(pl.col("stop_reason").str.starts_with("IterationLimit"))
        .then(pl.lit("iteration limit"))
        .when(pl.col("stop_reason").str.contains("memory limit exceeded", literal=True))
        .then(pl.lit("memory limit"))
        .otherwise(pl.lit("other"))
        .alias("leg_stop_kind")
    )
    per_pair = (
        df.with_columns(leg_stop_kind)
        .group_by("mode", "seed", "goal", "pair")
        .agg(
            pl.col("reached").sum().alias("n_reached"),
            pl.len().alias("n_attempts"),
            *[
                (pl.col("leg_stop_kind") == kind).sum().alias(f"n_{kind.replace(' ', '_')}")
                for kind in (
                    "node limit",
                    "saturated",
                    "time limit",
                    "iteration limit",
                    "memory limit",
                    "panic",
                    "other",
                )
            ],
            *[pl.col(col).first().alias(col) for col in base_cols],
            pl.col("base_stop_reason").first(),
        )
        .unpivot(
            index=[
                "mode",
                "seed",
                "goal",
                "pair",
                "n_reached",
                "n_attempts",
                "n_node_limit",
                "n_saturated",
                "n_time_limit",
                "n_iteration_limit",
                "n_memory_limit",
                "n_panic",
                "n_other",
                "base_stop_reason",
            ],
            on=base_cols,
            variable_name="metric",
            value_name="baseline_final",
        )
        .with_columns(
            pl.col("metric").str.strip_prefix("base_"),
            pl.when(pl.col("n_reached") == 0)
            .then(pl.lit("never reached"))
            .otherwise(pl.lit("reached at least once"))
            .alias("reachability"),
            pl.when(pl.col("n_reached") > 0)
            .then(pl.lit("reached"))
            .when(pl.col("n_node_limit") == pl.col("n_attempts"))
            .then(pl.lit("unreachable: node limit"))
            .when(pl.col("n_saturated") == pl.col("n_attempts"))
            .then(pl.lit("unreachable: saturated"))
            .when(pl.col("n_time_limit") == pl.col("n_attempts"))
            .then(pl.lit("unreachable: time limit"))
            .when(pl.col("n_iteration_limit") == pl.col("n_attempts"))
            .then(pl.lit("unreachable: iteration limit"))
            .when(pl.col("n_memory_limit") == pl.col("n_attempts"))
            .then(pl.lit("unreachable: memory limit"))
            .otherwise(pl.lit("unreachable: mixed/other"))
            .alias("leg_outcome"),
        )
        .with_columns(
            pl.when(
                (pl.col("metric") == "nodes")
                & pl.col("base_stop_reason").str.starts_with("NodeLimit")
            )
            .then(pl.col("base_stop_reason").str.extract(r"^NodeLimit\((\d+)\)$", 1))
            .when(
                (pl.col("metric") == "iters")
                & pl.col("base_stop_reason").str.starts_with("IterationLimit")
            )
            .then(pl.col("base_stop_reason").str.extract(r"^IterationLimit\((\d+)\)$", 1))
            .when(
                (pl.col("metric") == "total_time")
                & pl.col("base_stop_reason").str.starts_with("TimeLimit")
            )
            .then(pl.col("base_stop_reason").str.extract(r"^TimeLimit\(([^)]+)\)$", 1))
            .when(
                (pl.col("metric") == "memory")
                & pl.col("base_stop_reason").str.contains("memory limit exceeded", literal=True)
            )
            .then(
                pl.col("base_stop_reason").str.extract(
                    r"memory limit exceeded \((\d+) > \d+ bytes\)", 1
                )
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("baseline_stop_value")
        )
        .with_columns(
            pl.coalesce("baseline_stop_value", "baseline_final").alias("baseline"),
            pl.when(pl.col("baseline_stop_value").is_not_null())
            .then(pl.lit("stop-reason measurement"))
            .otherwise(pl.lit("final measurement"))
            .alias("measurement"),
        )
    )
    soft_limits = limits.filter(pl.col("series") == "leg").select("mode", "metric", "limit")
    plot = (
        per_pair.join(soft_limits, on=["mode", "metric"], how="inner")
        .drop_nulls("baseline")
        .filter((pl.col("baseline") > 0) & (pl.col("limit") > 0))
        .with_columns(
            (pl.col("baseline").rank("ordinal").over("mode", "metric", "reachability") - 1).alias(
                "group_rank"
            ),
            (pl.col("reachability") == "never reached")
            .sum()
            .over("mode", "metric")
            .alias("n_unreachable"),
            (pl.col("reachability") == "reached at least once")
            .sum()
            .over("mode", "metric")
            .alias("n_reachable"),
        )
        .with_columns(
            pl.when(pl.col("reachability") == "never reached")
            .then(pl.col("group_rank"))
            .otherwise(pl.col("n_unreachable") + pl.col("group_rank"))
            .alias("rank"),
            (pl.col("n_unreachable") - 0.5).alias("split_rank"),
        )
    )
    modes = [m for m in meta["modes"] if m in plot["mode"].unique().to_list()]
    outcome_order = [
        "unreachable: node limit",
        "unreachable: saturated",
        "unreachable: time limit",
        "unreachable: iteration limit",
        "unreachable: memory limit",
        "unreachable: mixed/other",
        "reached",
    ]
    outcome_colors = ["#d62728", "#eb6834", "#eda100", "#4a3aa7", "#e87ba4", "#35383d", "#a8adb4"]
    drawn_outcomes = set(plot["leg_outcome"].unique().to_list())
    outcome_order = [outcome for outcome in outcome_order if outcome in drawn_outcomes]
    outcome_colors = [
        color
        for outcome, color in zip(
            [
                "unreachable: node limit",
                "unreachable: saturated",
                "unreachable: time limit",
                "unreachable: iteration limit",
                "unreachable: memory limit",
                "unreachable: mixed/other",
                "reached",
            ],
            outcome_colors,
            strict=True,
        )
        if outcome in drawn_outcomes
    ]
    scale = alt.Scale(type="log")

    points = (
        alt.Chart()
        .mark_circle(opacity=0.72)
        .encode(
            x=alt.X(
                "rank:Q",
                title="pairs: never reached (left) | reached (right); sorted within each",
                axis=alt.Axis(labels=False, ticks=False),
            ),
            y=alt.Y("baseline:Q", title="baseline stop/final measurement (log)", scale=scale),
            color=alt.Color(
                "leg_outcome:N",
                sort=outcome_order,
                scale=alt.Scale(domain=outcome_order, range=outcome_colors),
                legend=alt.Legend(title="guided-leg outcome"),
            ),
            size=alt.Size(
                "reachability:N",
                scale=alt.Scale(domain=["never reached", "reached at least once"], range=[52, 18]),
                legend=None,
            ),
            tooltip=[
                "mode:N",
                "metric:N",
                "reachability:N",
                "leg_outcome:N",
                "seed:N",
                "goal:N",
                "base_stop_reason:N",
                "measurement:N",
                alt.Tooltip("baseline:Q", title="plotted baseline", format=".3s"),
                alt.Tooltip("baseline_final:Q", title="final baseline", format=".3s"),
                alt.Tooltip("limit:Q", title="soft limit", format=".3s"),
                "n_reached:Q",
                "n_attempts:Q",
                "n_node_limit:Q",
                "n_saturated:Q",
                "n_time_limit:Q",
                "n_iteration_limit:Q",
                "n_memory_limit:Q",
                "n_panic:Q",
                "n_other:Q",
            ],
        )
    )
    limit_rule = (
        alt.Chart()
        .transform_aggregate(soft_limit="max(limit)", groupby=["mode", "metric"])
        .mark_rule(strokeDash=[5, 4], color="#35383d", opacity=0.85)
        .encode(y=alt.Y("soft_limit:Q", scale=scale))
    )
    group_divider = (
        alt.Chart()
        .transform_aggregate(
            split_rank="max(split_rank)",
            n_unreachable="max(n_unreachable)",
            n_reachable="max(n_reachable)",
            groupby=["mode", "metric"],
        )
        .transform_filter((alt.datum.n_unreachable > 0) & (alt.datum.n_reachable > 0))
        .mark_rule(color="#7a7a77", opacity=0.7)
        .encode(x="split_rank:Q")
    )

    return (
        alt.layer(points, limit_rule, group_divider)
        .properties(width=260, height=200)
        .facet(
            row=alt.Row("mode:N", title=None, sort=modes),
            column=alt.Column("metric:N", title=None, sort=list(metrics)),
            data=plot,
        )
        .properties(title=_title("Unguided baseline vs guided soft limits", meta))
        .resolve_scale(y="independent", x="independent")
    )


def reachability(df: pl.DataFrame, gr: pl.DataFrame, meta: dict) -> alt.VConcatChart:
    """Plot per-pair reach rates and per-mode summaries."""
    modes = meta["modes"]
    color = _color(modes)

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
    """Plot goal × mode reachability, with the hardest goals at top."""
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
