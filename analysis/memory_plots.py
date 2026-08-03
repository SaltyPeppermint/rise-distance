"""Altair charts for the next-iteration memory model."""

import altair as alt
import polars as pl
from altair.typing import ChartType

from plots import PALETTE

# Whisker length in IQRs, matching `mark_boxplot`'s default extent.
EXTENT = 1.5

MODEL_COLORS = {
    "naive (carry forward)": "#9aa0a6",
    "ridge": PALETTE[0],
    "gradient boosting": PALETTE[1],
}


def _model_color() -> alt.Color:
    return alt.Color(
        "model:N",
        scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
        legend=alt.Legend(title=None),
    )


def metric_bars(metrics: pl.DataFrame, metric: str = "R2") -> ChartType:
    """Compare models per target on one metric."""
    return (
        alt.Chart(metrics)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric}:Q", title=metric),
            y=alt.Y("model:N", title=None, sort=list(MODEL_COLORS)),
            color=_model_color(),
            tooltip=["target:N", "model:N", alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
        .properties(width=260, height=alt.Step(22))
        .facet(column=alt.Column("target:N", title=None))
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                f"Model comparison ({metric})",
                subtitle=[
                    "grouped 5-fold CV by seed term · higher is better",
                    "the naive baseline is the bar to beat, not a formality",
                ],
                subtitleColor="#7a7a77",
            )
        )
    )


def predicted_vs_actual(predictions: pl.DataFrame, target: str, sample: int = 4000) -> ChartType:
    """Out-of-fold predictions against truth, with an ideal y = x rule."""
    plot = predictions.filter(pl.col("target") == target)
    if len(plot) > sample:
        plot = plot.sample(sample, seed=0)

    points = (
        alt.Chart(plot)
        .mark_circle(size=14, opacity=0.3)
        .encode(
            x=alt.X("actual:Q", title="actual", scale=alt.Scale(zero=False)),
            y=alt.Y("predicted:Q", title="predicted", scale=alt.Scale(zero=False)),
            color=_model_color(),
            tooltip=[
                "model:N",
                alt.Tooltip("actual:Q", format=".3f"),
                alt.Tooltip("predicted:Q", format=".3f"),
                "egraph_nodes:Q",
                "iter_index:Q",
            ],
        )
    )
    ideal = (
        alt.Chart(plot)
        .mark_line(strokeDash=[4, 4], color="#35383d", opacity=0.8)
        .encode(x=alt.X("actual:Q"), y=alt.Y("actual:Q"))
    )
    return (
        alt.layer(points, ideal)
        .properties(width=220, height=220)
        .facet(column=alt.Column("model:N", title=None, sort=list(MODEL_COLORS)))
        .properties(
            title=alt.TitleParams(
                f"Predicted vs actual — {target}",
                subtitle=["out-of-fold predictions · dashed line is a perfect model"],
                subtitleColor="#7a7a77",
            )
        )
    )


def growth_histogram(
    transitions: pl.DataFrame, col: str = "y_log_growth", bins: int = 80
) -> ChartType:
    """Distribution of the growth target, binned in Polars.

    `alt.Bin(maxbins=80)` bins in the browser, so the chart spec would carry
    every transition to draw 80 bars. Counting here ships 80 rows instead.
    """
    bounds = transitions.select(lo=pl.col(col).min(), hi=pl.col(col).max()).cast(pl.Float64)
    lo, hi = bounds.item(0, "lo"), bounds.item(0, "hi")
    width = (hi - lo) / bins
    counts = (
        transitions.select(
            # Right-open bins, with the top edge folded into the last bin so
            # the maximum does not land in a bin of its own.
            bin_index=((pl.col(col) - lo) / width).floor().clip(0, bins - 1).cast(pl.Int32)
        )
        .group_by("bin_index")
        .len(name="transitions")
        .with_columns(
            bin_start=lo + pl.col("bin_index") * width,
            bin_end=lo + (pl.col("bin_index") + 1) * width,
        )
        .sort("bin_index")
    )
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("bin_start:Q", bin="binned", title="log memory growth ratio"),
            x2="bin_end:Q",
            y=alt.Y("transitions:Q", title="transitions"),
        )
        .properties(
            width=520,
            height=200,
            title=alt.TitleParams(
                "Distribution of memory growth",
                subtitle=["0 means memory held steady · 0.69 means it doubled"],
                subtitleColor="#7a7a77",
            ),
        )
    )


def residual_summary(predictions: pl.DataFrame, target: str) -> pl.DataFrame:
    """Five-number summary per model, with whiskers clamped to real data.

    Aggregating here rather than in Vega keeps the notebook small: a boxplot
    needs five numbers per model, not one row per prediction.
    """
    quartiles = (
        predictions.filter(pl.col("target") == target)
        .group_by("model")
        .agg(
            # Linear interpolation to match Vega's quartiles; Polars defaults
            # to "nearest", which shifts the box edges by a hair.
            pl.col("residual").quantile(0.25, interpolation="linear").alias("q1"),
            pl.col("residual").median().alias("median"),
            pl.col("residual").quantile(0.75, interpolation="linear").alias("q3"),
            pl.col("residual").alias("residual"),
        )
        .with_columns((pl.col("q3") - pl.col("q1")).alias("iqr"))
        .with_columns(
            lo=pl.col("q1") - EXTENT * pl.col("iqr"),
            hi=pl.col("q3") + EXTENT * pl.col("iqr"),
        )
    )
    # Whiskers reach the furthest observation inside 1.5·IQR, matching what
    # `mark_boxplot(extent=1.5)` would have drawn from the raw rows. Clamping
    # the min/max to the fences would draw the fence itself, which invents a
    # whisker past where the data actually stops.
    #
    # `list.eval` cannot see sibling columns, so this explodes back to one row
    # per residual and filters against the fences directly.
    return (
        quartiles.explode("residual")
        .filter(pl.col("residual").is_between("lo", "hi"))
        .group_by("model", "q1", "median", "q3")
        .agg(
            lo_whisker=pl.col("residual").min(),
            hi_whisker=pl.col("residual").max(),
        )
    )


def residual_distribution(predictions: pl.DataFrame, target: str) -> ChartType:
    """Residual spread per model; a tight band centred on zero is the goal.

    Drawn from `residual_summary`, so the chart spec carries one row per model
    instead of one per prediction.
    """
    plot = residual_summary(predictions, target)
    x = alt.X("model:N", title=None, sort=list(MODEL_COLORS), axis=alt.Axis(labelAngle=-25))
    y_title = "predicted − actual (log)"

    whisker = (
        alt.Chart(plot)
        .mark_rule()
        .encode(
            x=x, y=alt.Y("lo_whisker:Q", title=y_title), y2="hi_whisker:Q", color=_model_color()
        )
    )
    box = (
        alt.Chart(plot)
        .mark_bar(size=18)
        .encode(x=x, y=alt.Y("q1:Q", title=y_title), y2="q3:Q", color=_model_color())
    )
    midline = (
        alt.Chart(plot)
        .mark_tick(color="white", size=18, thickness=2)
        .encode(x=x, y=alt.Y("median:Q", title=y_title))
    )
    return (
        alt.layer(whisker, box, midline)
        .properties(width=280, height=220)
        .properties(
            title=alt.TitleParams(
                f"Residual distribution — {target}",
                subtitle=["whiskers at 1.5·IQR · centred and narrow is better"],
                subtitleColor="#7a7a77",
            )
        )
    )


def residual_vs_size(predictions: pl.DataFrame, target: str, sample: int = 4000) -> ChartType:
    """Residuals against egraph size, to expose scale-dependent bias."""
    plot = predictions.filter((pl.col("target") == target) & (pl.col("egraph_nodes") > 0))
    if len(plot) > sample:
        plot = plot.sample(sample, seed=0)

    points = (
        alt.Chart(plot)
        .mark_circle(size=12, opacity=0.28)
        .encode(
            x=alt.X(
                "egraph_nodes:Q",
                title="egraph nodes this iteration (log)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y("residual:Q", title="predicted − actual (log)"),
            color=_model_color(),
            tooltip=["model:N", "egraph_nodes:Q", alt.Tooltip("residual:Q", format=".3f")],
        )
    )
    zero = (
        alt.Chart(plot)
        .mark_rule(strokeDash=[4, 4], color="#35383d", opacity=0.8)
        .encode(y=alt.datum(0))
    )
    return (
        alt.layer(points, zero)
        .properties(width=220, height=200)
        .facet(column=alt.Column("model:N", title=None, sort=list(MODEL_COLORS)))
        .properties(
            title=alt.TitleParams(
                f"Residual vs egraph size — {target}",
                subtitle=["drift away from the dashed line means scale-dependent bias"],
                subtitleColor="#7a7a77",
            )
        )
    )


def window_sweep_chart(sweep: pl.DataFrame, metric: str = "R2") -> ChartType:
    """Score against history depth, one line per model, faceted by target."""
    line = (
        alt.Chart(sweep)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("window:Q", title="iterations of history (1 = current model)"),
            y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
            color=_model_color(),
            tooltip=[
                "model:N",
                "window:Q",
                alt.Tooltip(f"{metric}:Q", format=".4f"),
                "n_features:Q",
            ],
        )
    )
    return (
        line.properties(width=260, height=220)
        .facet(column=alt.Column("target:N", title=None))
        .resolve_scale(y="independent")
        .properties(
            title=alt.TitleParams(
                f"Does more history help? ({metric})",
                subtitle=[
                    "same 10,826 rows at every window · grouped 5-fold CV by seed term",
                    "the naive line is flat by construction — it ignores history",
                ],
                subtitleColor="#7a7a77",
            )
        )
    )


def importance_bars(importance: pl.DataFrame) -> ChartType:
    """Permutation importance on held-out terms, scalars vs rewrite rules."""
    order = importance["feature"].to_list()
    bars = (
        alt.Chart(importance)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="drop in R² when shuffled"),
            y=alt.Y("feature:N", title=None, sort=order),
            color=alt.Color(
                "block:N",
                scale=alt.Scale(domain=["scalar", "rule"], range=[PALETTE[0], PALETTE[3]]),
                legend=alt.Legend(title=None),
            ),
            tooltip=["feature:N", "block:N", alt.Tooltip("importance:Q", format=".4f")],
        )
    )
    error = (
        alt.Chart(importance)
        .mark_rule(opacity=0.6)
        .encode(
            y=alt.Y("feature:N", sort=order),
            x=alt.X("lo:Q", title=""),
            x2="hi:Q",
        )
        .transform_calculate(lo="datum.importance - datum.std", hi="datum.importance + datum.std")
    )
    return (
        alt.layer(bars, error)
        .properties(width=340, height=alt.Step(16))
        .properties(
            title=alt.TitleParams(
                "Permutation importance (growth target)",
                subtitle=["measured on held-out terms · rule = one rewrite's application count"],
                subtitleColor="#7a7a77",
            )
        )
    )
