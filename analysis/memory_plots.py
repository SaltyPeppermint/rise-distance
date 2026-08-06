"""Altair charts for the upcoming-iteration peak-memory model."""

import json
from pathlib import Path

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

MEMORY_SERIES_COLORS = {
    "individual run": "#9aa9b8",
    "10th–90th percentile": "#72a5dc",
    "median": "#1f5f99",
    "memory limit": "#d62728",
}


def _model_color() -> alt.Color:
    return alt.Color(
        "model:N",
        scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
        legend=alt.Legend(title=None),
    )


def _parse_size(value: str | float) -> float:
    """Parse a byte count or a human size such as ``100M``."""
    if not isinstance(value, str):
        return float(value)

    text = value.strip().upper()
    suffixes = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    suffix = text[-1]
    if suffix in suffixes:
        return float(text[:-1]) * suffixes[suffix]
    return float(text)


def maximum_egraph_memory(iterations: pl.DataFrame, seed_dir: Path) -> ChartType:
    """Plot each term's sampled iteration peak and configured limit."""
    maxima = (
        iterations.filter(pl.col("iteration_peak_allocated") > 0)
        .group_by("term", "term_size")
        .agg((pl.col("iteration_peak_allocated").max() / 2**20).alias("max_memory_mib"))
    )
    summary = (
        maxima.group_by("term_size")
        .agg(
            pl.col("max_memory_mib").quantile(0.1).alias("p10"),
            pl.col("max_memory_mib").median().alias("median"),
            pl.col("max_memory_mib").quantile(0.9).alias("p90"),
        )
        .sort("term_size")
    )

    run_args = json.loads((seed_dir / "generation_args.json").read_text())
    raw_limit = run_args.get("max_memory")
    series = list(MEMORY_SERIES_COLORS)
    if raw_limit is None:
        series.remove("memory limit")

    series_color = alt.Color(
        "series:N",
        title=None,
        scale=alt.Scale(
            domain=series,
            range=[MEMORY_SERIES_COLORS[name] for name in series],
        ),
        legend=alt.Legend(orient="top"),
    )
    y_scale = alt.Scale(type="log")

    points = (
        alt.Chart(maxima)
        .transform_calculate(series="'individual run'")
        .mark_circle(size=16, opacity=0.18)
        .encode(
            x=alt.X("term_size:Q", title="term size"),
            y=alt.Y(
                "max_memory_mib:Q",
                title="maximum egraph memory (MiB)",
                scale=y_scale,
            ),
            color=series_color,
            tooltip=[
                "term_size:Q",
                alt.Tooltip(
                    "max_memory_mib:Q",
                    title="maximum memory (MiB)",
                    format=".2f",
                ),
            ],
        )
    )
    band = (
        alt.Chart(summary)
        .transform_calculate(series="'10th–90th percentile'")
        .mark_area(opacity=0.2)
        .encode(
            x="term_size:Q",
            y=alt.Y("p10:Q", scale=y_scale),
            y2="p90:Q",
            color=series_color,
        )
    )
    median = (
        alt.Chart(summary)
        .transform_calculate(series="'median'")
        .mark_line(strokeWidth=2)
        .encode(
            x="term_size:Q",
            y=alt.Y("median:Q", scale=y_scale),
            color=series_color,
            tooltip=[
                "term_size:Q",
                alt.Tooltip("p10:Q", title="10th percentile (MiB)", format=".2f"),
                alt.Tooltip("median:Q", title="median (MiB)", format=".2f"),
                alt.Tooltip("p90:Q", title="90th percentile (MiB)", format=".2f"),
            ],
        )
    )
    layers = [points, band, median]

    if raw_limit is not None:
        limit_mib = _parse_size(raw_limit) / 2**20
        limit = (
            alt.Chart(pl.DataFrame({"memory_limit_mib": [limit_mib]}))
            .transform_calculate(series="'memory limit'")
            .mark_rule(strokeWidth=2)
            .encode(
                y=alt.Y("memory_limit_mib:Q", scale=y_scale),
                color=series_color,
                tooltip=[
                    alt.Tooltip(
                        "memory_limit_mib:Q",
                        title="memory limit (MiB)",
                        format=".2f",
                    )
                ],
            )
        )
        layers.append(limit)

    return alt.layer(*layers).properties(
        width=360,
        height=200,
        title=alt.TitleParams(
            "Maximum egraph memory",
            subtitle=[f"{len(maxima):,} data points · one per term"],
            subtitleColor="#7a7a77",
        ),
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
    transitions: pl.DataFrame, col: str = "y_log_peak_growth", bins: int = 80
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
            x=alt.X("bin_start:Q", bin="binned", title="log peak growth ratio"),
            x2="bin_end:Q",
            y=alt.Y("transitions:Q", title="transitions"),
        )
        .properties(
            width=520,
            height=200,
            title=alt.TitleParams(
                "Distribution of upcoming peak growth",
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


BOUNDARY_COLORS = {"raw": "#9aa0a6", "conservative": PALETTE[0]}


def ceiling_decisions_chart(decisions: pl.DataFrame) -> ChartType:
    """Recall and precision of ceiling-crossing stops, by ceiling and boundary.

    Recall is the share of runs that would have broken the ceiling and were
    stopped first; precision is the share of stopped runs that really would
    have crossed. The safety margin trades the second for the first.
    """
    scores = decisions.unpivot(
        on=["recall", "precision"],
        index=["boundary", "ceiling_mib"],
        variable_name="metric",
        value_name="value",
    ).drop_nulls("value")

    return (
        alt.Chart(scores)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("ceiling_mib:Q", title="ceiling (MiB)", scale=alt.Scale(type="log")),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "boundary:N",
                scale=alt.Scale(
                    domain=list(BOUNDARY_COLORS), range=list(BOUNDARY_COLORS.values())
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                "boundary:N",
                alt.Tooltip("ceiling_mib:Q", format=".0f"),
                "metric:N",
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
        .properties(width=260, height=220)
        .facet(column=alt.Column("metric:N", title=None))
        .properties(
            title=alt.TitleParams(
                "Catching ceiling breaks before they happen",
                subtitle=[
                    "recall: crossings stopped in time · precision: stops that were warranted",
                    "the safety margin buys recall at the cost of stopping some healthy runs",
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
                "Permutation importance (peak-growth target)",
                subtitle=["measured on held-out terms · rule = upcoming scheduler state"],
                subtitleColor="#7a7a77",
            )
        )
    )
