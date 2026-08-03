"""Altair charts for the next-iteration memory model."""

import altair as alt
import polars as pl
from altair.typing import ChartType

from plots import PALETTE

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


def residual_distribution(predictions: pl.DataFrame, target: str) -> ChartType:
    """Residual spread per model; a tight band centred on zero is the goal."""
    plot = predictions.filter(pl.col("target") == target)
    return (
        alt.Chart(plot)
        .mark_boxplot(extent=1.5, size=18)
        .encode(
            x=alt.X("model:N", title=None, sort=list(MODEL_COLORS), axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("residual:Q", title="predicted − actual (log)"),
            color=_model_color(),
        )
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
