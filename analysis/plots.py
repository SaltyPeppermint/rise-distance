"""Altair plots for guided/unguided success and peak-memory comparisons."""

from collections.abc import Sequence

import altair as alt
import polars as pl

PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#008300",
    "#e87ba4",
    "#4a3aa7",
    "#eda100",
    "#1baf7a",
    "#e34948",
    "#7f6d5f",
    "#5e9ed6",
]
OUTCOME_ORDER = ["both", "guided only", "unguided only", "neither"]
OUTCOME_COLORS = ["#4c9f70", "#2a78d6", "#eb6834", "#b8b8b4"]

THEME = {
    "config": {
        "view": {"continuousWidth": 360, "continuousHeight": 260, "strokeOpacity": 0},
        "axis": {
            "grid": True,
            "gridColor": "#e8e8e6",
            "domainColor": "#c9c9c6",
            "tickColor": "#c9c9c6",
        },
        "legend": {"orient": "top", "titleFontSize": 11, "labelFontSize": 11},
        "title": {"fontSize": 13, "anchor": "start"},
        "range": {"category": PALETTE},
    }
}


def _title(text: str, meta: dict) -> alt.TitleParams:
    return alt.TitleParams(text, subtitle=meta.get("subtitle", []), subtitleColor="#777")


def _mode_color(modes: Sequence[str]) -> alt.Color:
    return alt.Color(
        "mode:N",
        sort=list(modes),
        scale=alt.Scale(domain=list(modes), range=PALETTE[: len(modes)]),
        legend=alt.Legend(title=None),
    )


def _guided_peak_scope(paired: pl.DataFrame) -> str:
    if "guided_peak_scope" not in paired.columns:
        return "guided workflow"
    scopes = paired["guided_peak_scope"].drop_nulls().unique().to_list()
    return str(scopes[0]) if len(scopes) == 1 else "guided"


def _memory_metric(frame: pl.DataFrame) -> str:
    """Metric label carried by `paired_*_successes`, for axis titles."""
    if "memory_metric" not in frame.columns:
        return "peak memory"
    labels = frame["memory_metric"].drop_nulls().unique().to_list()
    return str(labels[0]) if len(labels) == 1 else "peak memory"


def success_rates(rates: pl.DataFrame, meta: dict) -> alt.Chart:
    """Success rates with Wilson intervals."""
    points = (
        alt.Chart(rates)
        .mark_point(filled=True, size=75)
        .encode(
            x=alt.X("success_rate:Q", title="success rate", axis=alt.Axis(format="%")),
            y=alt.Y("mode:N", title=None, sort=list(meta["modes"])),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(domain=["guided", "unguided"], range=[PALETTE[0], PALETTE[1]]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                "mode:N",
                "method:N",
                "successes:Q",
                "n:Q",
                alt.Tooltip("success_rate:Q", format=".1%"),
            ],
        )
    )
    intervals = points.mark_rule().encode(x="ci_low:Q", x2="ci_high:Q")
    return (intervals + points).properties(title=_title("Success rate", meta), height=alt.Step(32))


def success_outcomes(outcomes: pl.DataFrame, meta: dict) -> alt.Chart:
    """Distribution of paired success outcomes."""
    return (
        alt.Chart(outcomes)
        .mark_bar()
        .encode(
            x=alt.X("share:Q", title="share of planned pairs", axis=alt.Axis(format="%")),
            y=alt.Y("mode:N", title=None, sort=list(meta["modes"])),
            color=alt.Color(
                "outcome:N",
                sort=OUTCOME_ORDER,
                scale=alt.Scale(domain=OUTCOME_ORDER, range=OUTCOME_COLORS),
                legend=alt.Legend(title=None),
            ),
            order=alt.Order("outcome:N", sort="ascending"),
            tooltip=[
                "mode:N",
                "outcome:N",
                "count:Q",
                alt.Tooltip("share:Q", format=".1%"),
            ],
        )
        .properties(title=_title("Paired outcomes", meta), height=alt.Step(30))
    )


def paired_peak_scatter(paired: pl.DataFrame, meta: dict) -> alt.Chart:
    """One explicitly scoped guided peak versus unguided verification."""
    guided_scope = _guided_peak_scope(paired)
    metric = _memory_metric(paired)
    title = f"{guided_scope.title()} vs unguided verification"
    points = (
        alt.Chart(paired)
        .mark_circle(size=45, opacity=0.58)
        .encode(
            x=alt.X(
                "unguided_peak_mib:Q",
                title=f"unguided {metric} (MiB, log)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "guided_peak_mib:Q",
                title=f"{guided_scope} {metric} (MiB, log)",
                scale=alt.Scale(type="log"),
            ),
            color=_mode_color(meta["modes"]),
            tooltip=[
                "mode:N",
                "start_term:N",
                "goal_term:N",
                alt.Tooltip("guided_peak_mib:Q", format=".1f"),
                alt.Tooltip("unguided_peak_mib:Q", format=".1f"),
                alt.Tooltip("peak_ratio:Q", format=".3f"),
            ],
        )
    )
    if paired.is_empty():
        return points.properties(
            title=_title(title, meta),
            width=420,
            height=380,
        )
    bounds = paired.select(
        pl.min_horizontal("guided_peak_mib", "unguided_peak_mib").min().alias("lo"),
        pl.max_horizontal("guided_peak_mib", "unguided_peak_mib").max().alias("hi"),
    ).row(0, named=True)
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [bounds["lo"], bounds["hi"]]}))
        .mark_line(strokeDash=[5, 4], color="#777")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(type="log")), y=alt.Y("x:Q", scale=alt.Scale(type="log"))
        )
    )
    return (diagonal + points).properties(
        title=_title(title, meta),
        width=420,
        height=380,
    )


def peak_ratio_ecdf(paired: pl.DataFrame, meta: dict) -> alt.Chart:
    """ECDF of one explicitly scoped guided peak versus unguided verification."""
    guided_scope = _guided_peak_scope(paired)
    metric = _memory_metric(paired)
    data = paired.with_columns(
        (pl.col("peak_ratio").rank("max").over("mode") / pl.len().over("mode")).alias("cdf")
    ).sort("mode", "peak_ratio")
    curves = (
        alt.Chart(data)
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=alt.X(
                "peak_ratio:Q",
                title=f"{guided_scope} / unguided verification {metric} (log)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y("cdf:Q", title="cumulative share", axis=alt.Axis(format="%")),
            color=_mode_color(meta["modes"]),
            order="peak_ratio:Q",
            tooltip=[
                "mode:N",
                alt.Tooltip("peak_ratio:Q", format=".3f"),
                alt.Tooltip("cdf:Q", format=".1%"),
            ],
        )
    )
    parity = (
        alt.Chart(pl.DataFrame({"ratio": [1.0]}))
        .mark_rule(strokeDash=[5, 4], color="#777")
        .encode(x=alt.X("ratio:Q", scale=alt.Scale(type="log")))
    )
    return (curves + parity).properties(
        title=_title(f"{guided_scope.title()} {metric} ratio", meta), width=460
    )


def verification_peak_ecdf(paired: pl.DataFrame, meta: dict) -> alt.Chart:
    """Absolute guided/unguided verification peaks, excluding candidate construction."""
    metric = _memory_metric(paired)
    data = pl.concat(
        [
            paired.select(
                "mode",
                (pl.col("guided_peak_mib")).alias("peak_mib"),
                pl.lit("guided verification").alias("method"),
            ),
            paired.select(
                "mode",
                (pl.col("unguided_peak_mib")).alias("peak_mib"),
                pl.lit("unguided verification").alias("method"),
            ),
        ]
    ).with_columns(
        (
            pl.col("peak_mib").rank("max").over("mode", "method") / pl.len().over("mode", "method")
        ).alias("cdf")
    )
    return (
        alt.Chart(data)
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=alt.X(
                "peak_mib:Q",
                title=f"verification {metric} (MiB, log)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y("cdf:Q", title="cumulative share", axis=alt.Axis(format="%")),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(
                    domain=["guided verification", "unguided verification"],
                    range=[PALETTE[0], PALETTE[1]],
                ),
                legend=alt.Legend(title=None),
            ),
            column=alt.Column("mode:N", title=None, sort=list(meta["modes"])),
            order="peak_mib:Q",
            tooltip=[
                "mode:N",
                "method:N",
                alt.Tooltip("peak_mib:Q", format=".1f"),
                alt.Tooltip("cdf:Q", format=".1%"),
            ],
        )
        .properties(
            title=_title(f"Verification-only {metric} (candidate construction excluded)", meta),
            width=300,
            height=240,
        )
    )


def attempts_to_success(frame: pl.DataFrame, meta: dict) -> alt.Chart:
    """Distribution of the successful guided attempt."""
    data = frame.filter(pl.col("guided_success")).drop_nulls("success_attempt")
    return (
        alt.Chart(data)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("success_attempt:O", title="attempt of first success"),
            y=alt.Y("count():Q", title="successful pairs"),
            color=_mode_color(meta["modes"]),
            column=alt.Column("mode:N", title=None, sort=list(meta["modes"])),
            tooltip=["mode:N", "success_attempt:O", "count():Q"],
        )
        .properties(title=_title("Attempts to success", meta), width=180, height=180)
    )
