"""Altair plots for guided success and guided-vs-brute-force peak-memory comparisons."""

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
# Peak RSS is the only memory metric reported; it names every memory axis.
MEMORY_LABEL = "peak RSS"
METHOD_ORDER = ["guided", "unguided"]
METHOD_COLORS = [PALETTE[0], PALETTE[1]]
OUTCOME_ORDER = ["both", "guided only", "unguided only", "neither"]
OUTCOME_COLORS = ["#4c9f70", "#2a78d6", "#eb6834", "#b8b8b4"]
WIN_ORDER = ["below brute force", "at or above"]
WIN_COLORS = ["#4c9f70", "#eb6834"]
BRUTE_COST_ORDER = ["guided failed", "guided proved, at or above", "guided proved, cheaper"]
BRUTE_COST_COLORS = ["#eb6834", "#eda100", "#4c9f70"]

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


def _method_color() -> alt.Color:
    return alt.Color(
        "method:N",
        sort=METHOD_ORDER,
        scale=alt.Scale(domain=METHOD_ORDER, range=METHOD_COLORS),
        legend=alt.Legend(title=None),
    )


def _guided_peak_scope(comparison: pl.DataFrame) -> str:
    if "guided_peak_scope" not in comparison.columns:
        return "guided workflow"
    scopes = comparison["guided_peak_scope"].drop_nulls().unique().to_list()
    return str(scopes[0]) if len(scopes) == 1 else "guided"


def success_rates(rates: pl.DataFrame, meta: dict) -> alt.Chart:
    """Success rates with Wilson intervals."""
    points = (
        alt.Chart(rates)
        .mark_point(filled=True, size=75)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("success_rate:Q", title="success rate", axis=alt.Axis(format="%")),
            y=alt.Y("mode:N", title=None, sort=list(meta["modes"])),
            color=_method_color(),
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
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("share:Q", title="share of planned pairs", axis=alt.Axis(format="%")),
            y=alt.Y("mode:N", title=None, sort=list(meta["modes"])),
            color=alt.Color(
                "outcome:N",
                sort=OUTCOME_ORDER,
                scale=alt.Scale(domain=OUTCOME_ORDER, range=OUTCOME_COLORS),
                legend=alt.Legend(title=None),
            ),
            order=alt.Order("outcome:N", sort="ascending"),
            tooltip=["mode:N", "outcome:N", "count:Q", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(title=_title("Paired outcomes", meta), height=alt.Step(30))
    )


def failure_causes(breakdown: pl.DataFrame, meta: dict) -> alt.Chart:
    """Pair-level failure causes, guided against unguided."""
    # A cause absent from a method carries no row
    causes = (
        breakdown.group_by("failure")
        .agg(pl.col("count").sum().alias("total"))
        .sort("total", "failure", descending=[True, False])["failure"]
        .to_list()
    )
    return (
        alt.Chart(breakdown)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("count:Q", title="failed pairs"),
            y=alt.Y("failure:N", title=None, sort=causes),
            yOffset=alt.YOffset("method:N", sort=METHOD_ORDER),
            color=_method_color(),
            row=alt.Row(
                "mode:N",
                title=None,
                sort=list(meta["modes"]),
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11),
            ),
            tooltip=[
                "mode:N",
                "method:N",
                "failure:N",
                "count:Q",
                alt.Tooltip("share_of_failures:Q", format=".1%", title="share of failures"),
                alt.Tooltip("share_of_planned:Q", format=".1%", title="share of planned pairs"),
            ],
        )
        .properties(title=_title("Failure causes", meta), width=420, height=alt.Step(34))
    )


def peak_scatter(comparison: pl.DataFrame, meta: dict) -> alt.Chart:
    """One explicitly scoped guided peak versus the brute-force proof cost."""
    guided_scope = _guided_peak_scope(comparison)
    title = f"{guided_scope.title()} vs brute-force proof"
    points = (
        alt.Chart(comparison)
        .mark_circle(size=45, opacity=0.58)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "brute_peak_mib:Q",
                title=f"brute-force proof {MEMORY_LABEL} (MiB, log)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "guided_peak_mib:Q",
                title=f"{guided_scope} {MEMORY_LABEL} (MiB, log)",
                scale=alt.Scale(type="log"),
            ),
            color=_mode_color(meta["modes"]),
            tooltip=[
                "mode:N",
                "start_term:N",
                "goal_term:N",
                alt.Tooltip("guided_peak_mib:Q", format=".1f"),
                alt.Tooltip("brute_peak_mib:Q", format=".1f"),
                alt.Tooltip("peak_ratio:Q", format=".3f"),
            ],
        )
    )
    if comparison.is_empty():
        return points.properties(
            title=_title(title, meta),
            width=420,
            height=380,
        )
    bounds = comparison.select(
        pl.min_horizontal("guided_peak_mib", "brute_peak_mib").min().alias("lo"),
        pl.max_horizontal("guided_peak_mib", "brute_peak_mib").max().alias("hi"),
    ).row(0, named=True)
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [bounds["lo"], bounds["hi"]]}))
        .mark_line(strokeDash=[5, 4], color="#777")
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", scale=alt.Scale(type="log")), y=alt.Y("x:Q", scale=alt.Scale(type="log"))
        )
    )
    return (diagonal + points).properties(
        title=_title(title, meta),
        width=420,
        height=380,
    )


def brute_cost_hist(binned: pl.DataFrame, meta: dict) -> alt.Chart:
    """Brute-force proof cost of every planned pair, grouped by guided outcome.

    One panel: each log-spaced bucket carries a bar per outcome, side by side
    from a shared baseline, so the pairs the guide could not prove, the ones it
    proved no cheaper than brute force, and the ones it proved cheaper are read
    against each other within the bucket. The bars sit in the slot edges the
    binner precomputed, an offset scale not applying to a continuous log axis.
    """
    return (
        alt.Chart(binned)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "slot_start_mib:Q",
                title=f"brute-force proof {MEMORY_LABEL} (MiB, log)",
                scale=alt.Scale(type="log", nice=False),
            ),
            x2="slot_end_mib:Q",
            y=alt.Y("count:Q", title="pairs"),
            # A bar given both x and x2 spans a range rather than resting on the
            # axis, so the baseline has to be named.
            y2=alt.datum(0),
            color=alt.Color(
                "outcome:N",
                sort=BRUTE_COST_ORDER,
                scale=alt.Scale(domain=BRUTE_COST_ORDER, range=BRUTE_COST_COLORS),
                legend=alt.Legend(title=None, columns=1),
            ),
            column=alt.Column("mode:N", title=None, sort=list(meta["modes"])),
            tooltip=[
                "mode:N",
                "outcome:N",
                "count:Q",
                alt.Tooltip("share:Q", format=".1%", title="share of outcome"),
                alt.Tooltip("bucket_share:Q", format=".1%", title="share of bucket"),
                alt.Tooltip("bucket_n:Q", title="pairs in bucket"),
                alt.Tooltip("bin_start_mib:Q", format=".0f", title="bucket from (MiB)"),
                alt.Tooltip("bin_end_mib:Q", format=".0f", title="bucket to (MiB)"),
            ],
        )
        .properties(
            title=_title(f"Brute-force proof {MEMORY_LABEL} by guided outcome", meta),
            width=560,
            height=300,
        )
    )


def peak_win_bars(counts: pl.DataFrame, meta: dict) -> alt.Chart:
    """Guided successes below versus at or above the brute-force proof peak."""
    data = counts.unpivot(
        index=["mode", "guided_peak_scope"],
        on=["n_below", "n_at_or_above"],
        variable_name="side",
        value_name="count",
    ).with_columns(
        pl.col("side").replace_strict({"n_below": WIN_ORDER[0], "n_at_or_above": WIN_ORDER[1]}),
        # Stack in WIN_ORDER rather than alphabetically by label.
        pl.col("side").replace_strict({"n_below": 0, "n_at_or_above": 1}).alias("side_order"),
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("count:Q", title="guided successes"),
            y=alt.Y("guided_peak_scope:N", title=None),
            color=alt.Color(
                "side:N",
                sort=WIN_ORDER,
                scale=alt.Scale(domain=WIN_ORDER, range=WIN_COLORS),
                legend=alt.Legend(title=None),
            ),
            order=alt.Order("side_order:Q", sort="ascending"),
            row=alt.Row(
                "mode:N",
                title=None,
                sort=list(meta["modes"]),
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11),
            ),
            tooltip=["mode:N", "guided_peak_scope:N", "side:N", "count:Q"],
        )
        .properties(
            title=_title(f"Guided {MEMORY_LABEL} versus brute-force proof", meta),
            width=420,
            height=alt.Step(26),
        )
    )


def peak_ratio_ecdf(comparison: pl.DataFrame, meta: dict) -> alt.Chart:
    """ECDF of one explicitly scoped guided peak over the brute-force proof cost."""
    guided_scope = _guided_peak_scope(comparison)
    data = comparison.with_columns(
        (pl.col("peak_ratio").rank("max").over("mode") / pl.len().over("mode")).alias("cdf")
    ).sort("mode", "peak_ratio")
    curves = (
        alt.Chart(data)
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "peak_ratio:Q",
                title=f"{guided_scope} / brute-force proof {MEMORY_LABEL} (log)",
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
        .encode(x=alt.X("ratio:Q", scale=alt.Scale(type="log")))  # ty: ignore[unresolved-attribute]
    )
    return (curves + parity).properties(
        title=_title(f"{guided_scope.title()} {MEMORY_LABEL} ratio", meta), width=460
    )


def absolute_peak_ecdf(comparison: pl.DataFrame, meta: dict) -> alt.Chart:
    """Absolute guided and brute-force peaks on a shared axis."""
    guided_scope = _guided_peak_scope(comparison)
    data = pl.concat(
        [
            comparison.select(
                "mode",
                (pl.col("guided_peak_mib")).alias("peak_mib"),
                pl.lit(guided_scope).alias("method"),
            ),
            comparison.select(
                "mode",
                (pl.col("brute_peak_mib")).alias("peak_mib"),
                pl.lit("brute-force proof").alias("method"),
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
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("peak_mib:Q", title=f"{MEMORY_LABEL} (MiB, log)", scale=alt.Scale(type="log")),
            y=alt.Y("cdf:Q", title="cumulative share", axis=alt.Axis(format="%")),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(
                    domain=[guided_scope, "brute-force proof"], range=[PALETTE[0], PALETTE[1]]
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
            title=_title(f"{guided_scope.title()} vs brute-force proof {MEMORY_LABEL}", meta),
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
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("success_attempt:O", title="attempt of first success"),
            y=alt.Y("count():Q", title="successful pairs"),
            color=_mode_color(meta["modes"]),
            column=alt.Column("mode:N", title=None, sort=list(meta["modes"])),
            tooltip=["mode:N", "success_attempt:O", "count():Q"],
        )
        .properties(title=_title("Attempts to success", meta), width=180, height=180)
    )
