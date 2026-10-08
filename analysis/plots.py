"""Altair plots for guided success and guided-vs-brute-force peak-memory comparisons."""

from collections.abc import Sequence

import altair as alt
import polars as pl

from helpers import BRUTE_COST_OUTCOMES, PAIR_OUTCOMES

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
# Success green, failure orange, and neutral gray
SUCCESS_COLOR = "#4c9f70"
FAILURE_COLOR = "#c1503f"
NEUTRAL_COLOR = "#b8b8b4"

GUIDED_COLOR = PALETTE[0]
UNGUIDED_COLOR = PALETTE[1]

METHOD_ORDER = ["guided", "unguided"]
METHOD_COLORS = [GUIDED_COLOR, UNGUIDED_COLOR]
OUTCOME_ORDER = ["both", "guided only", "unguided only", "neither"]
OUTCOME_COLORS = [PALETTE[2], GUIDED_COLOR, UNGUIDED_COLOR, NEUTRAL_COLOR]
# In `BRUTE_COST_OUTCOMES` order.
BRUTE_COST_COLORS = [FAILURE_COLOR, SUCCESS_COLOR]
DEPTH_ORDER = ["first success", "deepest attempt"]
DEPTH_COLORS = [SUCCESS_COLOR, NEUTRAL_COLOR]
# In `PAIR_OUTCOMES` order.
PAIR_OUTCOME_COLORS = [NEUTRAL_COLOR, FAILURE_COLOR]

# Size of one small multiple in the strategy × setting grid.
# Used as the multiple baseline for all others
CELL_WIDTH = 150
CELL_HEIGHT = 90

GRID_TOOLTIP = ["strategy:N", "setting:N"]

THEME: alt.theme.ThemeConfig = {
    "config": {
        "view": {"continuousWidth": 360, "continuousHeight": 260, "strokeOpacity": 0},
        "axis": {
            "grid": True,
            "gridColor": "#e8e8e6",
            "domainColor": "#c9c9c6",
            "tickColor": "#c9c9c6",
        },
        # Category labels name themselves.
        "axisYDiscrete": {"title": None, "labelLimit": 0},
        "axisYQuantitative": {"tickCount": 3},
        "legend": {"orient": "top", "titleFontSize": 11, "labelFontSize": 11, "labelLimit": 0},
        "header": {"title": None, "labelFontSize": 11},
        "title": {"fontSize": 13, "anchor": "start", "subtitleColor": "#777"},
        "range": {"category": PALETTE},
        "facet": {"spacing": 8},
    }
}


def _title(text: str, meta: dict, pooled: bool = True) -> alt.TitleParams:
    subtitle = list(meta.get("subtitle", []))
    if pooled and meta.get("pooled") and subtitle:
        subtitle[-1] += f" · {meta['pooled']}"
    return alt.TitleParams(text, subtitle=subtitle)


def _ordered_color(field: str, order: Sequence[str], colors: Sequence[str]) -> alt.Color:
    """Color `field` with `colors`, in `order` for both the scale and the untitled legend."""
    return alt.Color(
        field,
        sort=list(order),
        scale=alt.Scale(domain=list(order), range=list(colors)),
        legend=alt.Legend(title=None),
    )


def _strategy_row(meta: dict, field: str = "strategy") -> alt.Row:
    """One facet row per search strategy, labelled on the left like a y-axis."""
    return alt.Row(
        f"{field}:N",
        sort=list(meta["strategies"]),
        # Right-aligned header labels reserve twice their width, hence left.
        header=alt.Header(labelAngle=0, labelAlign="left", labelOrient="left"),
    )


def _grid(meta: dict) -> dict:
    """Row and column channels laying a chart out as a strategy × setting grid."""
    return {
        "row": _strategy_row(meta),
        "column": alt.Column("setting:N", sort=list(meta["settings"])),
    }


def _setting_axis(meta: dict) -> alt.Y:
    """Settings along y, for the one-value-per-cell charts faceted only by strategy."""
    return alt.Y("setting:N", sort=list(meta["settings"]))


def success_rates(rates: pl.DataFrame, meta: dict) -> alt.Chart:
    """Guided and unguided success rate per grid cell."""
    return (
        alt.Chart(rates)
        .mark_point(filled=True, size=75)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("success_rate:Q", title="success rate", axis=alt.Axis(format="%")),
            y=_setting_axis(meta),
            color=_ordered_color("method:N", METHOD_ORDER, METHOD_COLORS),
            row=_strategy_row(meta),
            tooltip=[
                *GRID_TOOLTIP,
                "method:N",
                "baseline:N",
                "successes:Q",
                "n:Q",
                alt.Tooltip("success_rate:Q", format=".1%"),
            ],
        )
        .properties(title=_title("Success rate", meta), width=CELL_WIDTH * 2)
    )


def success_outcomes(outcomes: pl.DataFrame, meta: dict) -> alt.Chart:
    """Distribution of paired success outcomes."""
    return (
        alt.Chart(outcomes)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("share:Q", title="share of planned pairs", axis=alt.Axis(format="%")),
            y=_setting_axis(meta),
            color=_ordered_color("outcome:N", OUTCOME_ORDER, OUTCOME_COLORS),
            order=alt.Order("outcome:N", sort="ascending"),
            row=_strategy_row(meta),
            tooltip=[*GRID_TOOLTIP, "outcome:N", "count:Q", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(title=_title("Paired outcomes", meta), width=CELL_WIDTH * 2)
    )


def search_outcomes(outcomes: pl.DataFrame, meta: dict) -> alt.Chart:
    """How every pair's search ended."""
    return (
        alt.Chart(outcomes)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("count:Q", title="pairs"),
            y=alt.Y("outcome:N"),
            color=alt.condition(
                alt.datum.outcome == "reached",
                alt.value(SUCCESS_COLOR),
                alt.value(FAILURE_COLOR),
            ),
            **_grid(meta),
            tooltip=[
                *GRID_TOOLTIP,
                "outcome:N",
                "count:Q",
                alt.Tooltip("share:Q", format=".1%", title="share of planned pairs"),
            ],
        )
        .properties(title=_title("How the search ended", meta), width=CELL_WIDTH)
    )


def attempt_failures(failures: pl.DataFrame, meta: dict) -> alt.Chart:
    """Failed attempts by stop reason, split by whether their pair was eventually proved."""
    return (
        alt.Chart(failures)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("count:Q", title="failed attempts"),
            y=alt.Y("failure:N"),
            yOffset=alt.YOffset("pair_outcome:N", sort=list(PAIR_OUTCOMES)),
            color=_ordered_color("pair_outcome:N", PAIR_OUTCOMES, PAIR_OUTCOME_COLORS),
            **_grid(meta),
            tooltip=[
                *GRID_TOOLTIP,
                alt.Tooltip("pair_outcome:N", title="pair"),
                "failure:N",
                "count:Q",
                alt.Tooltip("share:Q", format=".1%", title="share of the pair group's failures"),
            ],
        )
        # Sized per bar rather than per stop reason, so the pair of bars stays as
        # thin as the single bars of "How the search ended".
        .properties(title=_title("Failed attempts", meta), width=CELL_WIDTH, height=CELL_HEIGHT)
    )


def peak_scatter(comparison: pl.DataFrame, meta: dict) -> alt.FacetChart:
    """Guided peaks vs. brute-force memory cost."""
    guided_scope = comparison["guided_peak_scope"][0]
    title = f"{guided_scope.title()} vs brute-force proof {MEMORY_LABEL}"
    points = (
        alt.Chart()
        .mark_circle(size=30, opacity=0.5, color=GUIDED_COLOR)
        .encode(  # ty: ignore[unresolved-attribute]
            # The title names the metric; a grid cell is too narrow to repeat it.
            x=alt.X("brute_peak_mib:Q", title="brute force (MiB)"),
            y=alt.Y("guided_peak_mib:Q", title=f"{guided_scope} (MiB)"),
            tooltip=[
                *GRID_TOOLTIP,
                "mode:N",
                "start:N",
                "goal:N",
                alt.Tooltip("guided_peak_mib:Q", format=".1f"),
                alt.Tooltip("brute_peak_mib:Q", format=".1f"),
                alt.Tooltip("peak_ratio:Q", format=".3f"),
            ],
        )
    )
    bounds = comparison.select(
        pl.min_horizontal("guided_peak_mib", "brute_peak_mib").min().alias("lo"),
        pl.max_horizontal("guided_peak_mib", "brute_peak_mib").max().alias("hi"),
    ).row(0, named=True)
    # Its own data, so every cell draws it once instead of once per point.
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [bounds["lo"], bounds["hi"]]}))
        .mark_line(strokeDash=[5, 4], color="#777")
        .encode(x=alt.X("x:Q"), y=alt.Y("x:Q"))  # ty: ignore[unresolved-attribute]
    )
    return (
        alt.layer(diagonal, points, data=comparison)
        .properties(width=CELL_WIDTH, height=CELL_WIDTH)
        .facet(**_grid(meta))
        .properties(title=_title(title, meta))
    )


def brute_cost_hist(binned: pl.DataFrame, meta: dict) -> alt.Chart:
    """Success comparison for every pair, bucketed by brute force memory cost"""
    return (
        alt.Chart(binned)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "slot_start_mib:Q",
                # The title names the metric; a grid cell is too narrow to repeat it.
                title="brute force (MiB, log)",
                scale=alt.Scale(type="log", nice=False),
                axis=alt.Axis(grid=False, tickCount=3),
            ),
            x2="slot_end_mib:Q",
            y=alt.Y("count:Q", title="pairs"),
            # A bar given both x and x2 spans a range rather than resting on the
            # axis, so the baseline has to be named.
            y2=alt.datum(0),
            **_grid(meta),
            color=_ordered_color("outcome:N", BRUTE_COST_OUTCOMES, BRUTE_COST_COLORS),
            tooltip=[
                *GRID_TOOLTIP,
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
            width=CELL_WIDTH,
            height=CELL_HEIGHT,
        )
    )


def solved_diff_matrix(diff: pl.DataFrame, meta: dict) -> alt.VConcatChart:
    """Problems the row run solves that the column run does not, in strategy blocks.

    One block per strategy pair, built by hand rather than faceted: a facet with
    per-block axes repeats every label nine times and mis-orders its headers.
    """
    strategies, order = list(meta["strategies"]), list(meta["shorts"])
    peak = int(diff["only_row"].max())  # ty: ignore[invalid-argument-type]
    # Half the range keeps the light end of the scheme readable under dark text.
    cutoff = peak / 2
    axis = {"labelLimit": 0, "labelFontSize": 9, "grid": False, "ticks": False, "domain": False}
    tooltip = [
        alt.Tooltip("row_mode:N", title="row"),
        alt.Tooltip("col_mode:N", title="column"),
        alt.Tooltip("only_row:Q", title="row only"),
        alt.Tooltip("only_col:Q", title="column only"),
        alt.Tooltip("both:Q", title="shared"),
        alt.Tooltip("combined:Q", title="combined"),
        "neither:Q",
        alt.Tooltip("net:Q", title="row − column"),
        alt.Tooltip("n_shared:Q", title="planned pairs"),
        alt.Tooltip("share_only_row:Q", format=".1%", title="share of planned pairs"),
    ]
    color = alt.Color(
        "only_row:Q",
        title="pairs won",
        scale=alt.Scale(scheme="blues", domain=[0, peak]),
        legend=alt.Legend(gradientLength=140),
    )

    def block(row: int, col: int) -> alt.LayerChart:
        # Labels only on the outer edges: settings left and below, strategies
        # as the axis titles there and as the top row's titles.
        first_col, last_row = col == 0, row == len(strategies) - 1
        y_axis = (
            alt.Axis(title=strategies[row], titleFontWeight="normal", **axis)  # ty: ignore[invalid-argument-type]
            if first_col
            else None
        )
        x_axis = alt.Axis(labelAngle=-90, title=None, **axis) if last_row else None  # ty: ignore[invalid-argument-type]
        base = alt.Chart(
            diff.filter(
                (pl.col("row_strategy") == strategies[row])
                & (pl.col("col_strategy") == strategies[col])
            )
        ).encode(
            x=alt.X("col_short:N", sort=order, axis=x_axis),
            y=alt.Y("row_short:N", sort=order, axis=y_axis),
            tooltip=tooltip,
        )
        cells = base.mark_rect().encode(color=color)  # ty: ignore[unresolved-attribute]
        labels = base.mark_text(fontSize=9).encode(  # ty: ignore[unresolved-attribute]
            text="only_row:Q",
            color=alt.condition(alt.datum.only_row > cutoff, alt.value("white"), alt.value("#333")),
        )
        chart = (cells + labels).properties(width=alt.Step(22), height=alt.Step(22))
        if row == 0:
            chart = chart.properties(
                title=alt.TitleParams(
                    strategies[col], anchor="middle", fontSize=11, fontWeight="normal"
                )
            )
        return chart

    grid = [
        alt.hconcat(*(block(r, c) for c in range(len(strategies))), spacing=6)
        for r in range(len(strategies))
    ]
    return alt.vconcat(*grid, spacing=6).properties(
        title=_title("Pairwise solved differences: rows solve, columns do not", meta, pooled=False)
    )


def _sparse_ordinal(field: str, title: str, last: int, every: int = 5) -> alt.X:
    """Ordinal x over 1..`last`, labelled at 1 and every `every`th value to fit a grid cell.

    Every value gets a column, so one nothing landed on shows as a gap.
    """
    return alt.X(
        field,
        title=title,
        scale=alt.Scale(domain=list(range(1, last + 1))),
        axis=alt.Axis(
            labelAngle=0,
            grid=False,
            values=[1, *range(every, last + 1, every)],
        ),
    )


def attempts_to_success(frame: pl.DataFrame, meta: dict) -> alt.Chart:
    """Distribution of the successful guided attempt."""
    data = frame.filter(pl.col("guided_success"))
    max_attempt = int(data["success_attempt"].max())  # ty: ignore[invalid-argument-type]
    return (
        alt.Chart(data)
        .mark_bar(color=GUIDED_COLOR)
        .encode(  # ty: ignore[unresolved-attribute]
            x=_sparse_ordinal("success_attempt:O", "attempt of first success", max_attempt),
            y=alt.Y("count():Q", title="successful pairs"),
            **_grid(meta),
            tooltip=[*GRID_TOOLTIP, "success_attempt:O", "count():Q"],
        )
        .properties(title=_title("Attempts to success", meta), width=CELL_WIDTH, height=CELL_HEIGHT)
    )


def search_depth(depths: pl.DataFrame, meta: dict) -> alt.Chart:
    """How deep each run's search went: pairs per depth of first success and deepest attempt."""
    max_depth = int(depths["depth"].max() or 0)  # ty: ignore[invalid-argument-type]
    return (
        alt.Chart(depths)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=_sparse_ordinal("depth:O", "search-tree depth", max_depth),
            xOffset=alt.XOffset("measure:N", sort=DEPTH_ORDER),
            y=alt.Y("count:Q", title="pairs"),
            color=_ordered_color("measure:N", DEPTH_ORDER, DEPTH_COLORS),
            **_grid(meta),
            tooltip=[*GRID_TOOLTIP, "measure:N", "depth:O", "count:Q"],
        )
        .properties(title=_title("Search depth", meta), width=CELL_WIDTH, height=CELL_HEIGHT)
    )
