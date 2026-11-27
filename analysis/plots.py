"""Altair plots for guided success and guided-vs-brute-force peak-memory comparisons."""

import json
from collections.abc import Sequence
from typing import Literal

import altair as alt
import polars as pl

from helpers import BRUTE_COST_OUTCOMES, MEASURES, PAIR_OUTCOMES, pow2_bucket_label

PALETTE = (
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
)
SUCCESS_COLOR = "#4c9f70"
FAILURE_COLOR = "#c1503f"
NEUTRAL_COLOR = "#b8b8b4"

GUIDED_COLOR = PALETTE[0]
UNGUIDED_COLOR = PALETTE[1]

OUTCOME_ORDER = ("both", "guided only", "unguided only", "neither")
OUTCOME_COLORS = (PALETTE[2], GUIDED_COLOR, UNGUIDED_COLOR, NEUTRAL_COLOR)
BRUTE_COST_COLORS = (FAILURE_COLOR, SUCCESS_COLOR)
MEASURE_COLORS = (SUCCESS_COLOR, NEUTRAL_COLOR)
PAIR_OUTCOME_COLORS = (NEUTRAL_COLOR, FAILURE_COLOR)

# Size of one small multiple in the strategy × setting grid.
CELL_WIDTH = 150
CELL_HEIGHT = 90

GRID_TOOLTIP = ("strategy:N", "setting:N")

THEME: alt.theme.ThemeConfig = {
    "config": {
        "view": {"continuousWidth": 360, "continuousHeight": 260, "strokeOpacity": 0},
        "axis": {
            "grid": True,
            "gridColor": "#e8e8e6",
            "domainColor": "#c9c9c6",
            "tickColor": "#c9c9c6",
        },
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


def _pairs(meta: dict) -> str:
    """Unit for pair counts (`pairs × seeds` when seeds are pooled)."""
    return "pairs × seeds" if meta.get("pooled") else "pairs"


def _ordered_color(field: str, order: Sequence[str], colors: Sequence[str]) -> alt.Color:
    """Color `field` with `colors` in `order`, without a legend title."""
    return alt.Color(
        field,
        sort=list(order),
        scale=alt.Scale(domain=list(order), range=list(colors)),
        legend=alt.Legend(title=None),
    )


def _strategy_row(meta: dict) -> alt.Row:
    """One facet row per search strategy, labelled on the left like a y-axis."""
    return alt.Row(
        "strategy:N",
        sort=meta["strategies"],
        # Right-aligned header labels reserve twice their width, hence left.
        header=alt.Header(labelAngle=0, labelAlign="left", labelOrient="left"),
    )


def _grid(meta: dict) -> dict:
    """Row and column channels for the strategy × setting grid."""
    return {
        "row": _strategy_row(meta),
        # Split setting parts across lines so labels fit their cell width.
        "column": alt.Column(
            "setting:N",
            sort=meta["settings"],
            header=alt.Header(labelExpr="split(datum.label, ' · ')"),
        ),
    }


def _pow2_x(counts: pl.DataFrame, title: str) -> alt.X:
    """Ordinal x-axis over power-of-two bucket labels."""
    first = int(counts["bucket"].min() or 0)  # ty: ignore[invalid-argument-type]
    last = int(counts["bucket"].max() or 0)  # ty: ignore[invalid-argument-type]
    return alt.X(
        "range:O",
        title=title,
        scale=alt.Scale(domain=[pow2_bucket_label(b) for b in range(first, last + 1)]),
        axis=alt.Axis(labelAngle=-45, grid=False),
    )


def success_outcomes(outcomes: pl.DataFrame, meta: dict) -> alt.Chart:
    """Distribution of paired success outcomes."""
    return (
        alt.Chart(outcomes)
        .transform_calculate(outcome_rank=f"indexof({json.dumps(OUTCOME_ORDER)}, datum.outcome)")
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("share:Q", title="share of planned pairs", axis=alt.Axis(format="%")),
            y=alt.Y("setting:N", sort=meta["settings"]),
            color=_ordered_color("outcome:N", OUTCOME_ORDER, OUTCOME_COLORS),
            order=alt.Order("outcome_rank:Q", sort="ascending"),
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
            x=alt.X("count:Q", title=_pairs(meta)),
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
        .properties(title=_title("Failed attempts", meta), width=CELL_WIDTH, height=CELL_HEIGHT)
    )


def peak_scatter(comparison: pl.DataFrame, meta: dict) -> alt.FacetChart:
    """Guided peaks vs. brute-force memory cost."""
    guided_scope = comparison["guided_peak_scope"][0]
    title = f"{guided_scope.title()} vs brute-force proof peak RSS"
    points = (
        alt.Chart()
        .mark_circle(size=30, opacity=0.5, color=GUIDED_COLOR)
        .encode(  # ty: ignore[unresolved-attribute]
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
    # Separate data so each cell draws the diagonal once rather than once per point.
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
    """Brute-force proof cost per pair, binned and split by guided outcome."""
    return (
        alt.Chart(binned)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "slot_start_mib:Q",
                title="brute force (MiB, log)",
                scale=alt.Scale(type="log", nice=False),
                axis=alt.Axis(grid=False, tickCount=3),
            ),
            x2="slot_end_mib:Q",
            y=alt.Y("count:Q", title=_pairs(meta)),
            # Range bars (x + x2) need an explicit baseline.
            y2=alt.datum(0),
            **_grid(meta),
            color=_ordered_color("outcome:N", BRUTE_COST_OUTCOMES, BRUTE_COST_COLORS),
            tooltip=[
                *GRID_TOOLTIP,
                "outcome:N",
                "count:Q",
                alt.Tooltip("share:Q", format=".1%", title="share of outcome"),
                alt.Tooltip("bucket_share:Q", format=".1%", title="share of bucket"),
                alt.Tooltip("bucket_n:Q", title=f"{_pairs(meta)} in bucket"),
                alt.Tooltip("bin_start_mib:Q", format=".0f", title="bucket from (MiB)"),
                alt.Tooltip("bin_end_mib:Q", format=".0f", title="bucket to (MiB)"),
            ],
        )
        .properties(
            title=_title("Brute-force proof peak RSS by guided outcome", meta),
            width=CELL_WIDTH,
            height=CELL_HEIGHT,
        )
    )


def _pairwise_matrix(
    diff: pl.DataFrame,
    meta: dict,
    *,
    field: str,
    scheme: Literal["blues", "reds"],
    legend_title: str,
    title: str,
    share_field: str,
    lower_triangle: bool = False,
) -> alt.VConcatChart:
    """Strategy-block heatmap over `pairwise_solved_diff`, built manually to avoid repeated facet axes."""
    strategies, order = meta["strategies"], meta["shorts"]
    short_rank = {short: i for i, short in enumerate(order)}
    peak = int(diff[field].max())  # ty: ignore[invalid-argument-type]
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
        alt.Tooltip(f"{share_field}:Q", format=".1%", title="share of planned pairs"),
    ]
    color = alt.Color(
        f"{field}:Q",
        title=legend_title,
        scale=alt.Scale(scheme=scheme, domain=[0, peak]),
        legend=alt.Legend(gradientLength=140),
    )

    def block(row: int, col: int) -> alt.LayerChart:
        first_col, last_row = col == 0, row == len(strategies) - 1
        y_axis = (
            alt.Axis(title=strategies[row], titleFontWeight="normal", **axis)  # ty: ignore[invalid-argument-type]
            if first_col
            else None
        )
        x_axis = alt.Axis(labelAngle=-90, title=None, **axis) if last_row else None  # ty: ignore[invalid-argument-type]
        subset = diff.filter(
            (pl.col("row_strategy") == strategies[row])
            & (pl.col("col_strategy") == strategies[col])
        )
        if lower_triangle:
            if row < col:
                subset = subset.clear()
            elif row == col:
                subset = subset.filter(
                    pl.col("row_short").replace_strict(short_rank)
                    > pl.col("col_short").replace_strict(short_rank)
                )
        base = alt.Chart(subset).encode(
            x=alt.X("col_short:N", sort=order, scale=alt.Scale(domain=order), axis=x_axis),
            y=alt.Y("row_short:N", sort=order, scale=alt.Scale(domain=order), axis=y_axis),
            tooltip=tooltip,
        )
        cells = base.mark_rect().encode(color=color)  # ty: ignore[unresolved-attribute]
        labels = base.mark_text(fontSize=9).encode(  # ty: ignore[unresolved-attribute]
            text=f"{field}:Q",
            color=alt.condition(alt.datum[field] > cutoff, alt.value("white"), alt.value("#333")),
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
        title=alt.TitleParams(title, fontSize=11, fontWeight="normal")
    )


def pairwise_solved(diff: pl.DataFrame, meta: dict) -> alt.HConcatChart:
    """Pairwise solved differences next to combined solved counts, in strategy blocks."""
    return (
        alt.hconcat(
            _pairwise_matrix(
                diff,
                meta,
                field="only_row",
                scheme="blues",
                legend_title="pairs won",
                title="rows solve, columns do not",
                share_field="share_only_row",
            ),
            _pairwise_matrix(
                diff,
                meta,
                field="combined",
                scheme="reds",
                legend_title="pairs solved",
                title="combined: either row or column solves",
                share_field="share_combined",
                lower_triangle=True,
            ),
            spacing=24,
        )
        .resolve_scale(color="independent")
        .properties(title=_title("Pairwise solved", meta, pooled=False))
    )


def _measure_bars(
    counts: pl.DataFrame, measure: str, axis_title: str, title: str, meta: dict
) -> alt.Chart:
    """Pairs per power-of-two bucket and measure."""
    order = list(MEASURES[measure].values())
    unit = _pairs(meta)
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=_pow2_x(counts, axis_title),
            xOffset=alt.XOffset("measure:N", sort=order),
            y=alt.Y("count:Q", title=unit),
            color=_ordered_color("measure:N", order, MEASURE_COLORS),
            **_grid(meta),
            tooltip=[
                *GRID_TOOLTIP,
                "measure:N",
                alt.Tooltip("range:O", title=axis_title),
                alt.Tooltip("count:Q", title=unit),
            ],
        )
        .properties(title=_title(title, meta), width=CELL_WIDTH, height=CELL_HEIGHT)
    )


def search_attempts(attempts: pl.DataFrame, meta: dict) -> alt.Chart:
    """Pairs by attempt of first success and by attempts run."""
    return _measure_bars(attempts, "attempts", "attempts", "Attempts", meta)


def search_depth(depths: pl.DataFrame, meta: dict) -> alt.Chart:
    """Pairs by depth of first success and of deepest attempt."""
    return _measure_bars(depths, "depth", "search-tree depth", "Search depth", meta)


def search_time(times: pl.DataFrame, meta: dict) -> alt.Chart:
    """Pairs by wall time to first success and total wall time."""
    return _measure_bars(times, "time", "wall time (s)", "Wall time", meta)


def sample_tries(tries: pl.DataFrame, meta: dict) -> alt.Chart:
    """Draws per power-of-two bucket of `sample` tries."""
    return (
        alt.Chart(tries)
        .mark_bar(color=NEUTRAL_COLOR)
        .encode(  # ty: ignore[unresolved-attribute]
            x=_pow2_x(tries, "sample tries per draw"),
            y=alt.Y("count:Q", title="draws", scale=alt.Scale(type="symlog")),
            **_grid(meta),
            tooltip=[
                *GRID_TOOLTIP,
                alt.Tooltip("range:O", title="sample tries per draw"),
                alt.Tooltip("count:Q", title="draws"),
            ],
        )
        .properties(title=_title("Sample tries", meta), width=CELL_WIDTH, height=CELL_HEIGHT)
    )


def killed_time(killed: pl.DataFrame, meta: dict) -> alt.Chart:
    """Share of `sample` wall time spent in tries killed at the RSS cap."""
    return (
        alt.Chart(killed)
        .mark_bar(color=FAILURE_COLOR)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("killed_share:Q", title="share of sample time", axis=alt.Axis(format="%")),
            y=alt.Y("setting:N", sort=meta["settings"]),
            row=_strategy_row(meta),
            tooltip=[
                *GRID_TOOLTIP,
                alt.Tooltip("killed_share:Q", format=".1%", title="killed share"),
                alt.Tooltip("killed_time:Q", format=".0f", title="killed tries (s)"),
                alt.Tooltip("sample_time:Q", format=".0f", title="sample time (s)"),
                alt.Tooltip("retried:Q", title="retried draws"),
                "draws:Q",
            ],
        )
        .properties(title=_title("Sample time lost to killed tries", meta), width=CELL_WIDTH * 2)
    )
