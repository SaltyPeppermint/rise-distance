"""Per-iteration eqsat traces reshaped into next-iteration prediction rows.

`generate.py` writes a `Measurement` per seed term into `terms.json`: one
record per eqsat iteration with egraph size, rule applications, timings, and
the live-heap reading from jemalloc. This module flattens those traces into a
supervised frame where each row pairs iteration `i`'s state with iteration
`i+1`'s memory use.
"""

import json
from pathlib import Path

import polars as pl

# Per-iteration fields copied straight out of egg's `Iteration`.
ITER_SCALARS = (
    "egraph_nodes",
    "egraph_classes",
    "hook_time",
    "search_time",
    "apply_time",
    "rebuild_time",
    "total_time",
    "n_rebuilds",
)

SCHEDULER_SCALARS = (
    "n_banned",
    "n_unbanned_this_iter",
    "min_ban_remaining",
    "total_times_banned",
)


def _seed_dirs(pattern: str = "") -> list[Path]:
    """Seed-term directories matching `pattern`, oldest first."""
    data_dir = Path(__file__).parent / ".." / "data" / "seed_terms"
    if not data_dir.is_dir():
        return []
    return sorted(
        (
            d
            for d in data_dir.iterdir()
            if d.is_dir() and pattern in d.name and (d / "terms.json").is_file()
        ),
        key=lambda p: p.stat().st_mtime,
    )


def resolve_seed_dir(pattern: str = "") -> Path:
    """Resolve the newest seed-term directory matching `pattern`."""
    matches = _seed_dirs(pattern)
    if not matches:
        suffix = f" matching {pattern!r}" if pattern else ""
        raise FileNotFoundError(f"No seed-term directory with terms.json{suffix}")
    return matches[-1]


def load_iterations(seed_dir: Path) -> pl.DataFrame:
    """Flatten every seed term's iteration trace into one long frame.

    One row per (term, iteration). `applied` expands to one `rule_<name>`
    column per rewrite, zero-filled where a rule did not fire. `is_stop_iter`
    marks the final iteration of a run, whose measurements are not comparable
    with the others (see `build_transitions`).
    """
    groups = json.loads((seed_dir / "terms.json").read_text())

    rows: list[dict] = []
    for term_size, terms_map in groups:
        for term, payload in terms_map.items():
            # (node_count, ValidationResult, Measurement) per generate.rs.
            _nodes, validation, measurement = payload
            iterations = measurement["iterations"]
            for index, it in enumerate(iterations):
                row = {
                    "term": term,
                    "term_size": term_size,
                    "iter_index": index,
                    "n_iters": len(iterations),
                    "allocated": it["data"]["allocated"],
                    **{field: it["data"].get(field) for field in SCHEDULER_SCALARS},
                    "is_stop_iter": it["stop_reason"] is not None,
                    "run_stop_reason": json.dumps(validation["stop_reason"]),
                    **{field: it[field] for field in ITER_SCALARS},
                    **{f"rule_{rule}": count for rule, count in it["applied"].items()},
                }
                rows.append(row)

    if not rows:
        raise ValueError(f"No iteration traces in {seed_dir}")

    rule_columns = sorted({key for row in rows for key in row if key.startswith("rule_")})
    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        *[pl.col(col).fill_null(0) for col in rule_columns]
    )
    print(
        f"Loaded {seed_dir.name}: {df['term'].n_unique()} terms, {len(df)} iterations, "
        f"{len(rule_columns)} rewrite rules"
    )
    return df


def rule_columns(df: pl.DataFrame) -> list[str]:
    """Names of the per-rule application-count columns."""
    return sorted(col for col in df.columns if col.startswith("rule_"))


def build_transitions(df: pl.DataFrame, window: int = 1) -> pl.DataFrame:
    """Pair each iteration with the next one's memory use.

    Features are strictly iteration `i`; the targets read iteration `i+1`.
    Derived ratios (`nodes_per_class`, `bytes_per_node`, growth rates) stay on
    the feature side, so nothing downstream sees the future.

    The final iteration of a run is dropped as a *target*: egg records it with
    a pre-apply node count but a post-run heap reading, so its memory is not
    comparable with a mid-run iteration. It is still usable as a *source* row
    only if it is not the stop iteration, which the shift-based construction
    handles naturally.

    `window` is how many iterations of history each row sees, including the
    current one; `window > 1` adds the lag/delta columns from
    `add_window_features`. Those lags are taken *before* the row filters below,
    because dropped mid-run rows would otherwise let a window span a gap and
    quietly pair non-adjacent iterations.
    """
    ordered = df.sort("term", "iter_index")

    # Derived ratios must exist before lagging, so the window sees them too.
    ordered = ordered.with_columns(
        (pl.col("egraph_nodes") / pl.col("egraph_classes")).alias("nodes_per_class"),
        (pl.col("allocated") / pl.col("egraph_nodes")).alias("bytes_per_node"),
        pl.sum_horizontal(rule_columns(ordered)).alias("total_applied"),
    )
    ordered, _ = add_window_features(ordered, window)

    nxt = pl.col("allocated").shift(-1).over("term")

    # Growth over the *previous* iteration: a past-only feature, unlike the target.
    prev_allocated = pl.col("allocated").shift(1).over("term")
    prev_nodes = pl.col("egraph_nodes").shift(1).over("term")

    transitions = (
        ordered.with_columns(
            nxt.alias("next_allocated"),
            pl.col("is_stop_iter").shift(-1).over("term").alias("next_is_stop_iter"),
            (pl.col("allocated") / prev_allocated).alias("prev_growth"),
            (pl.col("egraph_nodes") / prev_nodes).alias("prev_node_growth"),
        )
        # Drop the run's last row (no successor) and any transition *into* a
        # stop iteration, whose heap reading is a post-run total.
        .filter(pl.col("next_allocated").is_not_null() & pl.col("next_is_stop_iter").not_())
        # Log targets need strictly positive readings on both ends.
        .filter((pl.col("allocated") > 0) & (pl.col("next_allocated") > 0))
        .with_columns(
            # First iteration has no predecessor, and a zero-heap predecessor
            # divides to infinity; treat both as flat growth.
            *[
                pl.when(pl.col(col).is_finite())
                .then(pl.col(col))
                .otherwise(1.0)
                .fill_null(1.0)
                .alias(col)
                for col in ("prev_growth", "prev_node_growth")
            ],
        )
        .with_columns(
            pl.col("next_allocated").log().alias("y_log_next"),
            (pl.col("next_allocated") / pl.col("allocated")).log().alias("y_log_growth"),
        )
    )
    print(
        f"Built {len(transitions)} transitions over {transitions['term'].n_unique()} terms "
        f"(dropped {len(df) - len(transitions)} rows: run ends, stop iterations, zero heap)"
    )
    return transitions


def feature_columns(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Split the feature space into scalar and per-rule blocks."""
    scalars = [
        "egraph_nodes",
        "egraph_classes",
        "nodes_per_class",
        "allocated",
        "bytes_per_node",
        "prev_growth",
        "prev_node_growth",
        "total_applied",
        "hook_time",
        "search_time",
        "apply_time",
        "rebuild_time",
        "total_time",
        "n_rebuilds",
        "iter_index",
        "term_size",
        *SCHEDULER_SCALARS,
    ]
    return [col for col in scalars if col in df.columns], rule_columns(df)


# Scalars worth carrying a history for. `iter_index` and `term_size` are
# constant or trivially known across a window, so lagging them adds nothing.
WINDOW_SCALARS = (
    "egraph_nodes",
    "egraph_classes",
    "nodes_per_class",
    "allocated",
    "bytes_per_node",
    "total_applied",
    "hook_time",
    "search_time",
    "apply_time",
    "rebuild_time",
    "total_time",
    "n_rebuilds",
    *SCHEDULER_SCALARS,
)

# Lagged differences are taken in log space for the heavy-tailed, strictly
# growing quantities, where a ratio is the meaningful step.
WINDOW_LOG_DELTAS = ("allocated", "egraph_nodes", "egraph_classes", "search_time", "total_time")


def add_window_features(df: pl.DataFrame, n: int) -> tuple[pl.DataFrame, list[str]]:
    """Attach `n - 1` iterations of scalar history to each transition row.

    `n` counts the window *including* the current iteration, so `n = 1` is the
    current model and adds nothing. For each lag `k` in `1..n-1` this emits
    `<col>_lag<k>` for every `WINDOW_SCALARS` column, plus per-step log deltas
    `d_log_<col>_<k>` measuring the move from lag `k` to lag `k - 1`.

    Rows earlier than `k` iterations into a run have no true lag; those are
    backfilled with the run's first iteration, which keeps every transition row
    and makes the sweep over `n` a like-for-like comparison. Backfilling makes
    the deltas of a warm-up row zero, i.e. "no change observed yet", rather
    than dropping the row or handing the model a null.

    Lags are taken over the term's full iteration ordering, so a window never
    silently spans a gap left by a filtered-out row.
    """
    if n < 1:
        raise ValueError(f"window must include at least the current iteration, got n={n}")
    if n == 1:
        return df, []

    ordered = df.sort("term", "iter_index")
    available = [col for col in WINDOW_SCALARS if col in ordered.columns]

    lagged, names = [], []
    for k in range(1, n):
        for col in available:
            name = f"{col}_lag{k}"
            # `shift` yields null past the start of a term; backfill with the
            # earliest iteration so warm-up rows survive.
            lagged.append(
                pl.col(col)
                .shift(k)
                .over("term")
                .fill_null(pl.col(col).first().over("term"))
                .alias(name)
            )
            names.append(name)

    windowed = ordered.with_columns(lagged)

    deltas, delta_names = [], []
    for k in range(1, n):
        for col in WINDOW_LOG_DELTAS:
            if col not in available:
                continue
            newer = pl.col(f"{col}_lag{k - 1}") if k > 1 else pl.col(col)
            older = pl.col(f"{col}_lag{k}")
            name = f"d_log_{col}_{k}"
            # Zero heap/count readings make the ratio undefined; a flat step is
            # the honest reading of "nothing measurable moved".
            deltas.append(
                pl.when((newer > 0) & (older > 0))
                .then((newer / older).log())
                .otherwise(0.0)
                .alias(name)
            )
            delta_names.append(name)

    windowed = windowed.with_columns(deltas)
    return windowed, names + delta_names


def window_columns(df: pl.DataFrame) -> list[str]:
    """Names of the lag and delta columns added by `add_window_features`."""
    return sorted(col for col in df.columns if "_lag" in col or col.startswith("d_log_"))
