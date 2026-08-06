"""Build predictor decision rows from scheduler-aware eqsat traces.

A row keyed by ``(term, upcoming_iter_index=k)`` represents the online hook
immediately before iteration ``k`` searches:

* allocation/egraph/scheduler state are sampled at the start of ``k``;
* timings, rebuilds, and applied counts come from completed iteration ``k-1``;
* the target is the sampled peak during iteration ``k``.

This explicit decision table replaces the old row-``k``/target-``k+1`` shift,
which attached scheduler state and egraph size to the wrong iteration.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl

PREVIOUS_WORK_FEATURES = (
    "total_applied",
    "hook_time",
    "search_time",
    "apply_time",
    "rebuild_time",
    "total_time",
    "n_rebuilds",
)

BASE_FEATURES = (
    "egraph_nodes",
    "egraph_classes",
    "nodes_per_class",
    "allocated",
    "bytes_per_node",
    "prev_growth",
    "prev_node_growth",
    *PREVIOUS_WORK_FEATURES,
    "iter_index",
    "term_size",
)

SCHEDULER_FEATURES = (
    "n_active",
    "n_banned",
    "n_newly_unbanned",
    "min_ban_remaining",
    "total_times_banned",
    "max_active_log2_match_limit",
    "log2_active_match_limit_sum",
    "max_active_times_banned",
)

RULE_FEATURE_SUFFIXES = (
    "will_search",
    "newly_unbanned",
    "times_banned",
    "ban_remaining",
    "log2_match_limit",
)


def escape_rule_name(name: str) -> str:
    """Percent-escape a raw UTF-8 egg rule name for an unambiguous column."""
    return "".join(
        chr(byte) if chr(byte).isalnum() and byte < 128 or byte == ord("-") else f"%{byte:02X}"
        for byte in name.encode()
    )


def rule_feature_name(rule: str, suffix: str) -> str:
    return f"rule_{escape_rule_name(rule)}_{suffix}"


def feature_schema(rules: Iterable[str]) -> tuple[str, ...]:
    """The deterministic manifest/ONNX/Rust feature order."""
    ordered_rules = tuple(sorted(rules))
    return (
        *BASE_FEATURES,
        *SCHEDULER_FEATURES,
        *(
            rule_feature_name(rule, suffix)
            for rule in ordered_rules
            for suffix in RULE_FEATURE_SUFFIXES
        ),
    )


def _seed_dirs(pattern: str = "") -> list[Path]:
    data_dir = Path(__file__).parent / ".." / "data" / "seed_terms"
    if not data_dir.is_dir():
        return []
    return sorted(
        (
            directory
            for directory in data_dir.iterdir()
            if directory.is_dir()
            and pattern in directory.name
            and (directory / "terms.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
    )


def resolve_seed_dir(pattern: str = "") -> Path:
    matches = _seed_dirs(pattern)
    if not matches:
        suffix = f" matching {pattern!r}" if pattern else ""
        raise FileNotFoundError(f"No seed-term directory with terms.json{suffix}")
    return matches[-1]


def _stop_kind(reason: object) -> str:
    if reason is None:
        return "completed"
    if isinstance(reason, str):
        return reason
    if isinstance(reason, dict) and len(reason) == 1:
        return next(iter(reason))
    return "Other"


def load_iterations(seed_dir: Path) -> pl.DataFrame:
    """Flatten traces while retaining exact upcoming per-rule state."""
    groups = json.loads((seed_dir / "terms.json").read_text())
    rows: list[dict] = []
    trace_rules: tuple[str, ...] | None = None

    for term_size, terms_map in groups:
        for term, payload in terms_map.items():
            _nodes, validation, measurement = payload
            iterations = measurement["iterations"]
            for index, iteration in enumerate(iterations):
                telemetry = iteration["data"]
                if "scheduler" not in telemetry or "iteration_peak_allocated" not in telemetry:
                    raise ValueError(
                        f"{seed_dir} uses the legacy end-allocation trace schema; "
                        "regenerate it with scheduler snapshots and iteration peaks"
                    )
                scheduler = telemetry["scheduler"]
                rules = tuple(sorted(rule["name"] for rule in scheduler["rules"]))
                if trace_rules is None:
                    trace_rules = rules
                elif rules != trace_rules:
                    raise ValueError(
                        f"inconsistent rule set in {seed_dir}: {rules!r} vs {trace_rules!r}"
                    )
                by_name = {rule["name"]: rule for rule in scheduler["rules"]}
                row = {
                    "term": term,
                    "term_size": term_size,
                    "iter_index": index,
                    "n_iters": len(iterations),
                    "egraph_nodes": iteration["egraph_nodes"],
                    "egraph_classes": iteration["egraph_classes"],
                    # Compatibility alias for exploratory plots; deployed
                    # decision rows overwrite this with the pre-search reading.
                    "allocated": telemetry["allocated"],
                    "iteration_end_allocated": telemetry["allocated"],
                    "iteration_start_allocated": telemetry["iteration_start_allocated"],
                    "iteration_peak_allocated": telemetry["iteration_peak_allocated"],
                    "iteration_peak_phase": telemetry["iteration_peak_phase"],
                    "iteration_peak_rule": telemetry["iteration_peak_rule"],
                    "scheduler_kind": scheduler["scheduler"],
                    "scheduler_rule_names": list(rules),
                    "iteration_total_applied": sum(iteration["applied"].values()),
                    "is_stop_iter": iteration["stop_reason"] is not None,
                    "stop_kind": _stop_kind(iteration["stop_reason"]),
                    "run_stop_reason": json.dumps(validation["stop_reason"]),
                    **{
                        feature: scheduler[feature]
                        for feature in SCHEDULER_FEATURES
                    },
                    **{
                        feature: iteration[feature]
                        for feature in PREVIOUS_WORK_FEATURES
                        if feature != "total_applied"
                    },
                }
                for rule in rules:
                    state = by_name[rule]
                    for suffix in RULE_FEATURE_SUFFIXES:
                        row[rule_feature_name(rule, suffix)] = state[suffix]
                rows.append(row)

    if not rows:
        raise ValueError(f"No iteration traces in {seed_dir}")
    frame = pl.DataFrame(rows, infer_schema_length=None)
    print(
        f"Loaded {seed_dir.name}: {frame['term'].n_unique()} terms, {len(frame)} iterations, "
        f"{len(trace_rules or ())} scheduler rules"
    )
    return frame


def rules_from_frame(df: pl.DataFrame) -> list[str]:
    """Return and validate the raw deterministic model rule set."""
    if "scheduler_rule_names" not in df.columns or df.is_empty():
        return []
    first = tuple(df["scheduler_rule_names"][0].to_list())
    for names in df["scheduler_rule_names"]:
        if tuple(names.to_list()) != first:
            raise ValueError("frame contains more than one scheduler rule set")
    return list(first)


def _trainable_expr() -> pl.Expr:
    completed = pl.col("stop_kind").is_in(["completed", "Saturated", "MemoryLimit"])
    useful_memory_stop = (
        (pl.col("stop_kind") != "MemoryLimit")
        | (pl.col("iteration_peak_phase") != "before_hooks")
    )
    return completed & useful_memory_stop


def build_decision_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Construct rows aligned exactly with the pre-search online hook."""
    ordered = df.sort("term", "iter_index")
    previous_start = pl.col("iteration_start_allocated").shift(1).over("term")
    previous_nodes = pl.col("egraph_nodes").shift(1).over("term")

    previous_work = [
        (
            pl.col("iteration_total_applied")
            if feature == "total_applied"
            else pl.col(feature)
        )
        .shift(1)
        .over("term")
        .alias(feature)
        for feature in PREVIOUS_WORK_FEATURES
    ]

    decisions = (
        ordered.with_columns(
            *previous_work,
            pl.col("iteration_start_allocated").alias("allocated"),
            (pl.col("egraph_nodes") / pl.col("egraph_classes")).alias("nodes_per_class"),
            (pl.col("iteration_start_allocated") / pl.col("egraph_nodes")).alias(
                "bytes_per_node"
            ),
            (pl.col("iteration_start_allocated") / previous_start).alias("prev_growth"),
            (pl.col("egraph_nodes") / previous_nodes).alias("prev_node_growth"),
            pl.col("iter_index").alias("upcoming_iter_index"),
            _trainable_expr().alias("target_trainable"),
        )
        # The online hook deliberately skips iteration zero because no previous
        # completed-work row exists yet.
        .filter(pl.col("iter_index") > 0)
        .filter(
            (pl.col("iteration_start_allocated") > 0)
            & (pl.col("iteration_peak_allocated") > 0)
        )
        .with_columns(
            *[
                pl.when(pl.col(column).is_finite())
                .then(pl.col(column))
                .otherwise(1.0)
                .fill_null(1.0)
                .alias(column)
                for column in (
                    "nodes_per_class",
                    "bytes_per_node",
                    "prev_growth",
                    "prev_node_growth",
                )
            ],
            (
                pl.col("iteration_peak_allocated")
                / pl.col("iteration_start_allocated")
            )
            .log()
            .alias("y_log_peak_growth"),
        )
    )
    print(
        f"Built {len(decisions)} decision rows over {decisions['term'].n_unique()} terms; "
        f"{decisions['target_trainable'].sum()} peak-growth targets are trainable"
    )
    return decisions


def build_transitions(
    df: pl.DataFrame, window: int = 1, *, keep_stop_transitions: bool = False
) -> pl.DataFrame:
    """Compatibility alias for notebooks; prediction now uses decision rows."""
    if window != 1:
        raise ValueError("history windows are not supported by the deployed decision-row schema")
    _ = keep_stop_transitions
    return build_decision_rows(df)


def feature_columns(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    rules = rules_from_frame(df)
    per_rule = [
        rule_feature_name(rule, suffix)
        for rule in rules
        for suffix in RULE_FEATURE_SUFFIXES
    ]
    return [*BASE_FEATURES, *SCHEDULER_FEATURES], per_rule


def rule_columns(df: pl.DataFrame) -> list[str]:
    """Compatibility name for the scheduler per-rule feature block."""
    return feature_columns(df)[1]


def window_columns(_df: pl.DataFrame) -> list[str]:
    return []
