import polars as pl

# The fields `attempt_summary` flattens a payload into, and the polars dtypes
# they are read back at. Both come from here so they cannot drift: an
# unreached or panicked attempt leaves most of them None, and an
# unreached-heavy prefix would otherwise make polars infer Null and reject the
# first real value.
ATTEMPT_DTYPES = {
    "reached": pl.Boolean,
    "panic": pl.Boolean,
    "stop_reason": pl.String,
    "iters": pl.Int64,
    "nodes": pl.Int64,
    "classes": pl.Int64,
    "total_applied": pl.Int64,
    "total_time": pl.Float64,
    "memory": pl.Int64,
    "peak_live_heap": pl.Int64,
}


# One row per `attempt` process.
ATTEMPT_SCHEMA = {
    "start_term": pl.String,
    "goal_term": pl.String,
    "policy": pl.String,
    "attempt": pl.Int64,
    "node_id": pl.Int64,
    "parent_id": pl.Int64,
    "depth": pl.Int64,
    "guide_s_expr": pl.String,
    "terminal": pl.Boolean,
    "started_at": pl.Float64,
    "wall_time": pl.Float64,
    **ATTEMPT_DTYPES,
    "attempt_peak_rss_bytes": pl.Int64,
}

# One row per `samples` process the search *used*, so a pool reused from the
# cache is charged to every pair that consumed it
EXPANSION_SCHEMA = {
    "start_term": pl.String,
    "goal_term": pl.String,
    "node_id": pl.Int64,
    "depth": pl.Int64,
    "status": pl.String,
    "cached": pl.Boolean,
    "saturated": pl.Boolean,
    "drawn": pl.Int64,
    "pushed": pl.Int64,
    "started_at": pl.Float64,
    "wall_time": pl.Float64,
    "guide_nodes": pl.Int64,
    "guide_classes": pl.Int64,
    "guide_time": pl.Float64,
    "guide_memory": pl.Int64,
    "guide_peak_live_heap": pl.Int64,
    "guide_stop_reason": pl.String,
    "sample_peak_rss_bytes": pl.Int64,
}

# One row per pair.
PAIR_SCHEMA = {
    "start_term": pl.String,
    "goal_term": pl.String,
    "policy": pl.String,
    "exploration_policy": pl.String,
    "max_depth": pl.Int64,
    "branching": pl.Int64,
    "attempt_budget": pl.Int64,
    "time_budget": pl.Float64,
    "guided_success": pl.Boolean,
    "search_stop_reason": pl.String,
    "success_attempt": pl.Int64,
    "success_depth": pl.Int64,
    "attempts_run": pl.Int64,
    "expansions_run": pl.Int64,
    "expansions_paid": pl.Int64,
    "saturated_expansions": pl.Int64,
    "root_saturated": pl.Boolean,
    "deepest_attempt": pl.Int64,
    "pair_wall_time": pl.Float64,
    "pair_cost_time": pl.Float64,
    "guided_stop_reason": pl.String,
    "guided_panic": pl.Boolean,
    "setup_status": pl.String,
    "sample_status": pl.String,
    "attempt_peak_rss_bytes": pl.Int64,
    "attempt_peak_rss_bytes_max": pl.Int64,
    "sample_peak_rss_bytes": pl.Int64,
    "guided_peak_rss_bytes": pl.Int64,
    "guided_peak_live_heap_bytes": pl.Int64,
}

# An rss_killed baseline leaves all measurement fields None.
UNGUIDED_SCHEMA = {
    "start_term": pl.String,
    "goal_term": pl.String,
    "unguided_success": pl.Boolean,
    "unguided_stop_reason": pl.String,
    "unguided_panic": pl.Boolean,
    "unguided_final_live_heap_bytes": pl.Int64,
    "unguided_peak_live_heap_bytes": pl.Int64,
    "unguided_peak_rss_bytes": pl.Int64,
}

# What an expansion contributes to a row when its `samples` process never got
# far enough to report.
EMPTY_GUIDE_META = {
    k: None
    for k in [
        "guide_nodes",
        "guide_classes",
        "guide_time",
        "guide_memory",
        "guide_peak_live_heap",
        "guide_stop_reason",
        "sample_peak_rss_bytes",
    ]
}
