import polars as pl

ATTEMPT_DTYPES = {
    "reached": pl.Boolean,
    "panic": pl.Boolean,
    "stop_reason": pl.String,
    "iters": pl.Int64,
    "nodes": pl.Int64,
    "classes": pl.Int64,
    "total_applied": pl.Int64,
    "total_time": pl.Float64,
    "final_live_heap": pl.Int64,
    "peak_live_heap": pl.Int64,
}


# One row per `attempt` process.
ATTEMPT_SCHEMA = {
    "start": pl.String,
    "goal": pl.String,
    "sample_policy": pl.String,
    "attempt": pl.Int64,
    "node_id": pl.Int64,
    "parent_id": pl.Int64,
    "depth": pl.Int64,
    "guide_s_expr": pl.String,
    "terminal": pl.Boolean,
    "started_at": pl.Float64,
    "wall_time": pl.Float64,
    **ATTEMPT_DTYPES,
    "peak_rss": pl.Int64,
}

# One row per `sample` process the search *used*, so a pool reused from the
# cache is charged to every pair that consumed it
EXPANSION_SCHEMA = {
    "start": pl.String,
    "goal": pl.String,
    "node_id": pl.Int64,
    "depth": pl.Int64,
    "status": pl.String,
    "cached": pl.Boolean,
    "saturated": pl.Boolean,
    "drawn": pl.Int64,
    "pushed": pl.Int64,
    "started_at": pl.Float64,
    "wall_time": pl.Float64,
    "iters": pl.Int64,
    "nodes": pl.Int64,
    "classes": pl.Int64,
    "total_time": pl.Float64,
    "final_live_heap": pl.Int64,
    "peak_live_heap": pl.Int64,
    "stop_reason": pl.String,
    "peak_rss": pl.Int64,
}

# One row per pair.
PAIR_SCHEMA = {
    "start": pl.String,
    "goal": pl.String,
    "sample_policy": pl.String,
    "search_policy": pl.String,
    "branching": pl.Int64,
    "max_attempts": pl.Int64,
    "max_pair_time": pl.Float64,
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
    "wall_time": pl.Float64,
    "guided_stop_reason": pl.String,
    "guided_panic": pl.Boolean,
    "setup_status": pl.String,
    "root_status": pl.String,
    "attempt_peak_rss": pl.Int64,
    "attempt_peak_rss_max": pl.Int64,
    "sample_peak_rss": pl.Int64,
    "guided_peak_rss": pl.Int64,
    "guided_peak_live_heap": pl.Int64,
}

# An out_of_memory baseline leaves all measurement fields None.
UNGUIDED_SCHEMA = {
    "start": pl.String,
    "goal": pl.String,
    "unguided_success": pl.Boolean,
    "unguided_stop_reason": pl.String,
    "unguided_panic": pl.Boolean,
    "unguided_final_live_heap": pl.Int64,
    "unguided_peak_live_heap": pl.Int64,
    "unguided_peak_rss": pl.Int64,
}

# What an expansion contributes to a row when its `sample` process never got
# far enough to report.
EMPTY_SAMPLE_META = dict.fromkeys(
    [
        "iters",
        "nodes",
        "classes",
        "total_time",
        "final_live_heap",
        "peak_live_heap",
        "stop_reason",
        "peak_rss",
    ]
)
