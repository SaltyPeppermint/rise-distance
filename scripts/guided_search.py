"""Drive the guide search from Python.

This driver reads enriched starts, constructs a guide-candidate menu per starts,
and verifies one attempt loop per start/goal pair. It owns the JSON/parquet I/O; the Rust
``candidates`` and ``verify`` binaries communicate through argv, stdin, and stdout.

Guide replay and leg search share the required ``--stop-*`` budget. Dimensions
without an override retain their search-phase limits from ``goal_args.json``.

Example:
    cargo build --release --bin candidates --bin verify
    uv run scripts/guided_search.py data/start_terms/dusky-cramp \\
        --stop-memory 4G \\
        --attempts 5 --k 10 \\
        --strategy no_replacement_count --full-union
"""

import dataclasses
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import polars as pl
import tyro
from tqdm import tqdm

from common import (
    eqsat_limits,
    exit_if_missing,
    limit_flags,
    parse_size,
    run_json_subprocess,
)

# Candidate-pool names emitted by `candidates` (see CandidatePool::name in src/cli.rs).
# The `with_replacement_*` pools are re-drawn *with* replacement across a leg's
# `k` picks; everything else is drawn without replacement.
CandidateStrategy = Literal[
    "no_replacement_count",
    "no_replacement_uniform",
    "with_replacement_count",
    "with_replacement_uniform",
]
SmallestStrategy = Literal["smallest_novel", "smallest_overall"]
Strategy = Literal[CandidateStrategy, SmallestStrategy]
SINGLE_CANDIDATE_STRATEGIES = get_args(SmallestStrategy)
Policies = Literal[
    "count",
    "uniform",
    "smallest_novel",
    "smallest_overall",
]

# Fields copied straight out of a `verify` LegResult onto a result row, with the
# polars dtype to pin for each. `verify` omits these on an unreached leg
# (`skip_serializing_if`), so they land as None; an unreached-heavy prefix would
# otherwise make polars infer Null and reject the first real value. Pinning the
# dtypes makes the schema count of row order (and of runs that reach
# nothing).
LEG_RESULT_DTYPES = {
    "iters": pl.Int64,
    "nodes": pl.Int64,
    "classes": pl.Int64,
    "total_applied": pl.Int64,
    "total_time": pl.Float64,
    "memory": pl.Int64,
    "peak_live_heap": pl.Int64,
    "verify_peak_rss_bytes": pl.Int64,
    "stop_reason": pl.String,
}
# `verify_peak_rss_bytes` is Python-side subprocess telemetry, not a field in
# Rust's `LegResult`. Keep its dtype in the output schema, but never look it up
# in the Rust payload: doing so would overwrite the measured value with None.
LEG_RESULT_FIELDS = tuple(field for field in LEG_RESULT_DTYPES if field != "verify_peak_rss_bytes")
ATTEMPT_SCHEMA = {
    "start_term": pl.String,
    "goal_term": pl.String,
    "strategy": pl.String,
    "k": pl.Int64,
    "attempt": pl.Int64,
    "reached": pl.Boolean,
    "gave_up": pl.Boolean,
    "panic": pl.Boolean,
    **LEG_RESULT_DTYPES,
    "guide_nodes": pl.Int64,
    "guide_classes": pl.Int64,
    "guide_time": pl.Float64,
    "guide_memory": pl.Int64,
    "guide_peak_live_heap": pl.Int64,
    "candidate_peak_rss_bytes": pl.Int64,
    "guide_stop_reason": pl.String,
}


@dataclass
class Args:
    # I/O
    path: tyro.conf.Positional[Path]
    """Start Term folder with `goal_terms.json` and `goal_args.json` (both written
    by `generate_goals.py`)."""

    output: Path | None = None
    """Run folder for `results.parquet`/`results.json`. Auto-created under
    `data/guided_search/` if omitted."""

    candidates_input: Path | None = None
    """Existing `candidates.json` candidate manifest to reuse. When set, skip
    guide replay and candidate construction. This permits paired strategy runs over
    exactly the same per-start term menus."""

    unguided_input: Path | None = None
    """Existing `unguided_results.parquet` to reuse. The pair universe and
    effective limits must match this run."""

    candidates_binary: Path = Path("target/release/candidates")
    verify_binary: Path = Path("target/release/verify")

    # guide-replay budget: at least one must be given; the replay ends at
    # whichever given budget trips first, unset ones are effectively unlimited.
    stop_iters: int | None = None
    """Guide-replay iteration budget."""
    stop_nodes: int | None = None
    """Guide-replay egraph-node budget."""
    stop_time: float | None = None
    """Guide-replay wall-clock budget in seconds."""
    stop_memory: str | None = None
    """Guide-replay absolute process live-heap ceiling (jemalloc
    `stats.allocated`, e.g. `4G`), enforced directly against the process live
    heap with nothing subtracted out."""

    # search policy
    attempts: int = 5
    """How many legs to try per (start, goal) pair, each with a freshly resampled
    guide subset. Counts the first try, so `attempts=1` means a single leg with
    no resampling. Stops early on the first reach; gives up after the last."""
    strategy: Strategy = "no_replacement_count"
    """Which candidate pool to restart with."""
    k: int = 1
    """Guide-set size: each start/goal pair runs one attempt loop drawing `k`
    guides per leg. Forced to `1` for the `smallest_*` strategies."""
    full_union: bool = True
    """Use the experimental full-union add for the leg egraph."""

    start_terms: int | None = None
    """Only process the first N start terms (sorted order, so stable across
    runs). All start terms if omitted."""

    goal_terms: int | None = None
    """Only use the first N goals per start term (file order, so stable across
    runs). All goals if omitted."""

    seed: int = 0
    """RNG seed. Used in Python and Rust"""

    size_goal: int = 5
    """Number of novel sizes the analysis must find."""

    jobs: int | None = None
    """Max concurrent `verify` legs (one start/goal pair per worker). Each pair's
    attempt loop stays sequential, so this parallelises across pairs. Defaults to
    `os.cpu_count()`. Lower it if the large leg egraphs exhaust RAM."""


@dataclass
class WorkItem:
    """One start/goal unit, its guide pool, and start-level guide metadata."""

    start_term: str
    goal: str
    pool: list
    guide_meta: dict


def pool_key(strategy: Strategy) -> Policies:
    """Map a driver strategy to the candidate-pool key `candidates` writes.

    The replacement prefix is a Python-side draw policy (`pick_subset`), not a
    pool: `candidates` emits `count`, `uniform`, so collapse the prefix to hit one of those.
    `smallest_*` keys pass through unchanged.
    """
    if strategy in SINGLE_CANDIDATE_STRATEGIES:
        return strategy
    if strategy.endswith("_count"):
        return "count"
    if strategy.endswith("_uniform"):
        return "uniform"
    raise ValueError(f"unknown strategy {strategy!r}")


def replay_limits(args: Args, cfg: dict) -> dict:
    """Build the guide-replay limits for `candidates`: the search-phase
    (brute-force) limits from `goal_args.json`, with each given `--stop-*`
    budget overriding its dimension. The replay ends at whichever limit trips
    first.
    """
    limits = eqsat_limits(cfg)
    if args.stop_iters is not None:
        limits["max_iters"] = args.stop_iters
    if args.stop_nodes is not None:
        limits["max_nodes"] = args.stop_nodes
    if args.stop_time is not None:
        limits["max_time"] = args.stop_time
    if args.stop_memory is not None:
        limits["max_memory"] = parse_size(args.stop_memory)
    return limits


@dataclass
class StartSpec:
    """One start terms's `candidates` input (`start_term`) and its goals, merged back onto the
    output record Python-side."""

    start_term: str
    goal_terms: list[str]


def flatten_enriched_start_terms(args: Args) -> list[StartSpec]:
    """Flatten the goal-enriched `goal_terms.json` into per-start term specs.

    Drop ``Err`` payloads, sort terms within file-ordered groups, and apply the
    optional start term and per-start goal limits.
    """
    groups = json.loads((args.path / "goal_terms.json").read_text())
    specs: list[StartSpec] = []
    for _size, terms_map in groups:
        for start in sorted(terms_map):
            ok = terms_map[start].get("Ok")
            if ok is None:
                continue  # Err start: goal stage failed, nothing to replay.
            goals = ok["goals"]
            if args.goal_terms is not None:
                goals = goals[: args.goal_terms]
            specs.append(StartSpec(start, goals))
    if args.start_terms is not None:
        specs = specs[: args.start_terms]
    return specs


def build_candidate_shard(
    args: Args,
    base_flags: list[str],
    limits: dict,
    menu_size: int,
    spec: StartSpec,
) -> list[dict]:
    """Run ``candidates`` for one start term and attach its goals to the result."""
    cmd = [
        str(args.candidates_binary),
        *base_flags,
        *limit_flags(limits),
        "--start-term",
        spec.start_term,
        "--n-candidates",
        str(menu_size),
    ]

    measured = run_json_subprocess(cmd, what=f"candidates for start term {spec.start_term!r}")
    records = measured.payload
    if not records:
        return [
            {
                "start_term": spec.start_term,
                "goal_terms": spec.goal_terms,
                "candidates": {},
                "candidate_status": "failed",
                "candidate_peak_rss_bytes": measured.peak_rss_bytes,
            }
        ]
    for record in records:
        record["goal_terms"] = spec.goal_terms
        record["candidate_status"] = "ok"
        record["candidate_peak_rss_bytes"] = measured.peak_rss_bytes
        print(f"!!! sampling_peak_rss_bytes {measured.peak_rss_bytes}")
    return records


def build_candidate_manifest(args: Args, cfg: dict, candidate_out: Path) -> Path:
    """Construct guide menus in parallel, one `candidates` subprocess per start term.

    Merge results in start terms order and write ``candidates.json`` for provenance.
    """
    specs = flatten_enriched_start_terms(args)
    policy = pool_key(args.strategy)
    candidate_out.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or os.cpu_count() or 1
    candidate_flags = [
        "--language",
        str(cfg["language"]),
        "--seed",
        str(args.seed),
        "--size-goal",
        str(args.size_goal),
        "--policy",
        str(policy),
    ]
    limits = replay_limits(args, cfg)
    # Menu size = exactly what the attempt loop consumes: no_replacement_* needs
    # k distinct guides per attempt across `attempts` disjoint attempts.
    menu_size = args.k * args.attempts
    print(
        f"Constructing guide-candidate menu ({menu_size}/strategy, policy={policy}) "
        f"for {len(specs)} start terms(s) "
        f"-> {candidate_out} ({jobs} workers)",
        file=sys.stderr,
    )

    shard_records: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool_exec:
        futures = {
            pool_exec.submit(
                build_candidate_shard,
                args,
                candidate_flags,
                limits,
                menu_size,
                spec,
            ): i
            for i, spec in enumerate(specs)
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="candidates",
            unit="start term",
        ):
            shard_records[futures[fut]] = fut.result()

    merged = [rec for i in range(len(specs)) for rec in shard_records[i]]
    merged_path = candidate_out / "candidates.json"
    merged_path.write_text(json.dumps(merged))
    return merged_path


def build_attempt_subsets(
    pool: list, strategy: str, k: int, attempts: int, rng: random.Random
) -> list[list]:
    """Prepare each attempt's guide subset for one start/goal pair.

    `smallest_*` yields a single term per attempt; `with_replacement_*` makes `k`
    picks with replacement per attempt.

    `no_replacement_*` slices consecutive `eff_k`-sized legs (`eff_k = min(k, pool)`)
    out of a shuffled pool, so the attempts are disjoint until the pool is
    exhausted (`attempts * eff_k > len(pool)`), where a fresh pass is appended.
    """
    if strategy in SINGLE_CANDIDATE_STRATEGIES:
        return [pool[:1] for _ in range(attempts)]
    if not pool:
        return [[] for _ in range(attempts)]
    if strategy.startswith("with_replacement"):
        return [[rng.choice(pool) for _ in range(k)] for _ in range(attempts)]

    # no_replacement_*: partition one (or more) shuffled passes into eff_k slices so
    # attempts stay disjoint until the pool is exhausted.
    eff_k = min(k, len(pool))
    needed = attempts * eff_k
    sequence: list = []
    while len(sequence) < needed:
        pass_pool = pool[:]
        rng.shuffle(pass_pool)
        sequence.extend(pass_pool)
    return [sequence[a * eff_k : (a + 1) * eff_k] for a in range(attempts)]


def run_legs(
    args: Args,
    base_flags: list[str],
    goal: str,
    subsets: list[list],
) -> tuple[list[dict], int]:
    """Verify one start/goal pair and return the legs run before early stopping.

    Attempt subsets are sent as JSON on stdin. Panic-guarded legs still produce
    results; process-level failures raise.
    """
    cmd = [str(args.verify_binary), *base_flags, "--goal-term", goal]
    if args.full_union:
        cmd.append("--full-union")
    measured = run_json_subprocess(cmd, what=f"verify for goal {goal!r}", input=json.dumps(subsets))
    print(f"!!! guided_attempt_peak_rss_bytes {measured.peak_rss_bytes}")
    return measured.payload, measured.peak_rss_bytes


def run_pair(args: Args, base_flags: list[str], item: WorkItem) -> list[dict]:
    """Run one start/goal pair's attempt loop and return its result rows.

    The final row is marked ``gave_up`` only if all attempts ran and failed.
    Reported ``k`` is the effective subset size. An empty pool is an error.
    """
    rng = random.Random(f"{args.seed}:{item.start_term}:{item.goal}")
    attempt_subsets = build_attempt_subsets(item.pool, args.strategy, args.k, args.attempts, rng)
    if any(not guides for guides in attempt_subsets):
        raise RuntimeError(
            f"empty candidate pool for start term {item.start_term!r} goal {item.goal!r}: "
            f"strategy {args.strategy!r} drew no guides"
        )

    results, verify_peak_rss_bytes = run_legs(args, base_flags, item.goal, attempt_subsets)
    rows: list[dict] = []
    for attempt, (guides, result) in enumerate(zip(attempt_subsets, results)):
        row = {
            "start_term": item.start_term,
            "goal_term": item.goal,
            "strategy": args.strategy,
            "k": len(guides),
            "attempt": attempt,
            "reached": result["reached"],
            "gave_up": False,
            "panic": result.get("panic", False),
            "verify_peak_rss_bytes": verify_peak_rss_bytes,
            **{field: result.get(field) for field in LEG_RESULT_FIELDS},
            **item.guide_meta,
        }
        rows.append(row)

    # Fewer results than subsets means `verify` early-stopped on a reach, so the
    # last row is a reach, not a give-up. Only mark give-up when every attempt
    # ran without reaching.
    if rows and len(results) == len(attempt_subsets) and not rows[-1]["reached"]:
        rows[-1]["gave_up"] = True
    return rows


def resolve_output_dir(args: Args) -> Path:
    """Resolve (and create) the run folder, auto-numbering `run.N` if unset."""
    out = args.output
    if out is None:
        base = Path("data/guided_search")
        base.mkdir(parents=True, exist_ok=True)
        existing = [int(p.suffix[1:]) for p in base.glob("run.*") if p.suffix[1:].isdigit()]
        out = base / f"run.{max(existing, default=0) + 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_work_items(start_term_records: list) -> list[WorkItem]:
    """Flatten `candidates`'s start term records into count (start, goal) items.

    Each item runs its own sequential attempt loop (early-stops on first reach);
    items run concurrently.
    """
    items: list[WorkItem] = []
    for record in start_term_records:
        if record.get("candidate_status") != "ok":
            continue
        pool = record["candidates"]
        guide_meta = {
            "guide_nodes": record["guide_nodes"],
            "guide_classes": record["guide_classes"],
            "guide_time": record["guide_time"],
            "guide_memory": record["guide_memory"],
            "guide_peak_live_heap": record["guide_peak_live_heap"],
            "candidate_peak_rss_bytes": record["candidate_peak_rss_bytes"],
            "guide_stop_reason": record["stop_reason"],
        }
        for goal in record["goal_terms"]:
            items.append(WorkItem(record["start_term"], goal, pool, guide_meta))
    return items


def warn_pool_shortfall(items: list[WorkItem], strategy: str, k: int, attempts: int) -> None:
    """Warn once per pool size when no_replacement_* must cap or reuse.

    Deduped on pool size since the shortfall is identical across pairs.
    """
    if not strategy.startswith("no_replacement"):
        return
    for pool_size in sorted({len(item.pool) for item in items}):
        eff_k = min(k, pool_size)
        if eff_k < k:
            print(
                f"WARNING: k={k} exceeds pool size {pool_size}; capping each leg "
                f"to {eff_k} distinct guides.",
                file=sys.stderr,
            )
        if attempts * eff_k > pool_size:
            print(
                f"WARNING: k={eff_k} x attempts={attempts} needs "
                f"{attempts * eff_k} guides but pool has {pool_size}; reshuffling "
                f"and reusing candidates across attempts "
                f"(excess {attempts * eff_k - pool_size}).",
                file=sys.stderr,
            )


def run_all_pairs(args: Args, base_flags: list[str], items: list[WorkItem]) -> list[dict]:
    """Run every work item's attempt loop concurrently and collect result rows."""
    print(f"Running legs for {len(items)} (start, goal) item(s)", file=sys.stderr)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs or os.cpu_count() or 1) as pool_exec:
        futures = [pool_exec.submit(run_pair, args, base_flags, item) for item in items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="legs", unit="pair"):
            rows.extend(fut.result())
    return rows


def expected_pairs(start_records: list[dict]) -> list[dict]:
    """Return every planned pair, including start terms whose construction failed."""
    return [
        {
            "start_term": record["start_term"],
            "goal_term": goal_term,
            "candidate_status": record["candidate_status"],
            "candidate_peak_rss_bytes": record["candidate_peak_rss_bytes"],
            "guide_peak_live_heap": record.get("guide_peak_live_heap"),
            "guide_stop_reason": record.get("stop_reason"),
        }
        for record in start_records
        for goal_term in record["goal_terms"]
    ]


def run_unguided_pair(args: Args, base_flags: list[str], pair: dict) -> dict:
    """Run the pair-matched single-start baseline with predictive stopping."""
    cmd = [
        str(args.verify_binary),
        *base_flags,
        "--start-term",
        pair["start_term"],
        "--goal-term",
        pair["goal_term"],
    ]
    measured = run_json_subprocess(cmd, what=f"unguided verify for goal term {pair['goal_term']!r}")
    result = measured.payload[0]
    print(f"!!! unguided_brute_force_peak_rss_bytes {measured.peak_rss_bytes}")
    return {
        "start_term": pair["start_term"],
        "goal_term": pair["goal_term"],
        "unguided_success": result["reached"],
        "unguided_stop_reason": result.get("stop_reason"),
        "unguided_panic": result.get("panic", False),
        "unguided_final_live_heap_bytes": result.get("memory"),
        "unguided_peak_live_heap_bytes": result.get("peak_live_heap"),
        "unguided_peak_rss_bytes": measured.peak_rss_bytes,
    }


def run_all_unguided(args: Args, base_flags: list[str], pairs: list[dict]) -> list[dict]:
    print(f"Running {len(pairs)} pair-matched unguided baseline(s)", file=sys.stderr)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs or os.cpu_count() or 1) as pool_exec:
        futures = [pool_exec.submit(run_unguided_pair, args, base_flags, pair) for pair in pairs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="unguided", unit="pair"):
            rows.append(future.result())
    return rows


def summarize_pairs(
    args: Args,
    start_records: list[dict],
    rows: list[dict],
) -> list[dict]:
    """Collapse attempt rows to one guided workflow row per planned pair."""
    attempts_by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        attempts_by_pair.setdefault((row["start_term"], row["goal_term"]), []).append(row)

    summary = []
    for pair in expected_pairs(start_records):
        attempts = sorted(
            attempts_by_pair.get((pair["start_term"], pair["goal_term"]), []),
            key=lambda row: row["attempt"],
        )
        successes = [row for row in attempts if row["reached"]]
        candidate_peak = pair["candidate_peak_rss_bytes"]
        verify_peak = attempts[0]["verify_peak_rss_bytes"] if attempts else None
        rss_peaks = [peak for peak in (candidate_peak, verify_peak) if peak is not None]
        live_peaks = [
            peak
            for peak in (
                pair["guide_peak_live_heap"],
                *(row.get("peak_live_heap") for row in attempts),
            )
            if peak is not None
        ]
        setup_status = pair["candidate_status"]
        if setup_status == "ok" and not attempts:
            setup_status = "empty_pool"
        summary.append(
            {
                **pair,
                "strategy": args.strategy,
                "k": args.k,
                "attempt_budget": args.attempts,
                "guided_success": bool(successes),
                "success_attempt": successes[0]["attempt"] + 1 if successes else None,
                "attempts_run": len(attempts),
                "guided_stop_reason": None
                if successes
                else (attempts[-1].get("stop_reason") if attempts else setup_status),
                "guided_panic": any(row["panic"] for row in attempts),
                "setup_status": setup_status,
                "verify_peak_rss_bytes": verify_peak,
                "guided_peak_rss_bytes": max(rss_peaks) if rss_peaks else None,
                "guided_peak_live_heap_bytes": max(live_peaks) if live_peaks else None,
            }
        )
    return summary


def report_results(
    args: Args,
    out: Path,
    start_records: list[dict],
    rows: list[dict],
    unguided_rows: list[dict],
    limits: dict,
) -> None:
    """Write attempt, pair, baseline, and joined comparison results."""
    df = pl.DataFrame(rows, schema=ATTEMPT_SCHEMA)
    df.write_parquet(out / "results.parquet")
    (out / "results.json").write_text(json.dumps(rows, indent=2))

    pair_rows = summarize_pairs(
        args,
        start_records,
        rows,
    )
    pairs = pl.DataFrame(pair_rows)
    unguided = pl.DataFrame(unguided_rows)
    comparison = pairs.join(unguided, on=["start_term", "goal_term"], how="left", validate="1:1")
    pairs.write_parquet(out / "pair_results.parquet")
    unguided.write_parquet(out / "unguided_results.parquet")
    comparison.write_parquet(out / "comparison.parquet")
    config = {
        **dataclasses.asdict(args),
        "effective_limits": limits,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    reached_pairs = int(pairs["guided_success"].sum())
    total_pairs = len(pairs)
    reach_rate = reached_pairs / total_pairs if total_pairs else 0.0
    print(
        f"\nReached {reached_pairs}/{total_pairs} start/goal pairs "
        f"(reach rate {reach_rate:.2f}) at k={args.k}. "
        f"Wrote {out / 'comparison.parquet'}",
        file=sys.stderr,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_model_provenance(model: str | Path | None) -> dict | None:
    """Hash the deployed model and adjacent manifest for reproducibility."""
    if model is None:
        return None
    model_path = Path(model)
    manifest_path = model_path.with_suffix(".json")
    return {
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def main() -> int:
    args = tyro.cli(Args, description=__doc__)

    if all(v is None for v in (args.stop_iters, args.stop_nodes, args.stop_time, args.stop_memory)):
        print(
            "No guide-replay budget given; pass at least one of "
            "--stop-iters/--stop-nodes/--stop-time/--stop-memory.",
            file=sys.stderr,
        )
        return 2
    exit_if_missing(args.candidates_binary, args.verify_binary)

    cfg = json.loads((args.path / "goal_args.json").read_text())
    limits = replay_limits(args, cfg)
    args.k = 1 if args.strategy in SINGLE_CANDIDATE_STRATEGIES else args.k
    base_flags = ["--language", str(cfg["language"]), *limit_flags(limits)]
    out = resolve_output_dir(args)

    if args.candidates_input is None:
        candidates_path = build_candidate_manifest(args, cfg, out / "candidate_run")
    else:
        candidates_path = args.candidates_input
        if not candidates_path.is_file():
            print(f"Candidate manifest not found: {candidates_path}", file=sys.stderr)
            return 2
        print(f"Reusing candidate manifest {candidates_path}", file=sys.stderr)
    start_records = json.loads(candidates_path.read_text())
    items = build_work_items(start_records)
    warn_pool_shortfall(items, args.strategy, args.k, args.attempts)

    rows = run_all_pairs(args, base_flags, items)

    pairs = expected_pairs(start_records)
    if args.unguided_input is None:
        unguided_rows = run_all_unguided(args, base_flags, pairs)
    else:
        if not args.unguided_input.is_file():
            print(f"Unguided result file not found: {args.unguided_input}", file=sys.stderr)
            return 2
        unguided_rows = pl.read_parquet(args.unguided_input).to_dicts()
        planned_keys = {(pair["start_term"], pair["goal_term"]) for pair in pairs}
        baseline_keys = {(row["start_term"], row["goal_term"]) for row in unguided_rows}
        if planned_keys != baseline_keys:
            print(
                "Unguided result pair universe does not match this run "
                f"(planned={len(planned_keys)}, baseline={len(baseline_keys)}).",
                file=sys.stderr,
            )
            return 2

    report_results(args, out, start_records, rows, unguided_rows, limits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
