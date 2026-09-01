"""Drive the guide search from Python.

This driver reads the start/goal pairs in ``problems.json`` (written by
``generate_problems.py``), constructs a guide-candidate menu per start term, and
verifies one attempt loop per pair.

Each leg is a separate ``verify`` process running one eqsat, so its peak RSS
covers the same unit of work as an unguided baseline process.
The loop early-stops on the first reach.

Guide replay and leg search share the required ``--stop-*`` budget. Dimensions
without an override retain their search-phase limits from ``problem_args.json``.

Example:
    cargo build --release --bin candidates --bin verify
    uv run scripts/guided_search.py data/problems/dusky-cramp \\
        --stop-memory 4G --attempts 5 \\
        --policy count --full-union

Pass ``--sampling-rss-max`` to hold each ``candidates`` process to a cgroup RSS
cap, retrying a killed replay at the last iteration that completed and remove
one more iter off each further attempt (``--sampling-retries``).
"""

import json
import os
import random
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict
from tqdm import tqdm

from common import (
    MeasuredJson,
    MemoryKilled,
    eqsat_finished,
    eqsat_limits,
    exit_if_missing,
    limit_flags,
    parse_size,
    run_json_subprocess,
    verify_summary,
)

# Runs one `candidates` command; `None` when it could not be kept under its cap.
RunCandidates = Callable[[list[str], str], MeasuredJson | None]

# TODO: the `smallest_novel`/`smallest_overall` policies are gone until the
# matching `Policy` variants are uncommented in src/cli.rs.
Policy = Literal["count", "uniform"]

# Leg fields derived from `verify`'s payload, with the polars dtype to pin for
# each. An unreached or panicked leg leaves most of them None, and an
# unreached-heavy prefix would otherwise make polars infer Null and reject the
# first real value. Pinning the dtypes makes the schema independent of row order
# (and of runs that reach nothing). `verify_peak_rss_bytes` comes from the
# `Measured` envelope around the payload rather than from the payload itself.
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
ATTEMPT_SCHEMA = {
    "start_term": pl.String,
    "goal_term": pl.String,
    "policy": pl.String,
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


class Args(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True, cli_kebab_case=True, cli_implicit_flags=True
    )

    # I/O
    path: CliPositionalArg[Path] = Field(
        description=(
            "Problem folder with `problems.json` and `problem_args.json` "
            "(both written by `generate_problems.py`)."
        )
    )

    output: Path | None = Field(
        default=None,
        description=(
            "Run folder for `results.parquet`/`results.json`. Auto-created under "
            "`data/guided_search/` if omitted."
        ),
    )

    candidates_binary: Path = Field(
        default=Path("target/release/candidates"),
        description="Path to the candidate-construction binary.",
    )

    verify_binary: Path = Field(
        default=Path("target/release/verify"), description="Path to the verification binary."
    )

    # Guide-replay budget
    #
    # At least one must be given. Replay ends when the first configured budget
    # is exhausted; omitted budgets are effectively unlimited.
    stop_iters: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Guide-replay iteration budget. Replay stops once this many "
            "iterations have been reached."
        ),
    )

    stop_nodes: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Guide-replay egraph-node budget. Replay stops once this many "
            "egraph nodes have been reached."
        ),
    )

    stop_time: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Guide-replay wall-clock budget in seconds. Replay stops once this "
            "time limit is reached."
        ),
    )

    stop_memory: str | None = Field(
        default=None,
        description=(
            "Guide-replay absolute process live-heap ceiling, expressed as a "
            "jemalloc `stats.allocated` limit such as `4G`. The limit is enforced "
            "directly against the process live heap with nothing subtracted out."
        ),
    )

    # Search policy
    attempts: int = Field(
        default=5,
        gt=0,
        description=(
            "Number of legs to try per (start, goal) pair, each using a freshly "
            "drawn guide. The first try counts, so `attempts=1` means one leg "
            "with no resampling. Stops early on the first reach and gives up "
            "after the final attempt."
        ),
    )

    sampling_rss_max: str | None = Field(
        default=None,
        description=(
            "Cap each `candidates` process at this cgroup RSS limit, as a human "
            "size such as `4G`. A process killed during guide replay is retried, "
            "still capped, with `--max-iters` cut to the iterations that "
            "completed. Uncapped if omitted."
        ),
    )

    sampling_retries: int = Field(
        default=1,
        ge=0,
        description=(
            "How often a `candidates` process killed at `--sampling-rss-max` is "
            "retried. The first retry runs at the iterations that completed, and "
            "each further run reduces `--max-iters` by one more. `0` disables "
            "retrying. Ignored without `--sampling-rss-max`."
        ),
    )

    policy: Policy = Field(default="count", description="Candidate-pool sampling starteg.")

    full_union: bool = Field(
        default=True,
        description="Use the experimental full-union add for the leg egraph.",
    )

    start_terms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only process the first N start terms in sorted order, making the "
            "cutoff stable across runs. All start terms are processed if omitted."
        ),
    )

    goal_terms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only use the first N goals per start term in file order, making the "
            "cutoff stable across runs. All goals are used if omitted."
        ),
    )

    seed: int = Field(
        default=0, description="RNG seed used by both the Python and Rust components."
    )

    jobs: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum number of concurrent `verify` legs, with one start/goal "
            "pair per worker. Each pair's attempt loop remains sequential, so "
            "parallelism is across pairs. Defaults to `os.cpu_count()`. Lower "
            "this if large leg egraphs exhaust available RAM."
        ),
    )

    @model_validator(mode="after")
    def validate_args(self) -> Args:
        if (
            self.stop_iters is None
            and self.stop_nodes is None
            and self.stop_time is None
            and self.stop_memory is None
        ):
            raise ValueError(
                "at least one of stop_iters, stop_nodes, stop_time, or "
                "stop_memory must be specified"
            )

        if (
            self.sampling_rss_max is not None
            and self.stop_memory is not None
            and parse_size(self.sampling_rss_max) < parse_size(self.stop_memory)
        ):
            raise ValueError(
                f"sampling_rss_max ({self.sampling_rss_max}) is below "
                f"stop_memory ({self.stop_memory})"
            )

        return self


@dataclass
class WorkItem:
    """One start/goal unit, its guide pool, and start-level guide metadata."""

    start_term: str
    goal_term: str
    pool: list
    guide_meta: dict


def replay_limits(args: Args, cfg: dict) -> dict:
    """Build the guide-replay limits for `candidates`"""
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
    start_term: str
    goal_terms: list[str]


def flatten_problems(args: Args) -> list[StartSpec]:
    """Group `problems.json`'s pair rows into per-start term specs.

    Start terms are sorted and goals keep file order, so both cutoffs are stable
    across runs.
    """
    rows = json.loads((args.path / "problems.json").read_text())
    goals: dict[str, list[str]] = {}
    for row in rows:
        goals.setdefault(row["start_term"], []).append(row["goal_term"])
    specs = [StartSpec(start, goals[start][: args.goal_terms]) for start in sorted(goals)]
    return specs[: args.start_terms]


def build_candidate_shard(
    args: Args,
    base_flags: list[str],
    limits: dict,
    menu_size: int,
    spec: StartSpec,
    run: RunCandidates,
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

    # print(f"CMD: {' '.join(cmd)}")

    measured = run(cmd, f"candidates for start term {spec.start_term!r}")
    if measured is None or not measured.payload:
        return [
            {
                "start_term": spec.start_term,
                "goal_terms": spec.goal_terms,
                "candidates": {},
                "candidate_status": "failed",
                # A capped-out child never printed its `Measured` envelope.
                "candidate_peak_rss_bytes": None if measured is None else measured.peak_rss_bytes,
            }
        ]
    records = measured.payload
    for record in records:
        record["goal_terms"] = spec.goal_terms
        record["candidate_status"] = "ok"
        record["candidate_peak_rss_bytes"] = measured.peak_rss_bytes
        # print(
        #     f"!!! sampling_peak_rss_bytes = {measured.peak_rss_bytes} | Start = {spec.start_term}"
        # )
    return records


def with_max_iters(cmd: list[str], max_iters: int) -> list[str]:
    """Copy `cmd` with its `--max-iters` value replaced."""
    out = list(cmd)
    out[out.index("--max-iters") + 1] = str(max_iters)
    return out


def build_candidate_manifest(args: Args, cfg: dict, candidate_out: Path) -> Path:
    """Construct guide menus in parallel, one `candidates` subprocess per start term.

    Merge results in start terms order and write ``candidates.json``.
    """
    specs = flatten_problems(args)
    candidate_out.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or os.cpu_count() or 1
    candidate_flags = [
        "--language",
        str(cfg["language"]),
        "--seed",
        str(args.seed),
        "--policy",
        str(args.policy),
    ]
    limits = replay_limits(args, cfg)
    cap = parse_size(args.sampling_rss_max) if args.sampling_rss_max is not None else None
    # The ceiling can also come from `problem_args.json`, which the arg
    # validator cannot see.
    heap_ceiling = limits.get("max_memory")
    if cap is not None and heap_ceiling is not None and cap < heap_ceiling:
        print(
            f"WARNING: sampling RSS cap {cap} is below the effective live-heap "
            f"ceiling {heap_ceiling}; replays get killed before it trips.",
            file=sys.stderr,
        )

    def run_plain(cmd: list[str], what: str) -> MeasuredJson | None:
        return run_json_subprocess(cmd, what=what)

    def run_capped(cmd: list[str], what: str) -> MeasuredJson | None:
        """Run under the RSS cap, retrying a replay-phase kill up to
        `--sampling-retries` times: the first retry replays the iterations that
        survived, each further one gives up another iteration."""
        cmd = [*cmd, "--print-success-iters"]
        attempt = cmd
        iters: int | None = None
        for retries_left in range(args.sampling_retries, -1, -1):
            try:
                # print(f"CMD: {' '.join(attempt)}")
                return run_json_subprocess(attempt, what=what, rss_max_bytes=cap)
            except MemoryKilled as killed:
                if eqsat_finished(killed.stderr):
                    print(f"CMD: {' '.join(attempt)}")
                    print(f"{what}: killed at RSS cap after replay, no retry", file=sys.stderr)
                    return None
                if not retries_left:
                    at = "" if iters is None else f" at --max-iters {iters}"
                    print(f"{what}: killed at RSS cap{at}, no retries left", file=sys.stderr)
                    return None
                if iters is None:
                    iters = killed.last_iter
                    if not iters:
                        print(f"{what}: killed at RSS cap before any iteration", file=sys.stderr)
                        return None
                else:
                    iters -= 1
                    if iters < 1:
                        print(
                            f"{what}: killed at RSS cap, no iterations left to cut", file=sys.stderr
                        )
                        return None
                print(
                    f"{what}: killed at RSS cap, retrying at --max-iters {iters}", file=sys.stderr
                )
                attempt = with_max_iters(cmd, iters)
        return None

    run = run_plain if cap is None else run_capped

    # Menu size = exactly what the attempt loop consumes: one guide per attempt.
    print(
        f"Constructing guide-candidate menu ({args.attempts}/policy, policy={args.policy}) "
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
                args.attempts,
                spec,
                run,
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


def build_attempt_guides(pool: list, attempts: int, rng: random.Random) -> list:
    """Pick each attempt's guide for one start/goal pair; `verify` unions one."""
    if not pool:
        return []

    sequence: list = []
    while len(sequence) < attempts:
        pass_pool = pool[:]
        rng.shuffle(pass_pool)
        sequence.extend(pass_pool)
    return sequence[:attempts]


def run_leg(
    args: Args,
    base_flags: list[str],
    goal: str,
    guide: list,
) -> tuple[dict, int]:
    """Verify one leg in its own process and return its summary and peak RSS."""
    cmd = [
        str(args.verify_binary),
        *base_flags,
        "--goal-term",
        goal,
        "--is-guide",
        "--start-term",
        json.dumps(guide),
    ]
    if args.full_union:
        cmd.append("--full-union")
    measured = run_json_subprocess(cmd, what=f"verify for goal {goal!r}")
    return verify_summary(measured.payload), measured.peak_rss_bytes


def run_pair(args: Args, base_flags: list[str], item: WorkItem) -> list[dict]:
    """Run one start/goal pair's attempt loop and return its result rows.

    Each attempt is a separate `verify` process, so `verify_peak_rss_bytes` is
    that leg's own peak rather than a high-water mark shared across the pair.
    The loop stops on the first reach.

    The final row is marked ``gave_up`` everything failed.
    An empty pool is an error.
    """
    rng = random.Random(f"{args.seed}:{item.start_term}:{item.goal_term}")
    guides = build_attempt_guides(item.pool, args.attempts, rng)
    if not guides:
        raise RuntimeError(
            f"empty candidate pool for start term {item.start_term!r} goal {item.goal_term!r}: "
            f"policy {args.policy!r} drew no guides"
        )

    rows: list[dict] = []
    ran_every_attempt = True
    for attempt, guide in enumerate(guides):
        summary, leg_peak_rss_bytes = run_leg(args, base_flags, item.goal_term, guide)
        rows.append(
            {
                "start_term": item.start_term,
                "goal_term": item.goal_term,
                "policy": args.policy,
                "attempt": attempt,
                "gave_up": False,
                "verify_peak_rss_bytes": leg_peak_rss_bytes,
                **summary,
                **item.guide_meta,
            }
        )
        if summary["reached"]:
            ran_every_attempt = False
            break

    if rows and ran_every_attempt and not rows[-1]["reached"]:
        rows[-1]["gave_up"] = True
    return rows


def resolve_output_dir(args: Args) -> Path:
    """Resolve (+ create) folder, auto-numbering `run.N` if unset."""
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

    Each start/goal pair stops early if it reaches the goal with a guide, the pairs
    run in parallel
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


def warn_pool_shortfall(items: list[WorkItem], policy: str, attempts: int) -> None:
    """Warn once per pool size when we must reuse guides.

    Deduped on pool size since the shortfall is identical across pairs.
    """

    for pool_size in sorted({len(item.pool) for item in items}):
        if attempts > pool_size:
            print(
                f"WARNING: attempts={attempts} needs that many guides but pool has "
                f"{pool_size}; reshuffling and reusing candidates across attempts "
                f"(excess {attempts - pool_size}).",
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
    summary = verify_summary(measured.payload)
    return {
        "start_term": pair["start_term"],
        "goal_term": pair["goal_term"],
        "unguided_success": summary["reached"],
        "unguided_stop_reason": summary["stop_reason"],
        "unguided_panic": summary["panic"],
        "unguided_final_live_heap_bytes": summary["memory"],
        "unguided_peak_live_heap_bytes": summary["peak_live_heap"],
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

        # Each attempt is now its own process, so pick which leg's peak to
        # report rather than inheriting a shared high-water mark.
        #
        # `verify_peak_rss_bytes` is the *decisive* leg: the one that reached, or
        # the last one tried if none did.
        # `verify_peak_rss_bytes_max` is the max across every leg run, which is
        # what the pair cost end to end.
        decisive = successes[0] if successes else (attempts[-1] if attempts else None)
        verify_peak = decisive["verify_peak_rss_bytes"] if decisive else None
        leg_peaks = [
            row["verify_peak_rss_bytes"]
            for row in attempts
            if row.get("verify_peak_rss_bytes") is not None
        ]
        verify_peak_max = max(leg_peaks) if leg_peaks else None

        rss_peaks = [peak for peak in (candidate_peak, verify_peak_max) if peak is not None]
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
                "policy": args.policy,
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
                "verify_peak_rss_bytes_max": verify_peak_max,
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
        **args.model_dump(),
        "effective_limits": limits,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    reached_pairs = int(pairs["guided_success"].sum())
    total_pairs = len(pairs)
    reach_rate = reached_pairs / total_pairs if total_pairs else 0.0
    print(
        f"\nReached {reached_pairs}/{total_pairs} start/goal pairs "
        f"(reach rate {reach_rate:.2f}). "
        f"Wrote {out / 'comparison.parquet'}",
        file=sys.stderr,
    )


def main() -> int:
    args = Args()

    if all(v is None for v in (args.stop_iters, args.stop_nodes, args.stop_time, args.stop_memory)):
        print(
            "No guide-replay budget given; pass at least one of "
            "--stop-iters/--stop-nodes/--stop-time/--stop-memory.",
            file=sys.stderr,
        )
        return 2
    exit_if_missing(args.candidates_binary, args.verify_binary)

    cfg = json.loads((args.path / "problem_args.json").read_text())
    limits = replay_limits(args, cfg)
    base_flags = ["--language", str(cfg["language"]), *limit_flags(limits)]
    out = resolve_output_dir(args)

    candidates_path = build_candidate_manifest(args, cfg, out / "candidate_run")
    start_records = json.loads(candidates_path.read_text())
    items = build_work_items(start_records)
    warn_pool_shortfall(items, args.policy, args.attempts)

    rows = run_all_pairs(args, base_flags, items)

    # The baseline is re-run here, not reused from `problems.json`: the search
    # budget is the `--stop-*` one, not the budget generation measured under.
    unguided_rows = run_all_unguided(args, base_flags, expected_pairs(start_records))

    report_results(args, out, start_records, rows, unguided_rows, limits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
