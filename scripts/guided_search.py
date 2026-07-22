"""Drive the guide search from Python.

The Rust `sample` and `verify` binaries touch no files: `sample` takes
everything on argv, `verify` takes a (seed, goal) pair's attempt subsets on
stdin plus `--goal`/`--language`/eqsat limits on argv. This driver owns all
file I/O: it reads `goal_args.json` (language and the search-phase eqsat
limits), flattens the goal-enriched `goal_terms.json` into per-seed specs, runs
one `sample` subprocess per seed, then runs one `verify` subprocess per seed/goal
pair. Python precomputes all `--attempts` guide subsets for the pair up front and
hands them over; `verify` runs one leg per subset and early-stops at the first
reach. All logging and data wrangling (parquet/JSON) live here.

The guide replay runs under an explicit budget given on the CLI: at least one
of `--stop-iters`/`--stop-nodes`/`--stop-time`/`--stop-memory` is required;
unset dimensions fall back to the search-phase (brute-force) limits from
`goal_args.json`, and the replay ends at whichever limit trips first. The point
is to ask "how much can still be solved under a tighter budget than the
brute-force full eqsat by breaking the search into guided legs". The leg search
itself keeps the search-phase limits.

Example:
    cargo build --release --bin sample --bin verify
    uv run scripts/guided_search.py data/seed_terms/dusky-cramp \\
        --stop-memory 4G \\
        --attempts 5 --k 10 \\
        --strategy no_replacement_count --full-union
"""

import dataclasses
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

# Strategy names emitted by `sample` (see Strategy::name in src/cli/sample.rs).
# The `with_replacement_*` pools are re-drawn *with* replacement across a leg's
# `k` picks; everything else is drawn without replacement.
SamplingStrategy = Literal[
    "no_replacement_count",
    "no_replacement_naive",
    "with_replacement_count",
    "with_replacement_naive",
]
SmallestStrategy = Literal["smallest_novel", "smallest_overall"]
Strategy = Literal[SamplingStrategy, SmallestStrategy]
SMALLEST_STRATEGIES = get_args(SmallestStrategy)

# Fields copied straight out of a `verify` LegResult onto a result row, with the
# polars dtype to pin for each. `verify` omits these on an unreached leg
# (`skip_serializing_if`), so they land as None; an unreached-heavy prefix would
# otherwise make polars infer Null and reject the first real value. Pinning the
# dtypes makes the schema independent of row order (and of runs that reach
# nothing).
LEG_RESULT_DTYPES = {
    "iters": pl.Int64,
    "nodes": pl.Int64,
    "classes": pl.Int64,
    "total_applied": pl.Int64,
    "total_time": pl.Float64,
    "memory": pl.Int64,
    "stop_reason": pl.String,
}
LEG_RESULT_FIELDS = tuple(LEG_RESULT_DTYPES)


@dataclass
class Args:
    # I/O
    path: tyro.conf.Positional[Path]
    """Seed folder with `goal_terms.json` and `goal_args.json` (both written
    by `generate_goals.py`)."""

    output: Path | None = None
    """Run folder for `results.parquet`/`results.json`. Auto-created under
    `data/guided_search/` if omitted."""

    sample_binary: Path = Path("target/release/sample")
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
    """Guide-replay live-heap budget (jemalloc `stats.allocated`, e.g. `4G`)."""

    # search policy
    attempts: int = 5
    """How many legs to try per (seed, goal) pair, each with a freshly resampled
    guide subset. Counts the first try, so `attempts=1` means a single leg with
    no resampling. Stops early on the first reach; gives up after the last."""
    strategy: Strategy = "no_replacement_count"
    """Which candidate pool to restart with."""
    k: int = 1
    """Guide-set size: each seed/goal pair runs one attempt loop drawing `k`
    guides per leg. Forced to `1` for the `smallest_*` strategies."""
    full_union: bool = False
    """Use the experimental full-union add for the leg egraph."""

    seeds: int | None = None
    """Only process the first N seed terms (sorted order, so stable across
    runs). All seeds if omitted."""

    goals: int | None = None
    """Only use the first N goals per seed term (file order, so stable across
    runs). All goals if omitted."""

    rng_seed: int = 0
    """RNG seed for subset selection (offset per attempt)."""

    jobs: int | None = None
    """Max concurrent `verify` legs (one seed/goal pair per worker). Each pair's
    attempt loop stays sequential, so this parallelises across pairs. Defaults to
    `os.cpu_count()`. Lower it if the large leg egraphs exhaust RAM."""


@dataclass
class WorkItem:
    """One independent (seed, goal) unit of work for the leg loop.

    Each item carries the candidate `pool` it draws guides from and the
    seed-level `guide_meta` copied onto every result row. Items are built up
    front (`build_work_items`) and run concurrently, one attempt loop each.
    """

    seed: str
    goal: str
    pool: list
    guide_meta: dict


def pool_key(strategy: str) -> str:
    """Map a driver strategy to the candidate-pool key `sample` writes.

    The replacement prefix is a Python-side draw policy (`pick_subset`), not a
    pool: `sample` only emits `sample_count` / `sample_naive`, so collapse the
    prefix to hit one of those. `smallest_*` keys pass through unchanged.
    """
    if strategy in SMALLEST_STRATEGIES:
        return strategy
    if strategy.endswith("_count"):
        return "sample_count"
    if strategy.endswith("_naive"):
        return "sample_naive"
    return strategy


def replay_limits(args: Args, cfg: dict) -> dict:
    """Build the guide-replay limits for `sample`: the search-phase
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
class SeedSpec:
    """One seed's `sample` input (`seed`) and its goals, merged back onto the
    output record Python-side."""

    seed: str
    goals: list[str]


def flatten_enriched_seeds(args: Args) -> list[SeedSpec]:
    """Flatten the goal-enriched `goal_terms.json` into per-seed specs.

    `goal_terms.json` is a list of `[size, {seed: payload}]` groups, each
    payload a `Result`-shaped `{"Ok": GoalGenMetadata}` / `{"Err": msg}`. We
    pull the seed and its goals off each `Ok`. `Err` seeds (goal stage failed)
    are dropped. Groups in file order, terms sorted for a deterministic
    `--seeds`; `--goals` keeps each seed's first N goals in file order.
    """
    groups = json.loads((args.path / "goal_terms.json").read_text())
    specs: list[SeedSpec] = []
    for _size, terms_map in groups:
        for seed in sorted(terms_map):
            ok = terms_map[seed].get("Ok")
            if ok is None:
                continue  # Err seed: goal stage failed, nothing to replay.
            goals = ok["goals"]
            if args.goals is not None:
                # First N in file order: deterministic across runs.
                goals = goals[: args.goals]
            specs.append(SeedSpec(seed, goals))
    if args.seeds is not None:
        specs = specs[: args.seeds]
    return specs


def run_sample_shard(
    args: Args, base_flags: list[str], limits: dict, menu_size: int, spec: SeedSpec
) -> list:
    """Run `sample` for a single seed and return its parsed record list, with
    the spec's `goals` spliced back onto each record.

    Everything goes on argv: `--seed`, `--language`, and `limits` (the
    `--stop-*` replay budget from `replay_limits`). `sample` prints a
    one-element `[SeedSamples]` array to stdout (empty on an internal failure).
    """
    cmd = [
        str(args.sample_binary),
        *base_flags,
        *limit_flags(limits),
        "--seed",
        spec.seed,
        "--samples-per-strategy",
        str(menu_size),
    ]
    records = run_json_subprocess(cmd, what=f"sample for seed {spec.seed!r}")
    for record in records:
        record["goals"] = spec.goals
    return records


def run_sample(args: Args, cfg: dict, sample_out: Path) -> Path:
    """Sample the guide menu in parallel, one `sample` subprocess per seed.

    Captures each subprocess's stdout, merges the records in seed order (so the
    output is deterministic regardless of completion order), and writes a single
    `samples.json` under `sample_out` for provenance.
    """
    specs = flatten_enriched_seeds(args)
    sample_out.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or os.cpu_count() or 1
    # sample gets the --stop-* replay budget; verify keeps the plain
    # search-phase limits.
    sample_flags = ["--language", str(cfg["language"])]
    limits = replay_limits(args, cfg)
    # Menu size = exactly what the attempt loop consumes: no_replacement_* needs
    # k distinct guides per attempt across `attempts` disjoint attempts.
    menu_size = args.k * args.attempts
    print(
        f"Sampling guide menu ({menu_size}/strategy) for {len(specs)} seed(s) "
        f"-> {sample_out} ({jobs} workers)",
        file=sys.stderr,
    )

    # index -> records, filled as subprocesses finish; merged in seed order below.
    shard_records: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool_exec:
        futures = {
            pool_exec.submit(run_sample_shard, args, sample_flags, limits, menu_size, spec): i
            for i, spec in enumerate(specs)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sampling", unit="seed"):
            shard_records[futures[fut]] = fut.result()

    merged = [rec for i in range(len(specs)) for rec in shard_records[i]]
    merged_path = sample_out / "samples.json"
    merged_path.write_text(json.dumps(merged))
    return merged_path


def build_attempt_subsets(
    pool: list, strategy: str, k: int, attempts: int, rng: random.Random
) -> list[list]:
    """Precompute each attempt's guide subset for one seed/goal pair.

    `smallest_*` yields a single term per attempt; `with_replacement_*` makes `k`
    picks with replacement per attempt.

    `no_replacement_*` slices consecutive `eff_k`-sized legs (`eff_k = min(k, pool)`)
    out of a shuffled pool, so the attempts are disjoint until the pool is
    exhausted (`attempts * eff_k > len(pool)`), where a fresh pass is appended.
    """
    if strategy in SMALLEST_STRATEGIES:
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
) -> list[dict]:
    """Run one `verify` subprocess for a whole (seed, goal) pair and return its
    parsed `LegResult` list.

    `verify` touches no files: the pair's attempt `subsets` (a list of guide
    subsets) go in on stdin as a JSON array of arrays, and
    `--goal`/`--language`/`--full-union`/the eqsat flags on argv. It runs one
    leg per subset, stops at the first reach, and prints one `LegResult` per leg
    actually run — so the returned list can be shorter than `subsets`. A
    panic-guarded leg still yields a result with `panic: true` and the loop
    continues; a nonzero exit means the process itself failed (bad request,
    parse error, ...) and raises.
    """
    cmd = [str(args.verify_binary), *base_flags, "--goal", goal]
    if args.full_union:
        cmd.append("--full-union")
    return run_json_subprocess(
        cmd, what=f"verify for goal {goal!r}", input=json.dumps(subsets)
    )


def run_pair(args: Args, base_flags: list[str], item: WorkItem) -> list[dict]:
    """Run one seed/goal pair's attempt loop and return its result rows.

    The whole loop runs in a single `verify` subprocess: Python precomputes all
    attempt subsets and hands them over, `verify` runs one leg per subset and
    early-stops at the first reach. Runs in a worker thread over no shared
    state; marks its last row `gave_up` if every attempt failed. Each row's `k`
    is the *effective* subset size, not the requested one, so a capped leg
    reports what it actually ran.

    An empty candidate pool is a hard error (nothing to draw guides from), so
    this raises rather than emitting a row.
    """
    rng = random.Random(f"{args.rng_seed}:{item.seed}:{item.goal}")
    attempt_subsets = build_attempt_subsets(item.pool, args.strategy, args.k, args.attempts, rng)
    if any(not guides for guides in attempt_subsets):
        raise RuntimeError(
            f"empty candidate pool for seed {item.seed!r} goal {item.goal!r}: "
            f"strategy {args.strategy!r} drew no guides"
        )

    # One subprocess for the whole pair; `verify` early-stops at the first reach,
    # so `results` is one LegResult per attempt actually run (<= len(subsets)).
    results = run_legs(args, base_flags, item.goal, attempt_subsets)
    rows: list[dict] = []
    for attempt, (guides, result) in enumerate(zip(attempt_subsets, results)):
        row = {
            "seed": item.seed,
            "goal": item.goal,
            "strategy": args.strategy,
            "k": len(guides),
            "attempt": attempt,
            "reached": result["reached"],
            "gave_up": False,
            "panic": result.get("panic", False),
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


def build_work_items(seed_records: list, strategy: str) -> list[WorkItem]:
    """Flatten `sample`'s seed records into independent (seed, goal) items.

    Each item runs its own sequential attempt loop (early-stops on first reach);
    items run concurrently.
    """
    items: list[WorkItem] = []
    for record in seed_records:
        pool = record["candidates"].get(pool_key(strategy), [])
        guide_meta = {
            "guide_nodes": record["guide_nodes"],
            "guide_classes": record["guide_classes"],
            "guide_time": record["guide_time"],
            "guide_memory": record["guide_memory"],
        }
        for goal in record["goals"]:
            items.append(WorkItem(record["seed"], goal, pool, guide_meta))
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
    print(f"Running legs for {len(items)} (seed, goal) item(s)", file=sys.stderr)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs or os.cpu_count() or 1) as pool_exec:
        futures = [pool_exec.submit(run_pair, args, base_flags, item) for item in items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="legs", unit="pair"):
            rows.extend(fut.result())
    return rows


def report_results(args: Args, out: Path, rows: list[dict]) -> None:
    """Write results/config to `out` and print the reach-rate summary."""
    df = pl.DataFrame(rows, schema_overrides=LEG_RESULT_DTYPES)
    df.write_parquet(out / "results.parquet")
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, default=str))

    # A pair counts as reached if any of its attempts reached.
    reached_pairs = df.filter(pl.col("reached")).select("seed", "goal").unique().height
    total_pairs = df.select("seed", "goal").unique().height
    reach_rate = reached_pairs / total_pairs if total_pairs else 0.0
    print(
        f"\nReached {reached_pairs}/{total_pairs} seed/goal pairs "
        f"(reach rate {reach_rate:.2f}) at k={args.k}. "
        f"Wrote {out / 'results.parquet'}",
        file=sys.stderr,
    )


def main() -> int:
    args = tyro.cli(Args, description=__doc__)

    if all(v is None for v in (args.stop_iters, args.stop_nodes, args.stop_time, args.stop_memory)):
        print(
            "No guide-replay budget given; pass at least one of "
            "--stop-iters/--stop-nodes/--stop-time/--stop-memory.",
            file=sys.stderr,
        )
        return 2
    exit_if_missing(args.sample_binary, args.verify_binary)

    cfg = json.loads((args.path / "goal_args.json").read_text())
    args.k = 1 if args.strategy in SMALLEST_STRATEGIES else args.k
    base_flags = ["--language", str(cfg["language"]), *limit_flags(eqsat_limits(cfg))]
    out = resolve_output_dir(args)

    samples_path = run_sample(args, cfg, out / "sample_run")
    seed_records = json.loads(samples_path.read_text())

    items = build_work_items(seed_records, args.strategy)
    warn_pool_shortfall(items, args.strategy, args.k, args.attempts)

    rows = run_all_pairs(args, base_flags, items)

    report_results(args, out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
