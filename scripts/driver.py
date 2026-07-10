"""Drive the guide search from Python.

Replaces the "second half" of the old `guide` binary. The Rust `sample` binary
produces a guide-candidate menu per seed (`samples.json`); this driver then runs
the individual search legs (one `verify` subprocess each), resampling fresh
guide subsets until it either reaches the goal or uses all `--attempts` for a
seed/goal pair. All logging and data wrangling (parquet/JSON) live here.

Example:
    cargo build --release --bin sample --bin verify
    uv run scripts/driver.py data/seed_terms/dusky-cramp \\
        --attempts 5 --k 1 2 5 10 50 100 \\
        --strategy no_replacement_count --full-union
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from tqdm import tqdm

# Strategy names emitted by `sample` (see Strategy::name in src/cli/sample.rs).
# The `with_replacement_*` pools are re-drawn *with* replacement across a leg's
# `k` picks; everything else is drawn without replacement.
SAMPLING_STRATEGIES = (
    "no_replacement_count",
    "no_replacement_naive",
    "with_replacement_count",
    "with_replacement_naive",
)
SMALLEST_STRATEGIES = ("smallest_novel", "smallest_overall")
ALL_STRATEGIES = SAMPLING_STRATEGIES + SMALLEST_STRATEGIES

# Fields copied straight out of a `verify` LegResult onto a result row.
LEG_RESULT_FIELDS = (
    "iters",
    "nodes",
    "classes",
    "total_applied",
    "total_time",
    "stop_reason",
)


@dataclass
class Args:
    # I/O
    path: tyro.conf.Positional[Path]
    """Seed folder with `terms.json` (enriched by `goal`) and `args.json`."""

    output: Path | None = None
    """Run folder for `results.parquet`/`results.json`. Auto-created under
    `data/guide_driver/` if omitted."""

    sample_binary: Path = Path("target/release/sample")
    verify_binary: Path = Path("target/release/verify")

    # search policy
    attempts: int = 5
    """How many legs to try per (seed, goal, k), each with a freshly resampled
    guide subset. Counts the first try, so `attempts=1` means a single leg with
    no resampling. Stops early on the first reach; gives up after the last."""
    strategy: str = "no_replacement_count"
    """Which candidate pool to restart with. One of: """ + ", ".join(ALL_STRATEGIES)  # pyright: ignore[reportUnusedExpression]
    k: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 5, 10, 50, 100])
    """Guide-set sizes to sweep: each seed/goal pair is tried independently at
    every `k` (its own attempt loop), reproducing the old reach-rate-vs-k curve.
    Forced to `[1]` for the `smallest_*` strategies."""
    full_union: bool = False
    """Use the experimental full-union add for the leg egraph."""

    # sampling
    samples_per_strategy: int = 10000
    """Menu size per sampling strategy (passed to `sample`)."""

    take_first: int | None = None
    """Only process the first N seeds."""

    seed: int = 0
    """RNG seed for subset selection (offset per attempt)."""

    jobs: int | None = None
    """Max concurrent `verify` legs (one seed/goal pair per worker). Each pair's
    attempt loop stays sequential, so this parallelises across pairs. Defaults to
    `os.cpu_count()`. Lower it if the large leg egraphs exhaust RAM."""


@dataclass
class WorkItem:
    """One independent (seed, goal, k) unit of work for the leg loop.

    Each item carries the candidate `pool` it draws guides from and the
    seed-level `guide_meta` copied onto every result row. Items are built up
    front (`build_work_items`) and run concurrently, one attempt loop each.
    """

    seed: str
    goal: str
    k: int
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


def count_seeds(args: Args) -> int:
    """Count the seeds to fan out over, honouring `--take-first`.

    `terms.json` is a list of `[size, {seed: payload}]` groups; the seed count is
    the total across all groups. We only need the count, not the payloads, since
    `sample` processes one seed per `--seed-index`.
    """
    groups = json.loads((args.path / "terms.json").read_text())
    total = sum(len(inner) for _size, inner in groups)
    if args.take_first is not None:
        total = min(total, args.take_first)
    return total


def run_sample_shard(args: Args, seed_index: int, shard_out: Path) -> Path:
    """Run `sample` for a single seed, writing `samples.json` into `shard_out`.

    Each shard gets its own directory so parallel runs don't collide. Output is
    captured rather than streamed, and only surfaced on failure.
    """
    cmd = [
        str(args.sample_binary),
        str(args.path),
        "--output",
        str(shard_out),
        "--samples-per-strategy",
        str(args.samples_per_strategy),
        "--seed-index",
        str(seed_index),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sample failed (code {proc.returncode}) for seed index {seed_index}:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return shard_out / "samples.json"


def run_sample(args: Args, sample_out: Path) -> Path:
    """Sample the guide menu in parallel, one `sample` subprocess per seed.

    Merges each shard's `samples.json` into a single `samples.json` under
    `sample_out`, in seed order so the output is deterministic regardless of
    completion order.
    """
    n_seeds = count_seeds(args)
    sample_out.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or os.cpu_count() or 1
    print(
        f"Sampling guide menu for {n_seeds} seed(s) -> {sample_out} ({jobs} workers)",
        file=sys.stderr,
    )

    # index -> records, filled as shards finish; merged in seed order below.
    shard_records: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool_exec:
        futures = {
            pool_exec.submit(run_sample_shard, args, i, sample_out / f"seed_{i:04d}"): i
            for i in range(n_seeds)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sampling", unit="seed"):
            i = futures[fut]
            shard_records[i] = json.loads(fut.result().read_text())

    merged = [rec for i in range(n_seeds) for rec in shard_records[i]]
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
    out of a shuffled pool, so a k's attempts are disjoint until the pool is
    exhausted (`attempts * eff_k > len(pool)`), where a fresh pass is appended.
    Cross-k reuse isn't handled here: each k is a separate WorkItem with its own
    shuffle.
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


def run_leg(
    args: Args,
    language: str,
    goal: str,
    guides: list,
) -> dict:
    """Run one `verify` subprocess and return the parsed `LegResult`."""
    request = {
        "language": language,
        "full_union": args.full_union,
        "goal": goal,
        "guides": guides,
    }
    proc = subprocess.run(
        [str(args.verify_binary), "--folder", str(args.path)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # A panic-guarded leg still exits 0 with `panic: true`; a nonzero code
        # means the process itself failed (bad request, parse error, ...).
        raise RuntimeError(
            f"verify failed (code {proc.returncode}) for goal {goal!r}:\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        # verify exited 0 but stdout wasn't the expected JSON, e.g. a stray
        # diagnostic leaked onto the JSON channel. Surface stdout/stderr so the
        # protocol violation is diagnosable instead of a bare decode error.
        raise RuntimeError(
            f"verify returned non-JSON stdout for goal {goal!r}: {e}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        ) from e


def make_row(item: WorkItem, strategy: str, attempt: int, guides: list) -> dict:
    """Build a result row for one attempt, pre-filled with empty leg metrics.

    `run_pair` overwrites the metric fields once the leg returns (or leaves them
    as-is for an empty pool). Note `k` is the *effective* subset size, not the
    requested one, so a capped leg reports what it actually ran.
    """
    return {
        "seed": item.seed,
        "goal": item.goal,
        "strategy": strategy,
        "k": len(guides),
        "attempt": attempt,
        "reached": False,
        "gave_up": False,
        "panic": False,
        **{field: None for field in LEG_RESULT_FIELDS},
        **item.guide_meta,
    }


def run_pair(args: Args, language: str, item: WorkItem) -> list[dict]:
    """Run one seed/goal pair's attempt loop and return its result rows.

    Attempts are sequential and early-stop on the first reach. Runs in a worker
    thread over no shared state; marks its last row `gave_up` if every attempt
    failed.
    """
    rng = random.Random(f"{args.seed}:{item.seed}:{item.goal}")
    attempt_subsets = build_attempt_subsets(item.pool, args.strategy, item.k, args.attempts, rng)
    rows: list[dict] = []
    for attempt, guides in enumerate(attempt_subsets):
        row = make_row(item, args.strategy, attempt, guides)
        if not guides:
            row["stop_reason"] = "empty_pool"
            rows.append(row)
            return rows
        result = run_leg(args, language, item.goal, guides)
        row["reached"] = result["reached"]
        row["panic"] = result.get("panic", False)
        for field in LEG_RESULT_FIELDS:
            row[field] = result.get(field)
        rows.append(row)
        if result["reached"]:
            return rows

    # Used every attempt without reaching: mark the last leg as given up.
    if rows:
        rows[-1]["gave_up"] = True
    return rows


def check_preconditions(args: Args) -> str | None:
    """Validate `--strategy` and that both binaries exist.

    Returns an error message for the caller to print (exit 2), or `None` when
    everything checks out.
    """
    if args.strategy not in ALL_STRATEGIES:
        return f"Unknown --strategy {args.strategy!r}; choose one of {ALL_STRATEGIES}"
    for binary in (args.sample_binary, args.verify_binary):
        if not binary.exists():
            return (
                f"Binary not found: {binary}. "
                "Build with `cargo build --release --bin sample --bin verify`."
            )
    return None


def resolve_output_dir(args: Args) -> Path:
    """Resolve (and create) the run folder, auto-numbering `run.N` if unset."""
    out = args.output
    if out is None:
        base = Path("data/guide_driver")
        base.mkdir(parents=True, exist_ok=True)
        existing = [int(p.suffix[1:]) for p in base.glob("run.*") if p.suffix[1:].isdigit()]
        out = base / f"run.{max(existing, default=0) + 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_work_items(seed_records: list, strategy: str, k_values: list[int]) -> list[WorkItem]:
    """Flatten `sample`'s seed records into independent (seed, goal, k) items.

    Each item runs its own sequential attempt loop (early-stops on first reach);
    items run concurrently. Sweeping k per pair reproduces the reach-rate-vs-k
    curve.
    """
    items: list[WorkItem] = []
    for record in seed_records:
        pool = record["candidates"].get(pool_key(strategy), [])
        guide_meta = {
            "guide_nodes": record["guide_nodes"],
            "guide_classes": record["guide_classes"],
            "guide_time": record["guide_time"],
        }
        for goal in record["goals"]:
            for k in k_values:
                items.append(WorkItem(record["seed"], goal, k, pool, guide_meta))
    return items


def warn_pool_shortfall(items: list[WorkItem], strategy: str, attempts: int) -> None:
    """Warn once per (pool size, k) when no_replacement_* must cap or reuse.

    Deduped on (pool size, k) since the shortfall is identical across pairs.
    """
    if not (strategy in SAMPLING_STRATEGIES and strategy.startswith("no_replacement")):
        return
    combos = {(len(item.pool), item.k) for item in items}
    for pool_size, k in sorted(combos):
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


def run_all_pairs(args: Args, language: str, items: list[WorkItem]) -> list[dict]:
    """Run every work item's attempt loop concurrently and collect result rows."""
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs or os.cpu_count() or 1) as pool_exec:
        futures = [pool_exec.submit(run_pair, args, language, item) for item in items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="legs", unit="pair×k"):
            rows.extend(fut.result())
    return rows


def report_results(args: Args, out: Path, rows: list[dict]) -> None:
    """Write results/config to `out` and print the reach-rate summary."""
    df = pl.DataFrame(rows)
    df.write_parquet(out / "results.parquet")
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, default=str))

    # Per-pair "reached" = reached at any swept k. Also report reach rate per k
    # (the point of the sweep): fraction of (seed, goal) pairs reached at each k.
    reached_pairs = df.filter(pl.col("reached")).select("seed", "goal").unique().height
    total_pairs = df.select("seed", "goal").unique().height
    per_k = (
        df.group_by("k", "seed", "goal")
        .agg(pl.col("reached").any().alias("reached"))
        .group_by("k")
        .agg(pl.col("reached").mean().alias("reach_rate"))
        .sort("k")
    )
    print(
        f"\nReached {reached_pairs}/{total_pairs} seed/goal pairs (at any k). "
        f"Wrote {out / 'results.parquet'}",
        file=sys.stderr,
    )
    print("Reach rate by k:", file=sys.stderr)
    for r in per_k.iter_rows(named=True):
        print(f"  k={r['k']:>3}: {r['reach_rate']:.2f}", file=sys.stderr)


def main() -> int:
    args = tyro.cli(Args, description=__doc__)

    error = check_preconditions(args)
    if error is not None:
        print(error, file=sys.stderr)
        return 2

    k_values = [1] if args.strategy in SMALLEST_STRATEGIES else args.k
    language = json.loads((args.path / "args.json").read_text())["language"]
    out = resolve_output_dir(args)

    print("RUNNING SAMPLING")
    samples_path = run_sample(args, out / "sample_run")
    print("SAMPLING FINISHED")
    seed_records = json.loads(samples_path.read_text())

    items = build_work_items(seed_records, args.strategy, k_values)
    warn_pool_shortfall(items, args.strategy, args.attempts)

    print("STARTING PAIRS")
    rows = run_all_pairs(args, language, items)

    report_results(args, out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
