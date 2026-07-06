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
    samples_per_strategy: int = 1000
    """Menu size per sampling strategy (passed to `sample`)."""

    take_first: int | None = None
    """Only process the first N seeds."""

    seed: int = 0
    """RNG seed for subset selection (offset per attempt)."""

    jobs: int | None = None
    """Max concurrent `verify` legs (one seed/goal pair per worker). Each pair's
    attempt loop stays sequential, so this parallelises across pairs. Defaults to
    `os.cpu_count()`. Lower it if the large leg egraphs exhaust RAM."""


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


def pick_subset(pool: list, strategy: str, k: int, rng: random.Random) -> list:
    """Choose this leg's guide subset from a strategy's candidate pool.

    `smallest_*` pools hold a single term (k is forced to 1). `with_replacement_*`
    draws `k` picks with replacement; the rest without. Returns raw guide node
    lists ready to embed in a `LegRequest`.
    """
    if strategy in SMALLEST_STRATEGIES:
        return pool[:1]
    if not pool:
        return []
    if strategy.startswith("with_replacement"):
        return [rng.choice(pool) for _ in range(k)]
    take = min(k, len(pool))
    return rng.sample(pool, take)


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
    return json.loads(proc.stdout)


def run_pair(
    args: Args,
    language: str,
    k: int,
    seed: str,
    goal: str,
    pool: list,
    guide_meta: dict,
) -> list[dict]:
    """Run one seed/goal pair's attempt loop and return its result rows.

    Attempts are sequential and early-stop on the first reach. Runs in a worker
    thread over no shared state; marks its last row `gave_up` if every attempt
    failed.
    """
    rows: list[dict] = []
    for attempt in range(args.attempts):
        rng = random.Random(f"{args.seed}:{seed}:{goal}:{attempt}")
        guides = pick_subset(pool, args.strategy, k, rng)
        row = {
            "seed": seed,
            "goal": goal,
            "strategy": args.strategy,
            "k": len(guides),
            "attempt": attempt,
            "reached": False,
            "gave_up": False,
            "iters": None,
            "nodes": None,
            "classes": None,
            "total_applied": None,
            "total_time": None,
            "stop_reason": None,
            "panic": False,
            **guide_meta,
        }
        if not guides:
            row["stop_reason"] = "empty_pool"
            rows.append(row)
            return rows
        result = run_leg(args, language, goal, guides)
        row["reached"] = result["reached"]
        row["iters"] = result.get("iters")
        row["nodes"] = result.get("nodes")
        row["classes"] = result.get("classes")
        row["total_applied"] = result.get("total_applied")
        row["total_time"] = result.get("total_time")
        row["stop_reason"] = result.get("stop_reason")
        row["panic"] = result.get("panic", False)
        rows.append(row)
        if result["reached"]:
            return rows

    # Used every attempt without reaching: mark the last leg as given up.
    if rows:
        rows[-1]["gave_up"] = True
    return rows


def main() -> int:
    args = tyro.cli(Args, description=__doc__)

    if args.strategy not in ALL_STRATEGIES:
        print(
            f"Unknown --strategy {args.strategy!r}; choose one of {ALL_STRATEGIES}",
            file=sys.stderr,
        )
        return 2
    k_values = [1] if args.strategy in SMALLEST_STRATEGIES else args.k

    for binary in (args.sample_binary, args.verify_binary):
        if not binary.exists():
            print(
                f"Binary not found: {binary}. "
                "Build with `cargo build --release --bin sample --bin verify`.",
                file=sys.stderr,
            )
            return 2

    language = json.loads((args.path / "args.json").read_text())["language"]

    out = args.output
    if out is None:
        base = Path("data/guide_driver")
        base.mkdir(parents=True, exist_ok=True)
        existing = [int(p.suffix[1:]) for p in base.glob("run.*") if p.suffix[1:].isdigit()]
        out = base / f"run.{max(existing, default=0) + 1}"
    out.mkdir(parents=True, exist_ok=True)

    print("RUNNING SAMPLING")
    samples_path = run_sample(args, out / "sample_run")
    print("SAMPLING FINISHED")
    seed_records = json.loads(samples_path.read_text())

    # Flatten to independent (seed, goal, k) work items. Each item runs its own
    # sequential attempt loop (early-stops on first reach); items run
    # concurrently. Sweeping k per pair reproduces the reach-rate-vs-k curve.
    items = []
    for record in seed_records:
        pool = record["candidates"].get(pool_key(args.strategy), [])
        guide_meta = {
            "guide_nodes": record["guide_nodes"],
            "guide_classes": record["guide_classes"],
            "guide_time": record["guide_time"],
        }
        for goal in record["goals"]:
            for k in k_values:
                items.append((record["seed"], goal, k, pool, guide_meta))

    jobs = args.jobs or os.cpu_count() or 1
    rows: list[dict] = []
    print("STARTING PAIRS")
    with ThreadPoolExecutor(max_workers=jobs) as pool_exec:
        futures = [
            pool_exec.submit(run_pair, args, language, k, seed, goal, cand_pool, guide_meta)
            for (seed, goal, k, cand_pool, guide_meta) in items
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="legs", unit="pair×k"):
            rows.extend(fut.result())

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
