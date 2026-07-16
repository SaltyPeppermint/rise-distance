"""Drive the guide search from Python.

Replaces the "second half" of the old `guide` binary. The Rust `sample` and
`verify` binaries touch no files: `sample` takes everything on argv, `verify`
takes the per-leg guides on stdin plus `--goal`/`--language`/eqsat limits on
argv. This
driver owns all file I/O: it reads `args.json` (language, eqsat limits, and the
optional guide-replay `stop_*` overrides), flattens the goal-enriched
`terms.json` into per-seed specs, runs one `sample` subprocess per seed
(splicing the seed's goals back onto the output, since `sample` no longer
echoes them), then runs the search legs (one `verify` subprocess each),
resampling fresh guide subsets until it either reaches the goal or uses all
`--attempts` for a seed/goal pair. All logging and data wrangling
(parquet/JSON) live here.

Example:
    cargo build --release --bin sample --bin verify
    uv run scripts/driver.py data/seed_terms/dusky-cramp \\
        --attempts 5 --k 1 2 5 10 50 100 \\
        --strategy no_replacement_count --full-union
"""

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


def eqsat_limits(path: Path) -> dict:
    """Read the search-phase eqsat limits out of `args.json`.

    The four required values live at the top level; `max_memory` (RSS ceiling
    in bytes) is optional and `None` when absent.
    """
    cfg = json.loads((path / "args.json").read_text())
    return {
        "max_iters": cfg["max_iters"],
        "max_nodes": cfg["max_nodes"],
        "max_time": cfg["max_time"],
        "max_memory": cfg.get("max_memory"),
        "backoff_scheduler": cfg["backoff_scheduler"],
    }


def replay_limits(path: Path) -> dict:
    """Compute the effective guide-replay limits for `sample`: the search-phase
    limits overridden by the optional `stop_*` keys in `args.json`
    (`stop_time`/`stop_nodes`/`stop_memory`, the latter in bytes). Only the
    guide replay uses these, not the leg search. The per-seed `max_iters`
    override is applied in `run_sample_shard`.
    """
    cfg = json.loads((path / "args.json").read_text())
    limits = eqsat_limits(path)
    for stop_key, max_key in (
        ("stop_time", "max_time"),
        ("stop_nodes", "max_nodes"),
        ("stop_memory", "max_memory"),
    ):
        value = cfg.get(stop_key)
        if value is not None:
            limits[max_key] = value
    return limits


def limit_flags(limits: dict) -> list[str]:
    """Turn an eqsat-limit dict into the `--max-*` CLI flags `sample` and
    `verify` take. `--max-memory` is added only when set;
    `--backoff-scheduler` is a presence flag, added only when true.
    """
    flags = [
        "--max-iters",
        str(limits["max_iters"]),
        "--max-nodes",
        str(limits["max_nodes"]),
        "--max-time",
        str(limits["max_time"]),
    ]
    if limits["max_memory"] is not None:
        flags += ["--max-memory", str(limits["max_memory"])]
    if limits["backoff_scheduler"]:
        flags.append("--backoff-scheduler")
    return flags


def language_flag(path: Path) -> list[str]:
    """Build the `--language` flag from `args.json`."""
    cfg = json.loads((path / "args.json").read_text())
    return ["--language", str(cfg["language"])]


@dataclass
class SeedSpec:
    """One seed's inputs from the goal-enriched `terms.json`: `seed` and
    `guide_iters` go to `sample` on argv; `goals` stays Python-side and is
    merged back onto `sample`'s output record."""

    seed: str
    guide_iters: int
    goals: list[str]


def flatten_enriched_seeds(args: Args) -> list[SeedSpec]:
    """Flatten the goal-enriched `terms.json` into per-seed specs.

    `terms.json` is a list of `[size, {seed: payload}]` groups, each payload a
    `Result`-shaped `{"Ok": GoalGenMetadata}` / `{"Err": msg}`. We pull the seed,
    its recorded `guide_iters` (`guide_egraph.iters`), and its goals off each
    `Ok`. `Err` seeds (goal stage failed) are dropped. Groups in file order,
    terms sorted for a deterministic `--take-first`.
    """
    groups = json.loads((args.path / "terms.json").read_text())
    specs: list[SeedSpec] = []
    for _size, terms_map in groups:
        for seed in sorted(terms_map):
            ok = terms_map[seed].get("Ok")
            if ok is None:
                continue  # Err seed: goal stage failed, nothing to replay.
            specs.append(SeedSpec(seed, ok["guide_egraph"]["iters"], ok["goals"]))
    if args.take_first is not None:
        specs = specs[: args.take_first]
    return specs


def run_sample_shard(args: Args, base_flags: list[str], limits: dict, spec: SeedSpec) -> list:
    """Run `sample` for a single seed and return its parsed record list.

    Everything goes on argv: `--seed`, `--language`, and `limits` (the folder's
    `replay_limits`) with `max_iters` set to the seed's recorded `guide_iters`.
    `sample` prints a one-element `[SeedSamples]` array to stdout (empty on an
    internal failure) and logs to stderr, which is surfaced only on a nonzero
    exit. The spec's `goals` are spliced back onto each record here, since
    `sample` no longer echoes them.
    """
    cmd = [
        str(args.sample_binary),
        *base_flags,
        *limit_flags({**limits, "max_iters": spec.guide_iters}),
        "--seed",
        spec.seed,
        "--samples-per-strategy",
        str(args.samples_per_strategy),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sample failed (code {proc.returncode}) for seed {spec.seed!r}:\n{proc.stderr}"
        )
    try:
        records = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"sample returned non-JSON stdout for seed {spec.seed!r}: {e}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        ) from e
    for record in records:
        record["goals"] = spec.goals
    return records


def run_sample(args: Args, sample_out: Path) -> Path:
    """Sample the guide menu in parallel, one `sample` subprocess per seed.

    Captures each subprocess's stdout, merges the records in seed order (so the
    output is deterministic regardless of completion order), and writes a single
    `samples.json` under `sample_out` for provenance.
    """
    specs = flatten_enriched_seeds(args)
    sample_out.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or os.cpu_count() or 1
    # sample gets the effective guide-replay limits (per-seed guide_iters
    # spliced in per shard); verify keeps the plain search-phase limits.
    sample_flags = language_flag(args.path)
    limits = replay_limits(args.path)
    print(
        f"Sampling guide menu for {len(specs)} seed(s) -> {sample_out} ({jobs} workers)",
        file=sys.stderr,
    )

    # index -> records, filled as subprocesses finish; merged in seed order below.
    shard_records: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool_exec:
        futures = {
            pool_exec.submit(run_sample_shard, args, sample_flags, limits, spec): i
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
    base_flags: list[str],
    goal: str,
    guides: list,
) -> dict:
    """Run one `verify` subprocess and return the parsed `LegResult`.

    `verify` touches no files: the per-leg `guides` go in on stdin, and
    `--goal`/`--language`/`--full-union`/the eqsat flags on argv.
    """
    cmd = [str(args.verify_binary), *base_flags, "--goal", goal]
    if args.full_union:
        cmd.append("--full-union")
    proc = subprocess.run(
        cmd,
        input=json.dumps(guides),
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


def run_pair(args: Args, base_flags: list[str], item: WorkItem) -> list[dict]:
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
        result = run_leg(args, base_flags, item.goal, guides)
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


def run_all_pairs(args: Args, base_flags: list[str], items: list[WorkItem]) -> list[dict]:
    """Run every work item's attempt loop concurrently and collect result rows."""
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs or os.cpu_count() or 1) as pool_exec:
        futures = [pool_exec.submit(run_pair, args, base_flags, item) for item in items]
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
    base_flags = language_flag(args.path) + limit_flags(eqsat_limits(args.path))
    out = resolve_output_dir(args)

    print("RUNNING SAMPLING")
    samples_path = run_sample(args, out / "sample_run")
    print("SAMPLING FINISHED")
    seed_records = json.loads(samples_path.read_text())

    items = build_work_items(seed_records, args.strategy, k_values)
    warn_pool_shortfall(items, args.strategy, args.attempts)

    print("STARTING PAIRS")
    rows = run_all_pairs(args, base_flags, items)

    report_results(args, out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
