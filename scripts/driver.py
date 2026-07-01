"""Drive the guide search from Python.

Replaces the "second half" of the old `guide` binary. The Rust `sample` binary
produces a guide-candidate menu per seed (`samples.json`); this driver then runs
the individual search legs (one `verify` subprocess each), restarting with fresh
guide subsets until it either reaches the goal or exhausts `--max-restarts` for a
seed/goal pair. All logging and data wrangling (parquet/JSON) live here.

Example:
    cargo build --release --bin sample --bin verify
    uv run scripts/driver.py data/seed_terms/dusky-cramp \\
        --max-restarts 5 --k 10 --strategy no_replacement_count --full-union
"""

from __future__ import annotations

import dataclasses
import json
import random
import subprocess
import sys
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
    max_restarts: int = 5
    """Give up on a seed/goal pair after this many failed legs."""
    strategy: str = "no_replacement_count"
    """Which candidate pool to restart with. One of: """ + ", ".join(ALL_STRATEGIES)
    k: int = 10
    """Guides unioned per leg. Forced to 1 for the `smallest_*` strategies."""
    full_union: bool = False
    """Use the experimental full-union add for the leg egraph."""

    # sampling
    samples_per_strategy: int = 1000
    """Menu size per sampling strategy (passed to `sample`)."""

    take_first: int | None = None
    """Only process the first N seeds."""

    seed: int = 0
    """RNG seed for subset selection (offset per restart round)."""


def run_sample(args: Args, sample_out: Path) -> Path:
    """Invoke `sample` and return the path to its `samples.json`."""
    cmd = [
        str(args.sample_binary),
        str(args.path),
        "--output",
        str(sample_out),
        "--samples-per-strategy",
        str(args.samples_per_strategy),
    ]
    if args.take_first is not None:
        cmd += ["--take-first", str(args.take_first)]
    print(f"Sampling guide menu -> {sample_out}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return sample_out / "samples.json"


def pick_subset(
    pool: list, strategy: str, k: int, rng: random.Random
) -> list:
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


def main() -> int:
    args = tyro.cli(Args, description=__doc__)

    if args.strategy not in ALL_STRATEGIES:
        print(
            f"Unknown --strategy {args.strategy!r}; choose one of {ALL_STRATEGIES}",
            file=sys.stderr,
        )
        return 2
    k = 1 if args.strategy in SMALLEST_STRATEGIES else args.k

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

    samples_path = run_sample(args, out / "sample_run")
    seed_records = json.loads(samples_path.read_text())

    rows: list[dict] = []
    for record in tqdm(seed_records, desc="seeds", unit="seed"):
        seed = record["seed"]
        pool = record["candidates"].get(args.strategy, [])
        # Per-seed guide-egraph overhead, carried into every row so analysis is
        # self-contained (analysis/helpers.py folds this into nodes/total_time).
        guide_meta = {
            "guide_nodes": record["guide_nodes"],
            "guide_classes": record["guide_classes"],
            "guide_time": record["guide_time"],
        }
        for goal in tqdm(record["goals"], desc="goals", unit="goal", leave=False):
            reached = False
            for rnd in range(args.max_restarts):
                rng = random.Random(f"{args.seed}:{seed}:{goal}:{rnd}")
                guides = pick_subset(pool, args.strategy, k, rng)
                row = {
                    "seed": seed,
                    "goal": goal,
                    "strategy": args.strategy,
                    "k": len(guides),
                    "restart_round": rnd,
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
                    break
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
                    reached = True
                    break
            if not reached and rows:
                # Mark the pair as given up on its last recorded round.
                rows[-1]["gave_up"] = True

    df = pl.DataFrame(rows)
    df.write_parquet(out / "results.parquet")
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    (out / "config.json").write_text(
        json.dumps(dataclasses.asdict(args), indent=2, default=str)
    )

    reached_pairs = (
        df.filter(pl.col("reached"))
        .select("seed", "goal")
        .unique()
        .height
    )
    total_pairs = df.select("seed", "goal").unique().height
    print(
        f"\nReached {reached_pairs}/{total_pairs} seed/goal pairs. "
        f"Wrote {out / 'results.parquet'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
