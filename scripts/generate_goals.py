"""Drive goal-term generation from Python.

The Rust `goal` binary touches no files: it processes one seed per invocation
(`--seed <expr>`, with `--language` and the eqsat limits also on argv) and
prints that seed's `{"Ok":..}`/`{"Err":..}` payload to stdout. This driver owns
all file I/O: it reads `generation_args.json` (for `language` and the eqsat
config), flattens/filters the `terms.json` seed list, fans the invocations out
across seeds, captures each payload, and writes the enriched copy to
`goal_terms.json` next to it (same `[size, {term: payload}]` grouping;
`terms.json` is never modified). A `goal_args.json` sidecar records the eqsat
config the goals were generated under (top-level, read by `guided_search.py`)
plus this run's CLI args under `driver_args`.

Example:
    cargo build --release --bin goal
    uv run scripts/generate_goals.py data/seed_terms/dusky-cramp \\
        --goals 10 --jobs 8
"""

import dataclasses
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import tqdm

from common import eqsat_limits, exit_if_missing, language_eqsat_flags, run_json_subprocess


@dataclass
class Args:
    # I/O
    path: tyro.conf.Positional[Path]
    """Seed folder with `terms.json` and `generation_args.json`. The enriched
    per-seed goal data is written to `goal_terms.json` in the same folder,
    alongside a `goal_args.json` sidecar."""

    force: bool = False
    """Overwrite existing `goal_terms.json`/`goal_args.json` in the seed
    folder instead of refusing."""

    goal_binary: Path = Path("target/release/goal")

    # goal generation
    goals: int = 10
    """Number of goal terms to sample per seed."""

    size_distribution: str | None = None
    """How to distribute the sample budget across sizes (forwarded if set)."""

    goal_sample_strategy: str | None = None
    """How to sample the GOAL terms (forwarded if set)."""

    skip_unmeasured: bool = True
    """Skip seeds whose `peak_memory_bytes` is missing or -1 (measurement
    failed)."""

    retry_step: int = 5
    """How much to grow `max_size` on each precompute retry."""

    max_retries: int = 20
    """How many times to retry precompute with a larger `max_size`."""

    sample_sizes: int = 5
    """How many sizes need to be present in the precomputed root histogram."""

    # fan-out
    seeds: int | None = None
    """Only process the first N seed terms (sorted order, so stable across
    runs). All seeds if omitted."""

    jobs: int | None = None
    """Max concurrent `goal` subprocesses (one seed each). Defaults to
    `os.cpu_count()`. Lower it if the per-seed eqsat egraphs exhaust RAM, since
    each seed's eqsat can use several GB."""


def is_measured(record: object) -> bool:
    """True iff the record's measurement (`record[2]`, a `{"peak", "iterations"}`
    object) is present and its `peak` is not the -1 failure sentinel."""
    if not (isinstance(record, list) and len(record) > 2):
        return False
    measurement = record[2]
    if not isinstance(measurement, dict):
        return False
    peak = measurement.get("peak")
    return isinstance(peak, int) and peak >= 0


def flatten_seeds(args: Args) -> list[str]:
    """Flatten `terms.json` into the list of seed s-expressions to process.

    Groups in file order; terms sorted within each group for deterministic
    `--seeds`; then the `skip_unmeasured` filter.
    """
    groups = json.loads((args.path / "terms.json").read_text())
    seeds: list[str] = []
    for _size, terms_map in groups:
        for term in sorted(terms_map):
            if args.skip_unmeasured and not is_measured(terms_map[term]):
                continue
            seeds.append(term)
    if args.seeds is not None:
        seeds = seeds[: args.seeds]
    return seeds


def run_goal_shard(args: Args, base_flags: list[str], seed: str) -> object:
    """Run `goal` for a single seed and return its parsed stdout payload."""
    cmd = [
        str(args.goal_binary),
        "--seed",
        seed,
        *base_flags,
        "--goals",
        str(args.goals),
        "--retry-step",
        str(args.retry_step),
        "--max-retries",
        str(args.max_retries),
        "--sample-sizes",
        str(args.sample_sizes),
    ]
    if args.size_distribution is not None:
        cmd += ["--size-distribution", args.size_distribution]
    if args.goal_sample_strategy is not None:
        cmd += ["--goal-sample-strategy", args.goal_sample_strategy]
    return run_json_subprocess(cmd, what=f"goal for seed {seed!r}")


def run_all_seeds(args: Args, base_flags: list[str], seeds: list[str]) -> dict[str, object]:
    """Run one `goal` subprocess per seed and collect `{seed_str: payload}`."""
    jobs = args.jobs or os.cpu_count() or 1
    print(
        f"Generating goals for {len(seeds)} seed(s) ({jobs} workers)",
        file=sys.stderr,
    )

    enriched: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_goal_shard, args, base_flags, seed): seed for seed in seeds}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="seeds", unit="seed"):
            enriched[futures[fut]] = fut.result()
    return enriched


def write_enriched_terms(src: Path, dst: Path, enriched: dict[str, object]) -> int:
    """Write the enriched copy of `src` (the seed `terms.json`) to `dst`,
    replacing each seed's record with its enriched payload. Preserves the
    `[size, {term: payload}]` grouping; seeds absent from `enriched` (e.g.
    dropped by `--seeds`) are omitted, and groups left empty are dropped.
    """
    groups = json.loads(src.read_text())
    out_groups = []
    for size, terms_map in groups:
        new_inner = {term: enriched[term] for term in terms_map if term in enriched}
        if new_inner:
            out_groups.append([size, new_inner])

    dst.write_text(json.dumps(out_groups, indent=2))
    return sum(len(inner) for _size, inner in out_groups)


def main() -> int:
    args = tyro.cli(Args, description=__doc__)

    exit_if_missing(args.goal_binary)

    goal_terms_path = args.path / "goal_terms.json"
    goal_args_path = args.path / "goal_args.json"
    if not args.force:
        existing = [p for p in (goal_terms_path, goal_args_path) if p.exists()]
        if existing:
            print(
                f"Refusing to overwrite {', '.join(str(p) for p in existing)}; "
                "pass --force to allow it.",
                file=sys.stderr,
            )
            return 2

    seeds = flatten_seeds(args)
    if not seeds:
        print("No seeds to process after filtering.", file=sys.stderr)
        return 0

    raw_cfg = json.loads((args.path / "generation_args.json").read_text())
    cfg = {"language": raw_cfg["language"], **eqsat_limits(raw_cfg)}
    # goal_args.json carries the eqsat config the goals were generated under
    # (top-level, where guided_search.py reads it) plus this run's CLI args under
    # `driver_args`, so downstream stages never reach back into
    # generation_args.json.
    goal_args_path.write_text(
        json.dumps({**cfg, "driver_args": dataclasses.asdict(args)}, indent=2, default=str)
    )

    enriched = run_all_seeds(args, language_eqsat_flags(cfg), seeds)

    written = write_enriched_terms(args.path / "terms.json", goal_terms_path, enriched)
    print(
        f"\nWrote {written} enriched seed(s) to {goal_terms_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
