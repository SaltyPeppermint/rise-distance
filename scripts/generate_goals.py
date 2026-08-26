"""Drive goal-term generation from Python.

Runs the file-free Rust ``goal`` binary once per selected seed and writes its
payloads to ``goal_terms.json`` without modifying ``terms.json``. The
``goal_args.json`` sidecar records the governing eqsat configuration and this
driver's arguments.

Example:
    cargo build --release --bin goal
    uv run scripts/generate_goals.py data/start_terms/dusky-cramp \\
        --goal-terms 10 --jobs 8
"""

import dataclasses
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from tqdm import tqdm

from common import eqsat_limits, exit_if_missing, language_eqsat_flags, run_json_subprocess

GoalSelectionPolicy = Literal[
    "count",
    "naive",
]


@dataclass
class Args:
    # I/O
    path: tyro.conf.Positional[Path]
    """Folder with `terms.json` and `generation_args.json`. The enriched
    per-seed goal data is written to `goal_terms.json` in the same folder,
    alongside a `goal_args.json` sidecar."""

    force: bool = False
    """Overwrite existing `goal_terms.json`/`goal_args.json` in the start
    folder instead of refusing."""

    goal_binary: Path = Path("target/release/goal")

    # goal generation
    n: int = 10
    """Number of goal candidates to draw per start term."""

    size_allocation: str | None = None
    """How to allocate the candidate budget across sizes (forwarded if set)."""

    policy: GoalSelectionPolicy | None = None
    """How to draw goal candidates: count or naive (forwarded if set)."""

    skip_unmeasured: bool = True
    """Skip start terms whose measurement is missing or empty (no iterations were
    recorded for the start term during generation)."""

    retry_step: int = 5
    """How much to grow `max_size` on each exact-size-search retry."""

    max_retries: int = 20
    """How many exact-size-search increments to allow."""

    size_goal: int = 5
    """How many novel root sizes exact construction must find."""

    # fan-out
    cutoff: int | None = None
    """Only process the first N start terms (sorted order, so stable across
    runs). All start terms if omitted."""

    jobs: int | None = None
    """Max concurrent `goal` subprocesses (one seed each). Defaults to
    `os.cpu_count()`. Lower it if the per-seed eqsat egraphs exhaust RAM, since
    each start term's eqsat can use several GB."""


def is_measured(record: object) -> bool:
    """True iff the record's measurement (`record[2]`, an `{"iterations"}`
    object) is present and non-empty. A kept term always ran at least one
    eqsat iteration during generation, so its measurement is non-empty; an
    empty list marks a term with no recorded iterations."""
    if not (isinstance(record, list) and len(record) > 2):
        return False
    measurement = record[2]
    if not isinstance(measurement, dict):
        return False
    iterations = measurement.get("iterations")
    return isinstance(iterations, list) and len(iterations) > 0


def flatten_start_terms(args: Args) -> list[str]:
    """Flatten `terms.json` into the list of seed s-expressions to process.

    Groups in file order; terms sorted within each group for deterministic
    `--n`; then the `skip_unmeasured` filter.
    """
    groups = json.loads((args.path / "terms.json").read_text())
    start_terms: list[str] = []
    for _size, terms_map in groups:
        for term in sorted(terms_map):
            if args.skip_unmeasured and not is_measured(terms_map[term]):
                continue
            start_terms.append(term)
    if args.cutoff is not None:
        start_terms = start_terms[: args.cutoff]
    return start_terms


def run_goal_shard(args: Args, base_flags: list[str], start_term: str) -> object:
    """Run `goal` for a single seed and return its parsed stdout payload."""
    cmd = [
        str(args.goal_binary),
        "--start-term",
        start_term,
        *base_flags,
        "--n",
        str(args.n),
        "--retry-step",
        str(args.retry_step),
        "--max-retries",
        str(args.max_retries),
        "--size-goal",
        str(args.size_goal),
    ]
    if args.size_allocation is not None:
        cmd += ["--size-allocation", args.size_allocation]
    if args.policy is not None:
        cmd += ["--exact-selection-policy", args.policy]
    return run_json_subprocess(cmd, what=f"goal for start term {start_term!r}")


def run_all_start_terms(
    args: Args, base_flags: list[str], start_terms: list[str]
) -> dict[str, object]:
    """Run one `goal` subprocess per seed and collect `{start_str: payload}`."""
    jobs = args.jobs or os.cpu_count() or 1
    print(
        f"Generating goals for {len(start_terms)} start term(s) ({jobs} workers)",
        file=sys.stderr,
    )

    enriched: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_goal_shard, args, base_flags, seed): seed for seed in start_terms
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="start terms", unit="term"):
            enriched[futures[fut]] = fut.result()
    return enriched


def write_enriched_terms(src: Path, dst: Path, enriched: dict[str, object]) -> int:
    """Write the enriched copy of `src` (the seed `terms.json`) to `dst`,
    replacing each seed's record with its enriched payload. Preserves the
    `[size, {term: payload}]` grouping; start terms absent from `enriched` (e.g.
    dropped by `--n`) are omitted, and groups left empty are dropped.
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

    start_terms = flatten_start_terms(args)
    if not start_terms:
        print("No start terms to process after filtering.", file=sys.stderr)
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

    enriched = run_all_start_terms(args, language_eqsat_flags(cfg), start_terms)

    written = write_enriched_terms(args.path / "terms.json", goal_terms_path, enriched)
    print(
        f"\nWrote {written} enriched start terms(s) to {goal_terms_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
