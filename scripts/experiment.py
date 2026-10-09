"""Run a whole experiment: build, compute the shared baseline, then the guided searches.

uv run scripts/experiment.py
uv run scripts/experiment.py --rerun-aborted [data/guided_search]
"""

import argparse
import asyncio
import itertools
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from baseline import BaselineArgs, compute_baseline
from common import LOG_FILE, PrioritySemaphore, PrioritySlot, load_pairs, log
from generate_problems import GenerateArgs, generate
from guided_search import SearchArgs, search

PROBLEMS = Path("data/problems/expensive-bird")
OUTPUT_BASE = Path("data/guided_search")
BASELINE_BASE = Path("data/baselines")

# What `guided_search.py` writes once the search finished.
RESULT_FILES = (
    "attempts.parquet",
    "attempts.json",
    "expansions.parquet",
    "pairs.parquet",
    "pools.json",
)

# Concurrent `sample`/`attempt` processes over all runs together.
JOBS = os.cpu_count() or 1

# Shared by `baseline.py` and `guided_search.py`, so both compute the same baseline.
BASELINE_FLAGS: dict[str, Any] = {"max_rss": "450M", "n_starts": 100}

BASE_FLAGS: dict[str, Any] = {
    **BASELINE_FLAGS,
    "full_union": True,
    "sampling_backoff": 50,
}

# Search stops at whatever is hit earlier
BUDGETS: list[dict[str, Any]] = [
    {"max_attempts": 30, "max_pair_time": 60},
]

GRID: dict[str, list[Any]] = {
    "seed": [123, 456],
    "sample_policy": ["uniform", "count"],
    "branching": [10, 30],
    "search_policy": ["dfs", "bfs"],
    "frontier": [True, False],
}


def git_short_hash() -> str:
    """Short HEAD hash, suffixed with `-dirty` if there are uncommitted changes"""
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode != 0
    return f"{rev}-dirty" if dirty else rev


async def generate_problems(problems: Path = PROBLEMS) -> None:
    args = GenerateArgs(
        starts=1000,
        min_size=30,
        max_size=60,
        language="math",
        seed=123,
        max_iters=2000,
        max_nodes=1000000,
        max_time=300,
        max_memory="500M",
        min_rss="500M",
        max_rss="1G",
        goals=2,
        output=problems,
    )
    print(await generate(args, PrioritySemaphore(20).at(0)), file=sys.stderr)


def run_number(path: Path) -> int:
    """The leading number of a run folder's name, or 0 if it has none."""
    return int(m.group()) if (m := re.match(r"\d+", path.name)) else 0


def new_run_dir(suffix: str, run: int | None = None) -> Path:
    """Create the folder `<run>_<suffix>` in `OUTPUT_BASE`, numbered after the last run by default."""
    if run is None:
        run = max(map(run_number, OUTPUT_BASE.iterdir()), default=0) + 1
    out_dir = OUTPUT_BASE / f"{run}_{suffix}"
    out_dir.mkdir()
    return out_dir


async def run_guided_search(args: SearchArgs, limit: PrioritySlot, label: str) -> None:
    """Run one guided search, logging to `experiment.log` in its folder.

    A failed run is only warned about, so it does not cancel the others.
    """
    with (args.output / "experiment.log").open("w", buffering=1) as log_file:
        # Scoped to this run's task and its subtasks.
        LOG_FILE.set(log_file)
        flags = args.model_dump()
        width = max(map(len, flags))
        log("FLAGS:\n" + "\n".join(f"    {k:<{width}} : {v}" for k, v in flags.items()))
        try:
            summary = await search(args, limit, desc=label, leave=False)
        except Exception as e:  # noqa: BLE001
            log(traceback.format_exc())
            tqdm.write(f"WARNING: run {label} failed: {e!r}", file=sys.stderr)
            return
    tqdm.write(f"{label}: {summary}", file=sys.stderr)


async def run_all(runs: list[SearchArgs], slots: PrioritySemaphore) -> None:
    """Run every guided search at once, earlier runs taking precedence for `slots`."""
    # Each run is labelled by its number plus the flags that differ between runs.
    dumps = [args.model_dump(exclude={"output"}) for args in runs]
    distinct = [key for key in dumps[0] if len({repr(dump[key]) for dump in dumps}) > 1]

    problems = {(args.path, args.n_starts, args.n_goals) for args in runs}
    counts = {len(load_pairs(*problem)) for problem in problems}
    pairs = str(min(counts)) if len(counts) == 1 else f"{min(counts)}-{max(counts)}"
    varying = f", varying {', '.join(distinct)}" if distinct else ""
    print(
        f"Running {len(runs)} guided search(es) over {pairs} pair(s) "
        f"on {JOBS} process slots{varying}",
        file=sys.stderr,
    )
    async with asyncio.TaskGroup() as group:
        for priority, (args, dump) in enumerate(zip(runs, dumps, strict=True)):
            label = " ".join(
                [str(run_number(args.output)), *(f"{key}={dump[key]}" for key in distinct)]
            )
            group.create_task(run_guided_search(args, slots.at(priority), label))


async def rerun_aborted(base: Path, slots: PrioritySemaphore) -> None:
    """Re-run every aborted run in `base` in place, with the flags its `config.json` holds.

    A run is aborted if it started (wrote `config.json`) but did not write all its results.
    """
    run_dirs = sorted(
        (
            p
            for p in base.iterdir()
            if (p / "config.json").is_file()
            and not all((p / name).is_file() for name in RESULT_FILES)
        ),
        key=run_number,
    )
    print(f"Re-running {len(run_dirs)} aborted run(s) in {base}")
    if not run_dirs:
        return

    runs = []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        # Derived from the other fields.
        config.pop("effective_limits", None)
        # In case the run was moved since.
        config["output"] = run_dir
        runs.append(SearchArgs(**config))
    await run_all(runs, slots)


async def experiment(slots: PrioritySemaphore) -> None:
    """Generate the problems if asked to, compute the baseline, then run the grid."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    suffix = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M}_{git_short_hash()}"

    # PROBLEM GENERATION
    # await generate_problems()

    # BASELINE, shared by every run below
    baseline = BASELINE_BASE / PROBLEMS.name
    baseline_args = BaselineArgs(path=PROBLEMS, output=baseline, **BASELINE_FLAGS)
    print(await compute_baseline(baseline_args, slots.at(0)), file=sys.stderr)

    # GRID SEARCH
    # Validated before any run folder is created.
    runs = [
        SearchArgs(
            **{**BASE_FLAGS, **budget, **dict(zip(GRID, values))},
            path=PROBLEMS,
            baseline=baseline,
            output=OUTPUT_BASE,
        )
        for budget, values in itertools.product(BUDGETS, itertools.product(*GRID.values()))
    ]
    runs = [args.model_copy(update={"output": new_run_dir(suffix)}) for args in runs]

    # # INDIVIDUAL RUN(s)
    # runs += [
    #     SearchArgs(
    #         **BASE_FLAGS,
    #         **BUDGETS[0],
    #         seed=456,
    #         search_policy="bfs",
    #         frontier=True,
    #         path=PROBLEMS,
    #         baseline=baseline,
    #         output=new_run_dir(suffix, run=18),
    #     )
    # ]

    await run_all(runs, slots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun-aborted",
        type=Path,
        nargs="?",
        const=OUTPUT_BASE,
        metavar="DIR",
        help=f"Only re-run the aborted runs in DIR (default: {OUTPUT_BASE}), then exit.",
    )
    cli = parser.parse_args()

    subprocess.run(["cargo", "build", "--release"], check=True)
    slots = PrioritySemaphore(JOBS)

    if cli.rerun_aborted is not None:
        asyncio.run(rerun_aborted(cli.rerun_aborted, slots))
    else:
        asyncio.run(experiment(slots))


if __name__ == "__main__":
    main()
