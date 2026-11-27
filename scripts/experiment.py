"""Run a whole experiment under benchmode: build, compute the shared baseline, then the guided searches.

uv run scripts/experiment.py [--language math|prop]
uv run scripts/experiment.py --sequential
uv run scripts/experiment.py --rerun-aborted [DIR]
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
from benchmode import benchmode_cgroup, benchmode_cpus, elevate
from common import LOG_FILE, PrioritySemaphore, PrioritySlot, load_pairs, log, problem_language
from generate_problems import GenerateArgs, generate
from guided_search import RESULT_FILES, SearchArgs, search

# The experiment (PROBLEMS appends `-<language>`; OUTPUT_BASE uses `<language>` subfolders).
PROBLEMS = Path("data/problems/expensive-bird")

# Shared by `baseline.py` and `guided_search.py`, so both compute the same baseline.
BASELINE_FLAGS: dict[str, Any] = {"max_rss": "450M", "n_starts": 100}

BASE_FLAGS: dict[str, Any] = {
    **BASELINE_FLAGS,
    "sampling_retries": 50,
    # "seed": 456,
    # "frontier": True,
    # "sample_policy": "count",
    "tree_width": 30,
    "full_union": False,
    "guides_per_attempt": 100,
    # "search_policy": "dfs",
    "sampling_backoff": "linear",
}

# Search stops at whatever is hit earlier
BUDGETS: tuple[dict[str, Any], ...] = (
    # {"max_pair_time": 120},
    {"max_pair_time": 300},
    # {"max_pair_time": 300, "max_attempt_time": 10},
    # {"max_attempts": 60},
)

GRID: dict[str, tuple[Any, ...]] = {
    "seed": (123, 456),
    "sample_policy": ("uniform", "count"),
    # "tree_width": (2, 6),
    # "sampling_backoff": ("gallop", "linear"),
    "search_policy": ("dfs", "bfs"),
    # "full_union": (True, False),
    # "guides_per_attempt": (1,2,5),
    "frontier": (True, False),
}


# Where results go
OUTPUT_BASE = Path("data/guided_search")
BASELINE_BASE = Path("data/baselines")


def paths_for_language(language: str) -> tuple[Path, Path]:
    """Return `(problems_dir, output_base)` for `language`."""
    return (
        PROBLEMS.with_name(f"{PROBLEMS.name}-{language}"),
        OUTPUT_BASE / language,
    )


def git_short_hash() -> str:
    """Short HEAD hash, suffixed with `-dirty` if there are uncommitted changes"""
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode != 0
    return f"{rev}-dirty" if dirty else rev


def run_number(path: Path) -> int:
    """The leading number of a run folder's name, or 0 if it has none."""
    return int(m.group()) if (m := re.match(r"\d+", path.name)) else 0


def new_run_dir(output_base: Path, suffix: str, run: int | None = None) -> Path:
    """Create the folder `<run>_<suffix>` in `output_base`, numbered after the last run by default."""
    if run is None:
        run = max(map(run_number, output_base.iterdir()), default=0) + 1
    out_dir = output_base / f"{run}_{suffix}"
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

    problems = {(args.path, args.n_starts) for args in runs}
    counts = {len(load_pairs(*problem)) for problem in problems}
    pairs = str(min(counts)) if len(counts) == 1 else f"{min(counts)}-{max(counts)}"
    varying = f", varying {', '.join(distinct)}" if distinct else ""
    cpus = ",".join(map(str, slots.cpus))
    print(
        f"Running {len(runs)} guided search(es) over {pairs} pair(s) "
        f"on {len(slots.cpus)} isolated core(s) ({cpus}){varying}",
        file=sys.stderr,
    )
    # Padded per column, so the labels (and with them the bars) line up.
    cells = [
        [str(run_number(args.output)), *(f"{key}={dump[key]}" for key in distinct)]
        for args, dump in zip(runs, dumps, strict=True)
    ]
    widths = [max(map(len, column)) for column in zip(*cells, strict=True)]
    labels = [
        " ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in cells
    ]
    async with asyncio.TaskGroup() as group:
        for priority, (args, label) in enumerate(zip(runs, labels, strict=True)):
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
        # In case the run was moved since.
        config["output"] = run_dir
        runs.append(SearchArgs(**config))
    await run_all(runs, slots)


async def generate_problems(slots: PrioritySemaphore, language: str, problems: Path) -> None:
    args = GenerateArgs(
        starts=1000,
        min_size=30,
        max_size=60,
        language=language,
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
    print(await generate(args, slots.at(0)), file=sys.stderr)


async def experiment(slots: PrioritySemaphore, language: str) -> None:
    """Generate problems if missing, compute the baseline, then run the grid."""
    problems, output_base = paths_for_language(language)
    output_base.mkdir(parents=True, exist_ok=True)
    suffix = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M}_{git_short_hash()}"

    # PROBLEM GENERATION (automatic when the problem folder does not yet exist)
    if (problems / "problems.json").is_file():
        actual = problem_language(problems)
        if actual != language:
            raise ValueError(
                f"{problems} holds {actual!r} problems, expected {language!r}"
            )
    else:
        await generate_problems(slots, language, problems)

    # BASELINE, shared by every run below
    baseline = BASELINE_BASE / problems.name
    baseline_args = BaselineArgs(path=problems, output=baseline, **BASELINE_FLAGS)
    print(await compute_baseline(baseline_args, slots.at(0)), file=sys.stderr)

    # GRID SEARCH
    # Validated before any run folder is created.
    runs = [
        SearchArgs(
            **{**BASE_FLAGS, **budget, **dict(zip(GRID, values))},
            path=problems,
            baseline=baseline,
            output=output_base,
        )
        for budget, values in itertools.product(BUDGETS, itertools.product(*GRID.values()))
    ]
    runs = [args.model_copy(update={"output": new_run_dir(output_base, suffix)}) for args in runs]

    # # INDIVIDUAL RUN(s)
    # runs += [
    #     SearchArgs(
    #         **BASE_FLAGS,
    #         **BUDGETS[0],
    #         seed=456,
    #         search_policy="bfs",
    #         frontier=True,
    #         path=problems,
    #         baseline=baseline,
    #         output=new_run_dir(output_base, suffix, run=18),
    #     )
    # ]

    await run_all(runs, slots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        default="math",
        help="Language to run the experiment for (e.g. math, prop; default: math).",
    )
    parser.add_argument(
        "--rerun-aborted",
        type=Path,
        nargs="?",
        const=Path(),
        metavar="DIR",
        help="Only re-run the aborted runs in DIR (default: the language's output folder), then exit.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run on a single isolated core instead of all isolated physical cores.",
    )
    cli = parser.parse_args()

    if os.environ.get("BENCHMODE") != "1":
        # Build before disabling SMT so cargo can use all hardware threads.
        subprocess.run(["cargo", "build", "--release"], check=True)
        raise SystemExit(elevate([sys.executable, *sys.argv], sequential=cli.sequential))

    slots = PrioritySemaphore(benchmode_cpus(), benchmode_cgroup())

    if cli.rerun_aborted is not None:
        _, output_base = paths_for_language(cli.language)
        rerun_dir = output_base if cli.rerun_aborted == Path() else cli.rerun_aborted
        asyncio.run(rerun_aborted(rerun_dir, slots))
    else:
        asyncio.run(experiment(slots, cli.language))


if __name__ == "__main__":
    main()
