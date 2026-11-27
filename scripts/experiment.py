"""Run a whole experiment under benchmode: build, compute the shared baseline, then the guided searches.

Re-running skips already-completed grid configs and resumes aborted ones in place.

uv run scripts/experiment.py [--language math [prop ...]]
uv run scripts/experiment.py --sequential
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

# The experiment (results go under `data/<EXPERIMENT>/<language>/{problems,baseline,guided_search}`).
DATA_BASE = Path("data")
EXPERIMENT = "expensive-bird"

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


def paths_for_language(language: str) -> tuple[Path, Path, Path]:
    """Return `(problems_dir, baseline_dir, guided_search_dir)` for `language`."""
    base = DATA_BASE / EXPERIMENT / language
    return base / "problems", base / "baseline", base / "guided_search"


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


def config_key(args: SearchArgs) -> str:
    """Canonical key for matching a run's settings independent of its output directory."""
    return json.dumps(args.model_dump(exclude={"output"}), sort_keys=True, default=str)


def existing_runs(base: Path) -> tuple[set[str], dict[str, Path]]:
    """Index existing folders in `base` by config key into completed keys and aborted paths."""
    completed: set[str] = set()
    aborted: dict[str, Path] = {}
    if not base.is_dir():
        return completed, aborted
    for run_dir in sorted(base.iterdir(), key=run_number):
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text())
            config["output"] = run_dir
            key = config_key(SearchArgs(**config))
        except ValueError as e:
            print(f"Ignoring incompatible config in {run_dir}: {e}", file=sys.stderr)
            continue
        if all((run_dir / name).is_file() for name in RESULT_FILES):
            completed.add(key)
            aborted.pop(key, None)
        elif key not in completed:
            aborted[key] = run_dir
    return completed, aborted


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
    problems, baseline, output_base = paths_for_language(language)
    output_base.mkdir(parents=True, exist_ok=True)
    suffix = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M}_{git_short_hash()}"

    # PROBLEM GENERATION (automatic when the problem folder does not yet exist)
    if (problems / "problems.json").is_file():
        actual = problem_language(problems)
        if actual != language:
            raise ValueError(f"{problems} holds {actual!r} problems, expected {language!r}")
    else:
        await generate_problems(slots, language, problems)

    # BASELINE, shared by every run below
    baseline_args = BaselineArgs(path=problems, output=baseline, **BASELINE_FLAGS)
    print(await compute_baseline(baseline_args, slots.at(0)), file=sys.stderr)

    # GRID SEARCH
    # Validated before any run folder is created; skip completed and resume aborted in place.
    desired = [
        SearchArgs(
            **{**BASE_FLAGS, **budget, **dict(zip(GRID, values))},
            path=problems,
            baseline=baseline,
            output=output_base,
        )
        for budget, values in itertools.product(BUDGETS, itertools.product(*GRID.values()))
    ]
    completed, aborted = existing_runs(output_base)
    runs: list[SearchArgs] = []
    skipped = resumed = created = 0
    for args in desired:
        key = config_key(args)
        if key in completed:
            skipped += 1
        elif key in aborted:
            runs.append(args.model_copy(update={"output": aborted[key]}))
            resumed += 1
        else:
            runs.append(args.model_copy(update={"output": new_run_dir(output_base, suffix)}))
            created += 1

    if skipped or resumed:
        print(
            f"Grid in {output_base}: {skipped} completed (skipped), "
            f"{resumed} aborted (resuming), {created} new",
            file=sys.stderr,
        )
    if not runs:
        return

    await run_all(runs, slots)


async def run_languages(slots: PrioritySemaphore, languages: list[str]) -> int:
    """Run `experiment` for each language in order, continuing past per-language failures."""
    failed: list[str] = []
    for language in dict.fromkeys(languages):
        if len(languages) > 1:
            print(f"\n=== Language: {language} ===", file=sys.stderr)
        try:
            await experiment(slots, language)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            print(f"ERROR: language {language!r} failed: {e!r}", file=sys.stderr)
            failed.append(language)
    if failed:
        print(f"Failed language(s): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        nargs="+",
        default=["math"],
        help="Language(s) to run the experiment for (e.g. math prop; default: math).",
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
    raise SystemExit(asyncio.run(run_languages(slots, cli.language)))


if __name__ == "__main__":
    main()
