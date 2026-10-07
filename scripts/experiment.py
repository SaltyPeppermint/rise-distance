"""Run a whole experiment: build, compute the shared baseline, then the guided searches.

uv run scripts/experiment.py
uv run scripts/experiment.py --rerun-aborted [data/guided_search]
"""

import argparse
import itertools
import json
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from common import cli_flags

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

# Shared by `baseline.py` and `guided_search.py`, so both compute the same baseline.
BASELINE_FLAGS = {"max_rss": "450M", "n_starts": 100}

BASE_FLAGS = {
    **BASELINE_FLAGS,
    "full_union": True,
    "sampling_backoff": 50,
}

# Mutually exclusive
BUDGETS = [{"max_attempts": 30}, {"max_pair_time": 60}]

GRID = {
    "seed": [123, 456],
    "sample_policy": ["uniform", "count"],
    "branching": [10, 30],
    "search_policy": ["dfs", "bfs"],
    "frontier": [True, False],
}


# Runs a whole driver as a transient service, so its memory is accounted for as one unit.
MEMRUN = [
    "systemd-run",
    "--user",
    "--wait",
    "--pipe",
    "--same-dir",
    "--property=MemoryAccounting=yes",
]


def git_short_hash() -> str:
    """Short HEAD hash, suffixed with `-dirty` if there are uncommitted changes"""
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode != 0
    return f"{rev}-dirty" if dirty else rev


def next_run_number(base: Path) -> int:
    """One past the highest leading number of any entry in `base`"""
    nums = [int(m.group()) for p in base.glob("*") if (m := re.match(r"\d+", p.name))]
    return max(nums, default=0) + 1


def run_driver(
    script: str,
    flags: Mapping[str, object],
    *positional: Path,
    log: Path | None = None,
    check: bool = True,
) -> int:
    """Run the driver `script` next to this file under `MEMRUN` and return its exit code.

    With `log`, its combined stdout/stderr is also teed into that file.
    """
    cmd = [
        *MEMRUN,
        "uv",
        "run",
        str(Path(__file__).with_name(script)),
        *cli_flags(True, **flags),  # `--no-flag`, as pydantic expects
        *map(str, positional),
    ]
    if log is None:
        return subprocess.run(cmd, check=check).returncode

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tee = subprocess.Popen(["tee", log], stdin=proc.stdout)
    assert proc.stdout is not None
    proc.stdout.close()  # so proc gets SIGPIPE if tee exits early
    tee.wait()
    proc.wait()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def generate_problems(problems: Path = PROBLEMS) -> None:
    flags = {
        "starts": 1000,
        "min_size": 30,
        "max_size": 60,
        "language": "math",
        "seed": 123,
        "jobs": 20,
        "max_iters": 2000,
        "max_nodes": 1000000,
        "max_time": 300,
        "max_memory": "500M",
        "min_rss": "500M",
        "max_rss": "1G",
        "goals": 2,
        "output": problems,
    }
    run_driver("generate_problems.py", flags)


def run_baseline(problems: Path = PROBLEMS) -> Path:
    """Compute the unguided baseline once, reusing a matching one that already exists"""
    out_dir = BASELINE_BASE / problems.name
    run_driver("baseline.py", {**BASELINE_FLAGS, "output": out_dir}, problems)
    return out_dir


def new_run_dir(suffix: str, run: int | None = None) -> Path:
    """Create the folder `<run>_<suffix>` in `OUTPUT_BASE`, numbered after the last run by default."""
    if run is None:
        run = next_run_number(OUTPUT_BASE)
    out_dir = OUTPUT_BASE / f"{run}_{suffix}"
    out_dir.mkdir()
    return out_dir


def run_guided_search(
    flags: Mapping[str, object], out_dir: Path, problems: Path = PROBLEMS
) -> None:
    """Print the flags and run `guided_search.py` into `out_dir`, logging there as well."""
    flags = {**flags, "output": out_dir}
    print(f"RUN: {out_dir.name}\nFLAGS:")
    width = max(map(len, flags))
    for k, v in flags.items():
        print(f"    {k:<{width}} : {v}")

    returncode = run_driver(
        "guided_search.py", flags, problems, log=out_dir / "experiment.log", check=False
    )
    if returncode != 0:
        print(f"WARNING: run {out_dir.name} exited with code {returncode}")


def rerun_aborted(base: Path) -> None:
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
        key=lambda p: int(m.group()) if (m := re.match(r"\d+", p.name)) else 0,
    )
    print(f"Re-running {len(run_dirs)} aborted run(s) in {base}")

    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        problems = Path(config.pop("path"))
        config.pop("effective_limits", None)
        # Its `output` is replaced by the folder it lives in now, in case it was moved since.
        run_guided_search(config, run_dir, problems)


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

    if cli.rerun_aborted is not None:
        rerun_aborted(cli.rerun_aborted)
        return

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    suffix = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M}_{git_short_hash()}"

    # PROBLEM GENERATION
    # generate_problems()

    # BASELINE, shared by every run below
    baseline = run_baseline()

    # GRID SEARCH
    for budget, values in itertools.product(BUDGETS, itertools.product(*GRID.values())):
        flags = {**BASE_FLAGS, **budget, **dict(zip(GRID, values)), "baseline": baseline}
        run_guided_search(flags, new_run_dir(suffix))

    # # INDIVIDUAL RUN(s)
    # run_guided_search(
    #     {
    #         **BASE_FLAGS,
    #         **BUDGETS[0],
    #         "seed": 456,
    #         "search_policy": "bfs",
    #         "frontier": True,
    #         "baseline": baseline,
    #     },
    #     new_run_dir(suffix, run=18),
    # )


if __name__ == "__main__":
    main()
