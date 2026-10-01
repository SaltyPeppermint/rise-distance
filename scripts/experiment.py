"""Run a whole experiment: build, compute the shared baseline, then the guided searches.

uv run scripts/experiment.py
"""

import itertools
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from common import cli_flags

PROBLEMS = Path("data/problems/expensive-bird")
OUTPUT_BASE = Path("data/guided_search")
BASELINE_BASE = Path("data/baselines")

# Shared by `baseline.py` and `guided_search.py`, so both compute the same baseline.
BASELINE_FLAGS = {"max_rss": "450M", "n_starts": 100}

BASE_FLAGS = {
    **BASELINE_FLAGS,
    "max_attempts": 30,
    "seed": 123,
    "full_union": True,
    "sampling_backoff": 50,
}

GRID = {
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


def run_guided_search(
    flags: dict[str, object],
    suffix: str,
    baseline: Path,
    problems: Path = PROBLEMS,
    run: int | None = None,
) -> None:
    if run is None:
        run = next_run_number(OUTPUT_BASE)
    out_dir = OUTPUT_BASE / f"{run}_{suffix}"
    out_dir.mkdir()
    print(f"RUN {run} FLAGS: {flags}")
    returncode = run_driver(
        "guided_search.py",
        {**BASE_FLAGS, **flags, "baseline": baseline, "output": out_dir},
        problems,
        log=out_dir / "experiment.log",
        check=False,
    )
    if returncode != 0:
        print(f"WARNING: run {run} exited with code {returncode}")


def main() -> None:
    subprocess.run(["cargo", "build", "--release"], check=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    suffix = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M}_{git_short_hash()}"

    # PROBLEM GENERATION
    # generate_problems()

    # BASELINE, shared by every run below
    baseline = run_baseline()

    # GRID SEARCH
    for values in itertools.product(*GRID.values()):
        run_guided_search(dict(zip(GRID, values)), suffix, baseline)

    # INDIVIDUAL RUN(s)
    run_guided_search(
        {
            "sampling_backoff": 50,
            "search_policy": "bfs",
            "frontier": True,
        },
        suffix,
        baseline,
        run=18,
    )


if __name__ == "__main__":
    main()
