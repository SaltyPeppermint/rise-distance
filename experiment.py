#!/usr/bin/env -S uv run --script
import itertools
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

MEMRUN = [
    "systemd-run",
    "--user",
    "--wait",
    "--pipe",
    "--same-dir",
    "--property=MemoryAccounting=yes",
]

PROBLEMS = Path("data/problems/expensive-bird")
OUTPUT_BASE = Path("data/guided_search")

BASE_FLAGS: dict[str, object] = {
    "--max-rss": "450M",
    "--sample-policy": "uniform",
    "--start-terms": 100,
    "--branching": 10,
    "--max-attempts": 30,
    "--seed": 42,
    "--full-union": True,
}

GRID = {
    "--sampling-backoff": [5, 20],
    "--max-depth": [1, 2],
    "--search-policy": ["dfs", "bfs"],
    "--frontier": [True, False],
}


def flag_args(flag: str, value: object) -> list[str]:
    """`True`/`False` become `--flag`/`--no-flag` but everything else is untouched"""
    if value is True:
        return [flag]
    if value is False:
        return [f"--no-{flag.removeprefix('--')}"]
    return [flag, str(value)]


def flags_to_args(flags: Mapping[str, object]) -> list[str]:
    return [arg for flag, value in flags.items() for arg in flag_args(flag, value)]


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


def generate_problems(problems: Path = PROBLEMS) -> None:
    flags = {
        "--starts": 1000,
        "--min-size": 30,
        "--max-size": 60,
        "--language": "math",
        "--seed": 123,
        "--jobs": 20,
        "--max-iters": 2000,
        "--max-nodes": 1000000,
        "--max-time": 300,
        "--max-memory": "500M",
        "--min-rss": "500M",
        "--rss-max": "1G",
        "--goals": 2,
        "--path": problems,
    }
    subprocess.run(
        [*MEMRUN, "uv", "run", "scripts/generate_problems.py", *flags_to_args(flags)], check=True
    )


def run_guided_search(
    flags: dict[str, object], suffix: str, problems: Path = PROBLEMS, run: int | None = None
) -> None:
    if run is None:
        run = next_run_number(OUTPUT_BASE)
    out_dir = OUTPUT_BASE / f"{run}_{suffix}"
    out_dir.mkdir()
    print(f"RUN {run} FLAGS: {flags}")
    proc = subprocess.Popen(
        [
            *MEMRUN,
            "uv",
            "run",
            "scripts/guided_search.py",
            *flags_to_args({**BASE_FLAGS, **flags}),
            "--output",
            str(out_dir),
            str(problems),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tee = subprocess.Popen(["tee", out_dir / "experiment.log"], stdin=proc.stdout)
    assert proc.stdout is not None
    proc.stdout.close()  # so proc gets SIGPIPE if tee exits early
    tee.wait()
    proc.wait()
    if proc.returncode != 0:
        print(f"WARNING: run {run} exited with code {proc.returncode}")


def main() -> None:
    subprocess.run(["cargo", "build", "--release"], check=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    suffix = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M}_{git_short_hash()}"

    # PROBLEM GENERATION
    # generate_problems()

    # GRID SEARCH
    # for values in itertools.product(*GRID.values()):
    #     run_guided_search(dict(zip(GRID, values)), suffix)

    # INDIVIDUAL RUN(s)
    run_guided_search(
        {
            "--sampling-backoff": 50,
            "--max-depth": 2,
            "--search-policy": "bfs",
            "--frontier": True,
        },
        suffix,
    )


if __name__ == "__main__":
    main()
