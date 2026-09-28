#!/usr/bin/env -S uv run --script
import itertools
import re
import subprocess
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

BASE_ARGS = [
    "--max-rss", "450M",
    "--sample-policy", "uniform",
    "--start-terms", "100",
    "--n-guides", "10",
    "--max-attempts", "30",
    "--seed", "42",
    "--full-union",
]  # fmt: skip

GRID = {
    "--sampling-backoff": [5, 20],
    "--max-depth": [1, 2],
    "--search-policy": ["depth", "width"],
    "--frontier": [True, False],
}


def flag_args(flag: str, value: object) -> list[str]:
    """`True`/`False` become `--flag`/`--no-flag` but everything else is untouched"""
    if value is True:
        return [flag]
    if value is False:
        return [f"--no-{flag.removeprefix('--')}"]
    return [flag, str(value)]


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


subprocess.run(["cargo", "build", "--release"], check=True)

# subprocess.run(
#     [
#         *MEMRUN,
#         "uv", "run", "scripts/generate_problems.py",
#         "--starts", "1000",
#         "--min-size", "30",
#         "--max-size", "60",
#         "--language", "math",
#         "--seed", "123",
#         "--jobs", "20",
#         "--max-iters", "2000",
#         "--max-nodes", "1000000",
#         "--max-time", "300",
#         "--max-memory", "500M",
#         "--min-rss", "500M",
#         "--rss-max", "1G",
#         "--goals", "2",
#         "--path", "data/problems/expensive-bird",
#     ],
#     check=True,
# )

OUTPUT_BASE = Path("data/guided_search")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
RUN_SUFFIX = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M}_{git_short_hash()}"
FIRST_RUN = next_run_number(OUTPUT_BASE)

for i, values in enumerate(itertools.product(*GRID.values()), start=FIRST_RUN):
    out_dir = OUTPUT_BASE / f"{i}_{RUN_SUFFIX}"
    out_dir.mkdir()
    grid_args = [arg for flag, value in zip(GRID, values) for arg in flag_args(flag, value)]
    print(f"GRID ARGS: {grid_args}")
    proc = subprocess.Popen(
        [
            *MEMRUN,
            "uv", "run", "scripts/guided_search.py",
            *BASE_ARGS,
            *grid_args,
            "--output", str(out_dir),
            "data/problems/expensive-bird",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )  # fmt: skip
    tee = subprocess.Popen(["tee", out_dir / "experiment.log"], stdin=proc.stdout)
    assert proc.stdout is not None
    proc.stdout.close()  # so proc gets SIGPIPE if tee exits early
    tee.wait()
    proc.wait()
