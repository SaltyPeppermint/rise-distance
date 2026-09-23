#!/usr/bin/env -S uv run --script
import itertools
import subprocess

MEMRUN = [
    "systemd-run",
    "--user",
    "--wait",
    "--pipe",
    "--same-dir",
    "--property=MemoryAccounting=yes",
]

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
}


def flag_args(flag: str, value: object) -> list[str]:
    """`True`/`False` become `--flag`/`--no-flag` but everything else is untouched"""
    if value is True:
        return [flag]
    if value is False:
        return [f"--no-{flag.removeprefix('--')}"]
    return [flag, str(value)]


subprocess.run(["cargo", "build", "--release"], check=True)

for i, values in enumerate(itertools.product(*GRID.values()), start=1):
    grid_args = [arg for flag, value in zip(GRID, values) for arg in flag_args(flag, value)]
    proc = subprocess.Popen(
        [
            *MEMRUN,
            "uv", "run", "scripts/guided_search.py",
            *BASE_ARGS,
            *grid_args,
            "--output", f"data/guided_search/{i}",
            "data/problems/expensive-bird",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )  # fmt: skip
    tee = subprocess.Popen(["tee", f"experiment_{i}.log"], stdin=proc.stdout)
    assert proc.stdout is not None
    proc.stdout.close()  # so proc gets SIGPIPE if tee exits early
    tee.wait()
    proc.wait()
