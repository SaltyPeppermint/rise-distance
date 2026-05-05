#!/usr/bin/env python3
import itertools
import os
import subprocess

params = {
    "--guide-sample-strategy": ["naive", "count-based"],
    "--full-union": [False, True],
}

base_cmd = [
    "cargo",
    "run",
    "--release",
    "--bin",
    "guide",
    "--",
    "--seed-json",
    "data/seed_terms/output.json",
    "--goals",
    "10",
    "--guides",
    "1000",
    "--time-limit",
    "0.2",
]

env = os.environ | {"RUST_BACKTRACE": "1"}

for i, combo in enumerate(itertools.product(*params.values())):
    extra = []
    for flag, val in zip(params.keys(), combo):
        if isinstance(val, bool):
            if val:
                extra.append(flag)
        else:
            extra += [flag, val]
    print("\n\n")
    print("=" * 32)
    print(f"RUN {i}")
    print(f"EXTRA ARGS:   {extra}\n")
    print(f"FULL COMMAND: {' '.join(base_cmd + extra)}")
    subprocess.run(base_cmd + extra, check=True, env=env)
