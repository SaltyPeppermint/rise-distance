#!/usr/bin/env python3
import itertools
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
    "guide-random",
    "--",
    "--seed",
    "(d x (- (pow x 3) (* 7 (pow x 2))))",
    "-n",
    "10",
    "-i",
    "5",
    "--max-size",
    "20",
    "--goals",
    "100",
    "--guides",
    "10000",
    "--eval-all",
]

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
    subprocess.run(base_cmd + extra, check=True)
