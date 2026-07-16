"""Measure peak resident set size of running eqsat on each seed term.

Spawns the `measure-size` Rust binary once per term in a seed-terms
`terms.json` (as produced by `scripts/generate_seeds.py`). The
binary reads its own peak RSS from `/proc/self/status` (VmHWM) at exit
and prints the value (bytes) on stdout -> this matches what htop
reports. A virtual-memory cap is enforced inside the binary via
RLIMIT_AS as a safety net against runaways.

The seed-terms file has shape
`[[size, {term: [attempts, validation, peak_memory_bytes?]}], ...]`.
The measured peak is written back in place as the 3rd entry of each
inner list, overwriting any existing value. On any failure (non-zero
exit, RLIMIT kill, timeout, unparseable output), the value is -1.

Example:
    cargo build --release --bin measure-size
    uv run scripts/measure_size.py --path data/seed_terms/foo-bar/terms.json \\
        --language math --rlimit-as 8G --max-time 30
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from common import exit_if_missing, parse_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Seed-terms JSON (e.g. data/seed_terms/foo-bar/terms.json). Edited in place.",
    )
    parser.add_argument("--binary", type=Path, default=Path("target/release/measure-size"))
    parser.add_argument(
        "--language",
        required=True,
        help="Language the terms are drawn from (e.g. math, prop).",
    )
    parser.add_argument(
        "--rlimit-as",
        type=parse_size,
        required=True,
        help="Per-term virtual-memory cap (e.g. 8G), enforced via RLIMIT_AS. Backstop only.",
    )
    parser.add_argument("--max-iters", type=int, default=11)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument(
        "--backoff-scheduler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use egg's BackoffScheduler instead of the SimpleScheduler.",
    )
    args = parser.parse_args()

    exit_if_missing(args.binary)

    with args.path.open("r") as f:
        big_collector = json.load(f)

    timeout = max(1, int(args.max_time * 4) + 5)
    cmd_base = [
        str(args.binary),
        "--language",
        args.language,
        "--max-iters",
        str(args.max_iters),
        "--max-nodes",
        str(args.max_nodes),
        "--max-time",
        str(args.max_time),
        "--rlimit-as",
        str(args.rlimit_as),
    ]
    if args.backoff_scheduler:
        cmd_base.append("--backoff-scheduler")

    flat: list[tuple[int, str]] = [
        (size_idx, term)
        for size_idx, (_size, terms_map) in enumerate(big_collector)
        for term in terms_map
    ]

    for size_idx, term in tqdm(flat, desc="terms"):
        try:
            proc = subprocess.run(
                [*cmd_base, "--term", term],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            peak = int(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else -1
        except subprocess.TimeoutExpired, ValueError, IndexError:
            peak = -1

        entry = big_collector[size_idx][1][term]
        if len(entry) >= 3:
            entry[2] = peak
        else:
            entry.append(peak)

    with args.path.open("w") as f:
        json.dump(big_collector, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
