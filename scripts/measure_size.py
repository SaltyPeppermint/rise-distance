"""Measure per-iteration eqsat stats and live-heap use on each seed term.

Spawns the `measure-size` Rust binary once per term in a seed-terms
`terms.json` (as produced by `scripts/generate_seeds.py`). The binary
runs eqsat and prints a JSON object `{"iterations": [<egg per-iteration
stats, each with an `allocated` field>, ...]}` on stdout; `allocated` is
jemalloc's live-heap bytes (`stats.allocated`) sampled at the start of
each iteration. Memory is bounded by egg's graceful `--max-memory`
per-iteration hook.

The seed-terms file has shape
`[[size, {term: [attempts, validation, {iterations}?]}], ...]`.
The measured `{iterations}` object is written back in place as the 3rd
entry of each inner list, overwriting any existing value. On any failure
(non-zero exit, timeout, unparseable output), the value is
`{"iterations": []}`.

Example:
    cargo build --release --bin measure-size
    uv run scripts/measure_size.py --path data/seed_terms/foo-bar/terms.json \\
        --language math --max-time 30
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from common import exit_if_missing, measure_term, subprocess_timeout


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

    timeout = subprocess_timeout(args.max_time)
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
    ]
    if args.backoff_scheduler:
        cmd_base.append("--backoff-scheduler")

    flat: list[tuple[int, str]] = [
        (size_idx, term)
        for size_idx, (_size, terms_map) in enumerate(big_collector)
        for term in terms_map
    ]

    for size_idx, term in tqdm(flat, desc="terms"):
        stats = measure_term([*cmd_base, "--term", term], timeout=timeout)

        entry = big_collector[size_idx][1][term]
        if len(entry) >= 3:
            entry[2] = stats
        else:
            entry.append(stats)

    with args.path.open("w") as f:
        json.dump(big_collector, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
