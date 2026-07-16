"""Measure how extraction cost evolves per iteration of eqsat on each seed term.

Spawns the `measure-cost` Rust binary once per term in a seed-terms
`terms.json` (as produced by `scripts/generate_and_measure.py`). The
binary runs eqsat and prints a JSON object on stdout with two
per-iteration arrays: `monotonic` (cost-function name -> cheapest
extracted cost) and `novelty` (cost-function name -> whether the ILP
extract changed since the previous iteration).

Egg limits (`--max-iters`, `--max-nodes`, `--max-time`,
`--backoff-scheduler`) are read from the sibling `args.json` so that
the cost trajectory is measured under the same conditions used to
generate and validate the terms.

The result is written to `<input_dir>/cost_evolution.json` with shape
`{term: {"monotonic": [{costfn: value, ...}, ...],
         "novelty":   [{costfn: bool,  ...}, ...]}}`.
On any per-term failure (non-zero exit, timeout, unparseable output),
the term maps to an empty dict. Plotting lives in
`analysis/cost_evolution.ipynb`.

Example:
    cargo build --release --bin measure-cost
    uv run scripts/measure_cost.py --path data/seed_terms/foo-bar/terms.json
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Seed-terms JSON (e.g. data/seed_terms/foo-bar/terms.json). "
        "Its sibling `args.json` is read for the egg limits.",
    )
    parser.add_argument("--binary", type=Path, default=Path("target/release/measure-cost"))
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of concurrent measure-cost processes. Defaults to 1 "
        "because parallel runs competing for CPU can cause time-limited "
        "runs to stop earlier than they would sequentially.",
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(
            f"Binary not found: {args.binary}. "
            f"Build with `cargo build --release --bin measure-cost`.",
            file=sys.stderr,
        )
        return 2

    args_path = args.path.parent / "args.json"
    if not args_path.exists():
        print(f"args.json not found alongside {args.path}", file=sys.stderr)
        return 2

    with args_path.open("r") as f:
        run_args = json.load(f)

    max_iters = int(run_args["max_iters"])
    max_nodes = int(run_args["max_nodes"])
    max_time = float(run_args["max_time"])
    max_memory = run_args.get("max_memory")  # optional RSS ceiling in bytes
    backoff = bool(run_args.get("backoff_scheduler", False))
    language = str(run_args["language"])

    with args.path.open("r") as f:
        big_collector = json.load(f)

    # big_collector is [[size, {term: [attempts, validation, peak_memory_bytes]}], ...]
    all_terms = [term for _size, terms_map in big_collector for term in terms_map]

    timeout = max(1, int(max_time * 40) + 5)
    cmd_base = [
        str(args.binary),
        "--language",
        language,
        "--max-iters",
        str(max_iters),
        "--max-nodes",
        str(max_nodes),
        "--max-time",
        str(max_time),
    ]
    if max_memory is not None:
        cmd_base += ["--max-memory", str(max_memory)]
    if backoff:
        cmd_base.append("--backoff-scheduler")

    def measure_one(term: str) -> dict:
        proc = subprocess.run(
            [*cmd_base, "--term", term],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"measure-cost exited with {proc.returncode} for term {term!r}\n"
                f"--- stderr ---\n{proc.stderr}\n"
                f"--- stdout ---\n{proc.stdout}"
            )
        last_line = proc.stdout.strip().splitlines()[-1]
        try:
            return json.loads(last_line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Could not parse measure-cost output for term {term!r}: {e}\n"
                f"--- last line ---\n{last_line}\n"
                f"--- full stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            ) from e

    results: dict[str, dict] = {}
    workers = max(1, args.parallelism)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(measure_one, t): t for t in all_terms}
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="terms"):
                results[futures[fut]] = fut.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    out_path = args.path.parent / "cost_evolution.json"
    with out_path.open("w") as f:
        json.dump(results, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
