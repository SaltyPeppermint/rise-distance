"""Generate random math terms and measure peak RSS of eqsat on each.

Shells out to `target/release/generate` to produce a nested JSON of terms
grouped by size, then to `target/release/measure-size` once per term to
record peak resident set size (VmHWM, matching htop). The peak is appended
to each inner term record. Egg limits (`--max-iters`, `--max-nodes`,
`--max-time`, `--max-memory`, `--backoff-scheduler`) are forwarded to both
binaries; `--rlimit-as` (virtual-memory backstop) only to `measure-size`.

`measure-size` emits a JSON object `{"peak": <VmHWM bytes>, "iterations":
[<egg per-iteration stats, each with an `rss` field>, ...]}`. The whole
object is stored as the 3rd entry of each inner term record, so the
`terms.json` payload has shape
`[[size, {term: [attempts, validation_result, {peak, iterations}]}], ...]`.

On any per-term failure (non-zero exit, RLIMIT kill, timeout, unparseable
output), the 3rd entry is `{"peak": -1, "iterations": []}`.

If `--path` is omitted, a fresh `data/seed_terms/<adjective>-<noun>/`
directory is created (collision-retry against existing siblings). The output
`terms.json` and a sidecar `generation_args.json` recording all CLI args are
written into the directory.
"""

import dataclasses
import json
import os
import secrets
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import tyro
from diceware.wordlist import WordList, get_wordlists_dir
from tqdm import tqdm

from common import MEASURE_FAILURE, exit_if_missing, measure_term, parse_size, subprocess_timeout


def _load_wordlist(name: str) -> list[str]:
    path = os.path.join(get_wordlists_dir(), f"wordlist_{name}.txt")
    return list(WordList(path))


def generate_unique_dir(parent: Path, max_attempts: int = 100) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    adjectives = _load_wordlist("en_adjectives")
    nouns = _load_wordlist("en_nouns")
    for _ in range(max_attempts):
        name = f"{secrets.choice(adjectives)}-{secrets.choice(nouns)}"
        candidate = parent / name
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError(
        f"Could not find an unused diceware name under {parent} after {max_attempts} attempts."
    )


@dataclass
class Args:
    # I/O
    path: Path | None = None
    """Output directory. Will contain `terms.json` and `generation_args.json`.
    If omitted, a fresh `data/seed_terms/<adjective>-<noun>/` is created."""

    generate_binary: Path = Path("target/release/generate")
    measure_binary: Path = Path("target/release/measure-size")

    # generate-only
    total_samples: int = tyro.MISSING
    min_size: int = tyro.MISSING
    max_size: int = tyro.MISSING
    distribution: str = tyro.MISSING
    language: str = tyro.MISSING
    seed: int = tyro.MISSING
    tolerance: int = 1
    retry_limit: int = 10000
    # measure-only; number of concurrent measure-size processes. Defaults
    # to 1 because parallel runs compete for RAM and can perturb peak RSS.
    parallelism: int = 1

    # measure-only; per-term virtual-memory cap (e.g. "8G"), enforced via
    # RLIMIT_AS in measure-size. Backstop only.
    rlimit_as: str = tyro.MISSING

    # shared egg args
    max_iters: int = 11
    max_nodes: int = 100_000
    max_time: float = 1.0
    max_memory: str | None = None
    """Graceful process RSS ceiling (e.g. "8G"), enforced by egg's per-iteration
    hook in both binaries. Unset = unbounded. Must be below `--rlimit-as`."""
    backoff_scheduler: bool = True
    """Use egg's BackoffScheduler (pass --no-backoff-scheduler for the
    SimpleScheduler)."""


def main() -> int:
    description = (__doc__ or "") + (
        "\n"
        "Example:\n"
        "    cargo build --release --bin generate --bin measure-size\n"
        "    uv run scripts/generate_seeds.py \\\n"
        "        --total-samples 1000 --min-size 10 --max-size 50 \\\n"
        "        --distribution uniform --language math --seed 42 --rlimit-as 8G \\\n"
        "        --max-iters 50 --max-nodes 100000 --max-time 10 \\\n"
        "        --measure_parallelism 10"
    )
    args = tyro.cli(Args, description=description)

    if args.path is None:
        args.path = generate_unique_dir(Path("data/seed_terms"))
        print(f"Auto-generated output dir: {args.path}", file=sys.stderr)
    else:
        args.path.mkdir(parents=True, exist_ok=True)

    terms_path = args.path / "terms.json"
    args_path = args.path / "generation_args.json"

    rlimit_as_bytes = parse_size(args.rlimit_as)
    max_memory_bytes = parse_size(args.max_memory) if args.max_memory is not None else None
    if max_memory_bytes is not None and max_memory_bytes >= rlimit_as_bytes:
        print(
            f"--max-memory ({max_memory_bytes}) must be below --rlimit-as "
            f"({rlimit_as_bytes}), or the hard RLIMIT_AS kill preempts the "
            "graceful RSS stop",
            file=sys.stderr,
        )
        return 1

    exit_if_missing(args.generate_binary, args.measure_binary)

    with args_path.open("w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2, default=str)

    gen_cmd = [
        str(args.generate_binary),
        "--total-samples",
        str(args.total_samples),
        "--min-size",
        str(args.min_size),
        "--max-size",
        str(args.max_size),
        "--distribution",
        args.distribution,
        "--language",
        args.language,
        "--seed",
        str(args.seed),
        "--tolerance",
        str(args.tolerance),
        "--retry-limit",
        str(args.retry_limit),
        "--path",
        str(terms_path),
        "--max-iters",
        str(args.max_iters),
        "--max-nodes",
        str(args.max_nodes),
        "--max-time",
        str(args.max_time),
    ]

    if max_memory_bytes is not None:
        gen_cmd += ["--max-memory", str(max_memory_bytes)]
    if args.backoff_scheduler:
        gen_cmd.append("--backoff-scheduler")

    print(f"Generating terms -> {terms_path}", file=sys.stderr)
    gen = subprocess.run(gen_cmd)
    if gen.returncode != 0:
        print(f"generate failed (exit {gen.returncode})", file=sys.stderr)
        return gen.returncode

    with terms_path.open("r") as f:
        big_collector = json.load(f)

    timeout = subprocess_timeout(args.max_time)
    measure_base = [
        str(args.measure_binary),
        "--language",
        args.language,
        "--max-iters",
        str(args.max_iters),
        "--max-nodes",
        str(args.max_nodes),
        "--max-time",
        str(args.max_time),
        "--rlimit-as",
        str(rlimit_as_bytes),
    ]
    if max_memory_bytes is not None:
        measure_base += ["--max-memory", str(max_memory_bytes)]
    if args.backoff_scheduler:
        measure_base.append("--backoff-scheduler")

    def measure_one(term: str) -> dict:
        return measure_term([*measure_base, "--term", term], timeout=timeout)

    # big_collector is [[size, {term_str: [attempts, validation_result]}], ...].
    # Collect (size_idx, term_str) refs in a flat list, measure in parallel,
    # then append the `{peak, iterations}` object to each inner record in place.
    flat: list[tuple[int, str]] = [
        (size_idx, term)
        for size_idx, (_size, terms_map) in enumerate(big_collector)
        for term in terms_map
    ]
    measurements: list[dict] = [dict(MEASURE_FAILURE) for _ in flat]
    workers = max(1, args.parallelism)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(measure_one, t): i for i, (_, t) in enumerate(flat)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="terms"):
            measurements[futures[fut]] = fut.result()

    for (size_idx, term), stats in zip(flat, measurements):
        big_collector[size_idx][1][term].append(stats)

    with terms_path.open("w") as f:
        json.dump(big_collector, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
