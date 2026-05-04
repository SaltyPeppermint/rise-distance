# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "tqdm", "tyro", "diceware"]
# ///
"""Generate random math terms and measure peak RSS of eqsat on each.

Shells out to `target/release/generate` to produce a CSV of terms, then to
`target/release/measure-size` once per row to record peak resident set
size (VmHWM, matching htop) in a `peak_memory_bytes` column. Egg limits
(`--max-iters`, `--max-nodes`, `--max-time`, `--backoff-scheduler`) are
forwarded to both binaries.

On any per-term failure (non-zero exit, RLIMIT kill, timeout, unparseable
output), `peak_memory_bytes` is -1.

If `--path` is omitted, a fresh `data/seed_terms/<adjective>-<noun>/`
directory is created (collision-retry against existing siblings). The output
CSV `terms.csv` and a sidecar `args.json` recording all CLI args are written
into the directory.
"""

from __future__ import annotations

import dataclasses
import json
import os
import secrets
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from diceware.wordlist import WordList, get_wordlists_dir
from tqdm import tqdm


def parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = 1
    for suf, m in (("K", 1024), ("M", 1024**2), ("G", 1024**3), ("T", 1024**4)):
        if s.endswith(suf):
            mult = m
            s = s[:-1]
            break
    return int(float(s) * mult)


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
        f"Could not find an unused diceware name under {parent} "
        f"after {max_attempts} attempts."
    )


@dataclass
class Args:
    # I/O
    path: Path | None = None
    """Output directory. Will contain `terms.csv` and `args.json`. If
    omitted, a fresh `data/seed_terms/<adjective>-<noun>/` is created."""

    generate_binary: Path = Path("target/release/generate")
    measure_binary: Path = Path("target/release/measure-size")

    # generate-only
    total_samples: int = tyro.MISSING
    min_size: int = tyro.MISSING
    max_size: int = tyro.MISSING
    distribution: str = tyro.MISSING
    seed: int = tyro.MISSING
    tolerance: int = 1
    retry_limit: int = 10000
    parallelism: int | None = None

    # measure-only; per-term virtual-memory cap (e.g. "8G"). Backstop only.
    max_memory: str = tyro.MISSING

    # measure-only; number of concurrent measure-size processes. Defaults
    # to 1 because parallel runs compete for RAM and can perturb peak RSS.
    measure_parallelism: int = 1

    # shared egg args
    max_iters: int = 11
    max_nodes: int = 100_000
    max_time: float = 1.0
    backoff_scheduler: bool = False


def main() -> int:
    description = (__doc__ or "") + (
        "\n"
        "Example:\n"
        "    cargo build --release --bin generate --bin measure-size\n"
        "    uv run scripts/generate_and_measure.py \\\n"
        "        --total-samples 1000 --min-size 10 --max-size 50 \\\n"
        "        --distribution uniform --seed 42 --max-memory 8G \\\n"
        "        --max-iters 50 --max-nodes 100000 --max-time 10 \\\n"
        "        --backoff-scheduler --measure_parallelism 10"
    )
    args = tyro.cli(Args, description=description)

    if args.path is None:
        args.path = generate_unique_dir(Path("data/seed_terms"))
        print(f"Auto-generated output dir: {args.path}", file=sys.stderr)
    else:
        args.path.mkdir(parents=True, exist_ok=True)

    csv_path = args.path / "terms.csv"
    json_path = args.path / "args.json"

    max_memory_bytes = parse_size(args.max_memory)

    with json_path.open("w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2, default=str)

    for binary in (args.generate_binary, args.measure_binary):
        if not binary.exists():
            print(
                f"Binary not found: {binary}. "
                f"Build with `cargo build --release --bin generate --bin measure-size`.",
                file=sys.stderr,
            )
            return 2

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
        "--seed",
        str(args.seed),
        "--tolerance",
        str(args.tolerance),
        "--retry-limit",
        str(args.retry_limit),
        "--path",
        str(csv_path),
        "--max-iters",
        str(args.max_iters),
        "--max-nodes",
        str(args.max_nodes),
        "--max-time",
        str(args.max_time),
    ]
    if args.parallelism is not None:
        gen_cmd += ["--parallelism", str(args.parallelism)]
    if args.backoff_scheduler:
        gen_cmd.append("--backoff-scheduler")

    print(f"Generating terms -> {csv_path}", file=sys.stderr)
    gen = subprocess.run(gen_cmd)
    if gen.returncode != 0:
        print(f"generate failed (exit {gen.returncode})", file=sys.stderr)
        return gen.returncode

    df = pl.read_csv(csv_path)
    if "term" not in df.columns:
        print("Generated CSV is missing a `term` column", file=sys.stderr)
        return 2

    timeout = max(1, int(args.max_time * 4) + 5)
    measure_base = [
        str(args.measure_binary),
        "--max-iters",
        str(args.max_iters),
        "--max-nodes",
        str(args.max_nodes),
        "--max-time",
        str(args.max_time),
        "--max-memory",
        str(max_memory_bytes),
    ]
    if args.backoff_scheduler:
        measure_base.append("--backoff-scheduler")

    def measure_one(term: str) -> int:
        try:
            proc = subprocess.run(
                [*measure_base, "--term", term],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return (
                int(proc.stdout.strip().splitlines()[-1])
                if proc.returncode == 0
                else -1
            )
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return -1

    terms = df["term"].to_list()
    measurements: list[int] = [-1] * len(terms)
    workers = max(1, args.measure_parallelism)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(measure_one, t): i for i, t in enumerate(terms)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="terms"):
            measurements[futures[fut]] = fut.result()

    df.with_columns(pl.Series("peak_memory_bytes", measurements)).write_csv(csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
