"""Generate validated start terms in isolated Rust processes.

TODO

Example:
    TODO
"""

import dataclasses
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from diceware.wordlist import WordList, get_wordlists_dir
from tqdm import tqdm

from common import exit_if_missing, parse_size, subprocess_timeout


def _load_wordlist(name: str) -> list[str]:
    path = os.path.join(get_wordlists_dir(), f"wordlist_{name}.txt")
    return list(WordList(path))


def generate_unique_dir(parent: Path, max_attempts: int = 100) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    adjectives = _load_wordlist("en_adjectives")
    nouns = _load_wordlist("en_nouns")
    for _ in range(max_attempts):
        candidate = parent / f"{secrets.choice(adjectives)}-{secrets.choice(nouns)}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError(f"Could not find an unused name under {parent}")


@dataclass
class Args:
    path: Path | None = None
    """Output directory. A fresh data/start_terms directory is used if omitted."""

    generate_binary: Path = Path("target/release/generate")
    total_samples: int = tyro.MISSING
    min_size: int = tyro.MISSING
    max_size: int = tyro.MISSING
    language: str = tyro.MISSING
    seed: int = tyro.MISSING
    retry_limit: int = 10000

    max_iters: int = 11
    max_nodes: int = 100_000
    max_time: float = 1.0
    max_memory: str | None = None
    """Absolute live-heap ceiling checked at egg's sampled rule boundaries."""
    predict_next_memory: Path | None = None

    workers: int = 4
    worker_timeout: float | None = None
    """External timeout per attempt; defaults to max_time*4+5 seconds."""
    max_job_attempts: int = 10000
    poll_interval: float = 0.1


@dataclass(frozen=True, order=True)
class Job:
    size: int
    slot: int
    attempt: int


@dataclass
class Running:
    job: Job
    seed: int
    process: subprocess.Popen[str]
    output_path: Path
    started: float
    peak_rss: int = 0


def derive_seed(global_seed: int, size: int, slot: int, attempt: int) -> int:
    """Stable 64-bit BLAKE2 seed; count of Python and scheduling."""
    h = hashlib.blake2b(digest_size=8, person=b"rise-seed-v1")
    for value in (global_seed, size, slot, attempt):
        h.update(int(value).to_bytes(16, "little", signed=True))
    return int.from_bytes(h.digest(), "little")


def semantic_config(args: Args) -> dict[str, Any]:
    """Arguments that affect the requested terms and their validation."""
    return {
        "generate_binary": str(args.generate_binary),
        "total_samples": args.total_samples,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "language": args.language,
        "seed": args.seed,
        "retry_limit": args.retry_limit,
        "max_iters": args.max_iters,
        "max_nodes": args.max_nodes,
        "max_time": args.max_time,
        "max_memory": args.max_memory,
        "predict_next_memory": (
            str(args.predict_next_memory) if args.predict_next_memory is not None else None
        ),
        "seed_derivation": "BLAKE2b-64(person=rise-seed-v1; signed-le128 fields)",
        "allocation_source": "generate plan",
    }


def _eqsat_flags(args: Args) -> list[str]:
    flags = [
        "--max-iters",
        str(args.max_iters),
        "--max-nodes",
        str(args.max_nodes),
        "--max-time",
        str(args.max_time),
    ]
    if args.max_memory is not None:
        flags += ["--max-memory", str(parse_size(args.max_memory))]
    if args.predict_next_memory is not None:
        flags += ["--predict-next-memory", str(args.predict_next_memory)]
    return flags


def base_generate_cmd(args: Args) -> list[str]:
    return [
        str(args.generate_binary),
        "one",
        "--language",
        args.language,
        "--retry-limit",
        str(args.retry_limit),
        *_eqsat_flags(args),
    ]


def request_plan(args: Args) -> list[tuple[int, int]]:
    cmd = [
        str(args.generate_binary),
        "plan",
        "--total-samples",
        str(args.total_samples),
        "--min-size",
        str(args.min_size),
        "--max-size",
        str(args.max_size),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"generate plan failed ({proc.returncode}):\n{proc.stderr}")
    plan = [(int(size), int(count)) for size, count in json.loads(proc.stdout)]
    if sum(count for _, count in plan) != args.total_samples:
        raise RuntimeError(f"Rust returned invalid allocation plan: {plan}")
    return plan


def print_summary(self, runtime: float) -> None:
    final = list(self.latest_attempt.values())
    peaks = sorted(int(row.get("peak_rss_bytes") or 0) for row in final)
    runtimes = sorted(float(row.get("runtime_seconds") or 0) for row in final)

    def percentile(values: list[float] | list[int], p: float):
        if not values:
            return 0.0
        index = round((len(values) - 1) * p)
        return float(values[index])

    retries = sum(1 for row in final if int(row["attempt"]) > 0)
    print(
        f"Completed {self.args.total_samples} terms in {runtime:.2f}s; "
        f"attempts={len(final)}, retries={retries}; "
        f"runtime p50/p95={percentile(runtimes, 0.5):.2f}/{percentile(runtimes, 0.95):.2f}s; "
        f"peak RSS p50/p95/max={percentile(peaks, 0.5) / 2**20:.1f}/"
        f"{percentile(peaks, 0.95) / 2**20:.1f}/{percentile(peaks, 1) / 2**20:.1f} MiB",
        file=sys.stderr,
    )


def validate_args(args: Args) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.total_samples < 0 or args.min_size > args.max_size:
        raise ValueError("invalid sample count or size range")


def main() -> int:
    args = tyro.cli(Args, description=__doc__)
    try:
        validate_args(args)
        if args.path is None:
            args.path = generate_unique_dir(Path("data/start_terms"))
            print(f"Auto-generated output dir: {args.path}", file=sys.stderr)
        else:
            args.path.mkdir(parents=True, exist_ok=True)
        exit_if_missing(args.generate_binary)
        plan = request_plan(args)
        prepare_generation_args(args)
        print(
            f"Generating {args.total_samples} terms with {args.workers} workers -> "
            f"{args.path / 'terms.json'}",
            file=sys.stderr,

        )
        return 0



if __name__ == "__main__":
    raise SystemExit(main())
