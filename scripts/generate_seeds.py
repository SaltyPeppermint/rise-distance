"""Generate random math terms, measuring eqsat live-heap use on each.

The Rust ``generate`` binary validates terms and records per-iteration egraph
statistics and jemalloc live-heap deltas in ``terms.json``. This driver forwards
the egg limits and records its arguments in ``generation_args.json``.

Without ``--path``, output goes to a new adjective-noun directory under
``data/seed_terms``.
"""

import dataclasses
import json
import os
import secrets
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from diceware.wordlist import WordList, get_wordlists_dir

from common import exit_if_missing, parse_size


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

    total_samples: int = tyro.MISSING
    min_size: int = tyro.MISSING
    max_size: int = tyro.MISSING
    distribution: str = tyro.MISSING
    language: str = tyro.MISSING
    seed: int = tyro.MISSING
    tolerance: int = 1
    retry_limit: int = 10000

    # egg args
    max_iters: int = 11
    max_nodes: int = 100_000
    max_time: float = 1.0
    max_memory: str | None = None
    """Graceful live-heap ceiling (jemalloc `stats.allocated`, e.g. "8G"),
    enforced by egg's per-iteration hook. Unset = unbounded."""
    backoff_scheduler: bool = True
    """Use egg's BackoffScheduler (pass --no-backoff-scheduler for the
    SimpleScheduler)."""


def main() -> int:
    description = (__doc__ or "") + (
        "\n"
        "Example:\n"
        "    cargo build --release --bin generate\n"
        "    uv run scripts/generate_seeds.py \\\n"
        "        --total-samples 1000 --min-size 10 --max-size 50 \\\n"
        "        --distribution uniform --language math --seed 42 \\\n"
        "        --max-iters 50 --max-nodes 100000 --max-time 10"
    )
    args = tyro.cli(Args, description=description)

    if args.path is None:
        args.path = generate_unique_dir(Path("data/seed_terms"))
        print(f"Auto-generated output dir: {args.path}", file=sys.stderr)
    else:
        args.path.mkdir(parents=True, exist_ok=True)

    terms_path = args.path / "terms.json"
    args_path = args.path / "generation_args.json"

    max_memory_bytes = parse_size(args.max_memory) if args.max_memory is not None else None

    exit_if_missing(args.generate_binary)

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
