"""Generate start/goal problems that really need the memory budget.

Three stages, each fanned out over isolated Rust processes:
1. `start` samples one validated start term per size slot.
2. `candidates` draws goal candidates from each start term's novel frontier.
3. `verify` runs the unguided start->goal search; its peak RSS (`ru_maxrss`,
   the only memory number to trust here) decides whether the pair is kept.

Writes `problems.json` (accepted pairs) and `problem_args.json` (config).

Example:
    cargo build --release --bin start --bin candidates --bin verify
    uv run scripts/generate_problems.py --starts 10 --min-size 10 --max-size 12 \
      --language math --seed 42 --max-memory 4G --min-rss 3G
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from diceware.wordlist import WordList, get_wordlists_dir
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from common import (
    exit_if_missing,
    parse_size,
    run_json_subprocess,
    uniform_candidate_allocation,
    verify_summary,
)


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


class Args(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    # I/O
    path: Path | None = Field(
        default=None,
        description="Output directory. A fresh `data/problems` directory is used if omitted.",
    )

    start_bin: Path = Field(
        default=Path("target/release/start"), description="Start-term generation binary."
    )

    candidates_bin: Path = Field(
        default=Path("target/release/candidates"), description="Goal-candidate binary."
    )

    verify_bin: Path = Field(
        default=Path("target/release/verify"), description="Verification binary."
    )

    # Start terms
    starts: int = Field(gt=0, description="Number of start terms, spread uniformly over the sizes.")

    min_size: int = Field(gt=0, description="Smallest start-term size.")

    max_size: int = Field(gt=0, description="Largest start-term size (inclusive).")

    language: str = Field(min_length=1, description="Language.")

    seed: int = Field(description="Global seed; per-slot seeds are derived from it.")

    retry_limit: int = Field(
        default=10_000, gt=0, description="Candidate draws per `start` process before it gives up."
    )

    # Goal candidates
    goals: int = Field(default=10, gt=0, description="Goal candidates drawn per start term.")

    policy: Literal["count", "uniform"] = Field(
        default="count", description="Frontier draw policy used by `candidates`."
    )

    retry_step: int = Field(
        default=5, gt=0, description="How much to grow `max_size` on each exact-size-search retry."
    )

    max_retries: int = Field(
        default=20, ge=0, description="How many exact-size-search increments to allow."
    )

    size_goal: int = Field(
        default=5, gt=0, description="How many novel root sizes exact construction must find."
    )

    # Eqsat limits, shared by all three stages
    max_iters: int = Field(default=11, gt=0, description="Maximum eqsat iterations.")

    max_nodes: int = Field(default=100_000, gt=0, description="Maximum eqsat egraph nodes.")

    max_time: float = Field(default=1.0, gt=0, description="Maximum eqsat wall-clock seconds.")

    max_memory: str | None = Field(
        default=None,
        description="Absolute live-heap ceiling (e.g. `4G`), unlimited if omitted.",
    )

    # Acceptance
    min_rss: str = Field(
        default="0",
        description=(
            "Keep a (start, goal) pair only if the unguided `verify` run's peak "
            "RSS reaches this (e.g. `3G`), so cheap problems are dropped."
        ),
    )

    jobs: int | None = Field(
        default=None, gt=0, description="Concurrent subprocesses. Defaults to `os.cpu_count()`."
    )

    @model_validator(mode="after")
    def validate_sizes(self) -> Args:
        if self.min_size > self.max_size:
            raise ValueError(f"min_size ({self.min_size}) must be <= max_size ({self.max_size})")
        return self


def derive_seed(*fields: int) -> int:
    """Stable 64-bit BLAKE2 seed, independent of scheduling."""
    h = hashlib.blake2b(digest_size=8, person=b"rise-seed-v1")
    for value in fields:
        h.update(int(value).to_bytes(16, "little", signed=True))
    return int.from_bytes(h.digest(), "little")


def eqsat_flags(args: Args) -> list[str]:
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
    return flags


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def fan_out(jobs: int, fn, items: list, desc: str) -> list:
    """Run `fn` over `items` concurrently, dropping the ones that returned None."""
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(fn, item) for item in items]
        results = [
            fut.result()
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="job")
        ]
    return [r for r in results if r is not None]


def run_start(args: Args, flags: list[str], slot: tuple[int, int]) -> dict[str, Any] | None:
    """Sample one validated start term of the slot's size."""
    size, index = slot
    cmd = [
        str(args.start_bin),
        "--size",
        str(size),
        "--seed",
        str(derive_seed(args.seed, size, index)),
        "--language",
        args.language,
        "--retry-limit",
        str(args.retry_limit),
        *flags,
    ]
    try:
        payload = run_json_subprocess(cmd, what=f"start for size {size} slot {index}").payload
    except RuntimeError as e:
        warn(str(e))
        return None
    return {"start_term": payload["term"], "start_size": size, "attempts": payload["attempt"]}


def run_candidates(args: Args, flags: list[str], start: dict[str, Any]) -> dict[str, Any] | None:
    """Draw goal candidates from one start term's novel frontier."""
    cmd = [
        str(args.candidates_bin),
        "--language",
        args.language,
        "--start-term",
        start["start_term"],
        "--n-candidates",
        str(args.goals),
        "--seed",
        str(args.seed),
        "--policy",
        args.policy,
        "--retry-step",
        str(args.retry_step),
        "--max-retries",
        str(args.max_retries),
        "--size-goal",
        str(args.size_goal),
        *flags,
    ]
    try:
        records = run_json_subprocess(
            cmd, what=f"candidates for start term {start['start_term']!r}"
        ).payload
    except RuntimeError as e:
        warn(str(e))
        return None
    if not records:
        warn(f"candidates found nothing for start term {start['start_term']!r}")
        return None
    return {**start, "goal_terms": records[0]["candidate_s_expr"]}


def run_verify(args: Args, flags: list[str], pair: dict[str, Any]) -> dict[str, Any] | None:
    """Measure what the unguided start->goal search actually costs."""
    cmd = [
        str(args.verify_bin),
        "--language",
        args.language,
        "--start-term",
        pair["start_term"],
        "--goal-term",
        pair["goal_term"],
        *flags,
    ]
    try:
        measured = run_json_subprocess(cmd, what=f"verify for goal {pair['goal_term']!r}")
    except RuntimeError as e:
        warn(str(e))
        return None
    return {**pair, **verify_summary(measured.payload), "peak_rss_bytes": measured.peak_rss_bytes}


def main() -> int:
    args = Args()
    exit_if_missing(args.start_bin, args.candidates_bin, args.verify_bin)

    out = args.path or generate_unique_dir(Path("data/problems"))
    out.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or os.cpu_count() or 1
    flags = eqsat_flags(args)
    min_rss = parse_size(args.min_rss)

    slots = [
        (size, index)
        for size, count in uniform_candidate_allocation(
            list(range(args.min_size, args.max_size + 1)), args.starts
        )
        for index in range(count)
    ]
    print(f"Generating {len(slots)} start term(s) -> {out} ({jobs} workers)", file=sys.stderr)
    starts = fan_out(jobs, lambda slot: run_start(args, flags, slot), slots, "starts")

    # Distinct terms only; two slots of one size can sample the same term.
    seen: set[str] = set()
    starts = [s for s in starts if not (s["start_term"] in seen or seen.add(s["start_term"]))]

    enriched = fan_out(jobs, lambda s: run_candidates(args, flags, s), starts, "candidates")
    pairs = [
        {**{k: v for k, v in start.items() if k != "goal_terms"}, "goal_term": goal}
        for start in enriched
        for goal in start["goal_terms"]
    ]

    measured = fan_out(jobs, lambda p: run_verify(args, flags, p), pairs, "verify")
    problems = [row for row in measured if (row["peak_rss_bytes"] or 0) >= min_rss]

    (out / "problems.json").write_text(json.dumps(problems, indent=2))
    (out / "problem_args.json").write_text(
        json.dumps(
            {
                "language": args.language,
                "max_iters": args.max_iters,
                "max_nodes": args.max_nodes,
                "max_time": args.max_time,
                "max_memory": parse_size(args.max_memory) if args.max_memory else None,
                "driver_args": args.model_dump(),
            },
            indent=2,
            default=str,
        )
    )

    reached = sum(1 for row in problems if row["reached"])
    print(
        f"\nKept {len(problems)}/{len(measured)} pair(s) at peak RSS >= {min_rss} bytes "
        f"({reached} reached the goal) from {len(starts)} start term(s) "
        f"-> {out / 'problems.json'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
