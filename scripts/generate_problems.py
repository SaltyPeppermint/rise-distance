"""Generate start/goal problems that really need the memory budget.

Three stages, each fanned out over isolated Rust processes:
1. `start` samples one validated start term per size slot.
2. `sample` draws goal samples from each start term's novel frontier.
3. `attempt` runs the unguided start->goal search; its peak RSS (`VmHWM`, the
   only memory number to trust here) decides whether the pair is kept.

Writes `problems.json` (accepted pairs) and `problem_args.json` (config).
`experiment.py` drives it.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common import (
    MemoryKilled,
    PrioritySlot,
    SamplePolicy,
    attempt_summary,
    cli_flags,
    exit_if_missing,
    fan_out,
    log,
    parse_size,
    run_json_subprocess,
)


class GenerateArgs(BaseModel):
    # So a misspelled key in `experiment.py` fails instead of being dropped.
    model_config = ConfigDict(extra="forbid")

    # I/O
    output: Path = Field(description="Output directory.")

    start_bin: Path = Field(
        default=Path("target/release/start"), description="Start-term generation binary."
    )

    sample_bin: Path = Field(
        default=Path("target/release/sample"), description="Goal-sample binary."
    )

    attempt_bin: Path = Field(default=Path("target/release/attempt"), description="Attempt binary.")

    # Start terms
    starts: int = Field(gt=0, description="Number of start terms, spread uniformly over the sizes.")

    min_size: int = Field(gt=0, description="Smallest start-term size.")

    max_size: int = Field(gt=0, description="Largest start-term size (inclusive).")

    language: str = Field(min_length=1, description="Language.")

    seed: int = Field(description="Global seed; per-slot seeds are derived from it.")

    retry_limit: int = Field(
        default=10_000, gt=0, description="Samples draws per `start` process before it gives up."
    )

    # Goal Samples
    goals: int = Field(default=10, gt=0, description="Goal samples drawn per start term.")

    policy: SamplePolicy = Field(
        default=SamplePolicy.Count, description="Frontier draw policy used by `sample`."
    )

    size_search_steps: int = Field(
        default=100, ge=0, description="How many exact-size-search increments to allow."
    )

    # Eqsat limits, shared by all three stages
    max_iters: int = Field(default=11, gt=0, description="Maximum eqsat iterations.")

    max_nodes: int = Field(default=100_000, gt=0, description="Maximum eqsat egraph nodes.")

    max_time: float = Field(default=1.0, gt=0, description="Maximum eqsat wall-clock seconds.")

    max_memory: str | None = Field(
        default=None, description="Absolute live-heap ceiling (e.g. `4G`), unlimited if omitted."
    )

    # Acceptance
    min_rss: str = Field(
        default="0",
        description=(
            "Keep a (start, goal) pair only if the unguided `attempt` run's peak "
            "RSS reaches this (e.g. `3G`), so cheap problems are dropped."
        ),
    )

    max_rss: str | None = Field(
        default=None,
        description=("Cap each `start` process at this cgroup RSS limit, as a human size"),
    )

    start_retries: int = Field(
        default=10,
        ge=0,
        description=("How often a `start` slot killed at `max_rss` is redrawn under a bumped seed"),
    )

    @model_validator(mode="after")
    def validate_sizes(self) -> GenerateArgs:
        if self.min_size > self.max_size:
            raise ValueError(f"min_size ({self.min_size}) must be <= max_size ({self.max_size})")
        return self

    @model_validator(mode="after")
    def validate_memory(self) -> GenerateArgs:
        if (
            self.max_rss is not None
            and self.max_memory is not None
            and parse_size(self.max_rss) < parse_size(self.max_memory)
        ):
            raise ValueError(
                f"max_rss ({self.max_rss}) is below max_memory ({self.max_memory}); "
                "every run would be killed before its live-heap ceiling trips"
            )
        return self

    @property
    def limits(self) -> dict[str, int | float | None]:
        """The eqsat limits shared by all three stages."""
        return {
            "max_iters": self.max_iters,
            "max_nodes": self.max_nodes,
            "max_time": self.max_time,
            "max_memory": None if self.max_memory is None else parse_size(self.max_memory),
        }


def uniform_sample_allocation(sizes: list[int], total_samples: int) -> list[tuple[int, int]]:
    if not sizes:
        return []

    size_count = len(sizes)
    base = total_samples // size_count
    remainder = total_samples % size_count

    return [(size, base + int(i < remainder)) for i, size in enumerate(sizes)]


def derive_seed(*fields: int) -> int:
    """Stable 64-bit BLAKE2 seed, independent of scheduling."""
    h = hashlib.blake2b(digest_size=8, person=b"rise-seed-v1")
    for value in fields:
        h.update(int(value).to_bytes(16, "little", signed=True))
    return int.from_bytes(h.digest(), "little")


async def run_start(
    args: GenerateArgs, limit: PrioritySlot, flags: list[str], slot: tuple[int, int]
) -> dict[str, Any] | None:
    """Sample one validated start term of the slot's size, under the RSS cap.

    A slot killed at the cap is redrawn under a bumped seed. `start` seeds its
    RNG once per process and runs a validity eqsat per draw, so the same seed
    would replay the same sequence into the same fatal draw; only a different
    seed gives the slot a different term to spend its budget on.

    The seed and reseed count are recorded on the returned row, since a
    reseeded slot is no longer reproducible from `seed` and the slot alone.
    They are also what the caller counts the cap's cost from.
    """
    size, index = slot
    max_rss = parse_size(args.max_rss) if args.max_rss is not None else None
    for retry in range(args.start_retries + 1):
        fields = (args.seed, size, index) if retry == 0 else (args.seed, size, index, retry)
        seed = derive_seed(*fields)
        cmd = [
            str(args.start_bin),
            *cli_flags(size=size, seed=seed, language=args.language, retry_limit=args.retry_limit),
            *flags,
        ]
        what = f"start for size {size} slot {index}"
        if retry:
            what += f" (reseed {retry})"
        try:
            measured = await run_json_subprocess(cmd, what=what, limit=limit, max_rss_bytes=max_rss)
        except MemoryKilled as killed:
            log(f"WARNING: {killed!s}")
            continue
        except RuntimeError as e:
            log(f"WARNING: {e!s}")
            return None
        return {
            "start": measured.payload["term"],
            "start_size": size,
            "start_draws": measured.payload["draws"],
            "start_seed": seed,
            "start_retry": retry,
        }

    log(f"WARNING: start for size {size} slot {index}: killed at the RSS cap, no reseeds left")
    return None


async def run_samples(
    args: GenerateArgs, limit: PrioritySlot, flags: list[str], row: dict[str, Any]
) -> dict[str, Any] | None:
    """Draw goal samples from one start term's novel frontier."""
    cmd = [
        str(args.sample_bin),
        *cli_flags(
            language=args.language,
            start=row["start"],
            n_samples=args.goals,
            seed=args.seed,
            policy=args.policy,
            size_search_steps=args.size_search_steps,
            frontier=True,
        ),
        *flags,
    ]
    measured = await run_json_subprocess(
        cmd, what=f"samples for start term {row['start']!r}", limit=limit
    )
    records = measured.payload
    if not records:
        log(f"WARNING: samples found nothing for start term {row['start']!r}")
        return None
    return {**row, "goals": records[0]["samples_s_expr"]}


async def run_attempt(
    args: GenerateArgs, limit: PrioritySlot, flags: list[str], pair: dict[str, Any]
) -> dict[str, Any]:
    """Measure what the unguided start->goal search actually costs."""
    cmd = [
        str(args.attempt_bin),
        *cli_flags(language=args.language, start=pair["start"], goal=pair["goal"]),
        *flags,
    ]
    measured = await run_json_subprocess(
        cmd, what=f"attempt for goal {pair['goal']!r}", limit=limit
    )
    return {**pair, **attempt_summary(measured.payload), "peak_rss": measured.peak_rss}


async def generate(args: GenerateArgs, limit: PrioritySlot) -> str:
    """Generate the problems into `output` and return a one-line summary."""
    exit_if_missing(args.start_bin, args.sample_bin, args.attempt_bin)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    limits = args.limits
    flags = cli_flags(**limits)
    min_rss = parse_size(args.min_rss)

    slots = [
        (size, index)
        for size, count in uniform_sample_allocation(
            list(range(args.min_size, args.max_size + 1)), args.starts
        )
        for index in range(count)
    ]
    log(f"Generating {len(slots)} start term(s) -> {out}")
    starts = await fan_out(lambda slot: run_start(args, limit, flags, slot), slots, "starts")
    # A kept slot's `start_retry` is how many kills it was redrawn past; the
    # slots that came back with nothing were killed past their last reseed (or
    # failed outright, which warns for itself above). Silent when the cap cost
    # nothing.
    reseeded = sum(s["start_retry"] > 0 for s in starts)
    empty = len(slots) - len(starts)
    if args.max_rss is not None and (reseeded or empty):
        log(
            f"RSS cap {args.max_rss}: {reseeded}/{len(starts)} kept slot(s) needed a "
            f"reseed, {empty} slot(s) produced nothing"
        )

    # Distinct terms only; two slots of one size can sample the same term.
    unique: dict[str, dict] = {}
    for s in starts:
        unique.setdefault(s["start"], s)
    starts = list(unique.values())

    enriched = await fan_out(lambda s: run_samples(args, limit, flags, s), starts, "samples")
    pairs = [
        {**{k: v for k, v in row.items() if k != "goals"}, "goal": goal}
        for row in enriched
        for goal in row["goals"]
    ]

    measured = await fan_out(lambda p: run_attempt(args, limit, flags, p), pairs, "attempt")
    problems = [row for row in measured if (row["peak_rss"] or 0) >= min_rss]

    (out / "problems.json").write_text(json.dumps(problems, indent=2))
    (out / "problem_args.json").write_text(
        json.dumps(
            {
                "language": args.language,
                **limits,
                "driver_args": args.model_dump(),
            },
            indent=2,
            default=str,
        )
    )

    reached = sum(row["reached"] for row in problems)
    return (
        f"Kept {len(problems)}/{len(measured)} pair(s) at peak RSS >= {min_rss} bytes "
        f"({reached} reached the goal) from {len(starts)} start term(s) "
        f"-> {out / 'problems.json'}"
    )
