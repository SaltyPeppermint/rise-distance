"""Drive the guide search from Python.

This driver reads the start/goal pairs in ``problems.json`` (written by
``generate_problems.py``) and runs one guide *search* per pair: a tree of guide
chains explored through a work queue.

Example:
    cargo build --release --bin samples --bin attempt
    uv run scripts/guided_search.py data/problems/dusky-cramp \\
        --stop-memory 4G --n-guides 5 --max-depth 3 --max-attempts 20 \\
        --max-total-time 300 --exploration-policy width \\
        --policy count --full-union

Pass ``--max-rss`` to hold each ``samples`` process to a cgroup RSS cap,
retrying a killed replay at the last iteration that completed and removing one
more iter off each further retry (``--sampling-backoff``).
"""

import asyncio
import itertools
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import Field
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from common import (
    MeasuredJson,
    MemoryKilled,
    attempt_summary,
    cli_flags,
    eqsat_finished,
    exit_if_missing,
    fan_out,
    parse_size,
    rss_killed_summary,
    run_json_subprocess,
)
from schemes import ATTEMPT_SCHEMA, EMPTY_GUIDE_META, EXPANSION_SCHEMA, PAIR_SCHEMA, UNGUIDED_SCHEMA

# TODO: the `smallest_novel`/`smallest_overall` policies are gone for now
type SamplePolicy = Literal["count", "uniform"]

type SearchPolicy = Literal["depth", "width"]

# How egg's `StopReason::Saturated` renders through `{:?}`
SATURATED = "Saturated"


class Args(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True, cli_kebab_case=True, cli_implicit_flags=True
    )

    # I/O
    path: CliPositionalArg[Path] = Field(
        description=(
            "Problem folder with `problems.json` and `problem_args.json` "
            "(both written by `generate_problems.py`)."
        )
    )

    output: Path | None = Field(
        default=None,
        description=(
            "Run folder for `results.parquet`/`results.json`. Auto-created under "
            "`data/guided_search/` if omitted."
        ),
    )

    sample_bin: Path = Field(
        default=Path("target/release/sample"), description="Path to the sample-construction binary."
    )

    attempt_bin: Path = Field(
        default=Path("target/release/attempt"), description="Path to the attempt binary."
    )

    # Guide-replay budget
    #
    # At least one must be given. Replay ends when the first configured budget
    # is exhausted; omitted budgets are effectively unlimited.
    stop_iters: int | None = Field(
        default=None, gt=0, description=("Guide-replay iteration budget.")
    )

    stop_nodes: int | None = Field(
        default=None, gt=0, description=("Guide-replay egraph-node budget.")
    )

    stop_time: float | None = Field(
        default=None, gt=0, description=("Guide-replay wall-clock budget in seconds.")
    )

    # Search budget
    max_total_time: float | None = Field(
        default=None, gt=0, description=("Wall-clock budget for one pair's whole search")
    )

    max_depth: int = Field(default=1, gt=0, description=("Longest guide chain to explore"))

    max_attempts: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Cap on `attempt` processes per pair. With `--n-guides g` and "
            "`--max-depth d` the tree grows exponential in size so cap it!"
        ),
    )

    search_policy: SearchPolicy = Field(
        default="depth", description=("breadth first vs depth search of the space")
    )

    # Search policy
    n_guides: int = Field(
        default=5,
        gt=0,
        description=("Guides drawn each time a node is expanded. -> Branching"),
    )

    max_rss: str = Field(
        default="4G",
        description=(
            "Cap each `samples` process at this cgroup RSS limit, as a human "
            "size such as `4G`. A process killed during guide replay is retried, "
            "still capped, with `--max-iters` cut to the iterations that "
            "completed. Uncapped if omitted."
        ),
    )

    sampling_backoff: int = Field(
        default=1,
        ge=0,
        description=(
            "How often a `samples` process killed at `--max-rss` is retried. `0` disables retrying."
        ),
    )

    size_search_steps: int = Field(
        default=200, ge=0, description="How many exact-size-search increments to allow."
    )

    sample_policy: SamplePolicy = Field(
        default="count", description="sample-pool sampling starteg."
    )

    frontier: bool = Field(
        default=False, description="Sample from the frontier of terms, not the whole egraph"
    )

    full_union: bool = Field(
        default=True, description="Use the full-union add for the attempt egraph."
    )

    start_terms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only process the first N start terms in sorted order, making the "
            "cutoff stable across runs. All start terms are processed if omitted."
        ),
    )

    goal_terms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only use the first N goals per start term in file order, making the "
            "cutoff stable across runs. All goals are used if omitted."
        ),
    )

    seed: int = Field(default=0, description="RNG seed used in Python and Rust.")

    jobs: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum number of concurrent pair searches. Each pair's own "
            "sampling/attempt loop stays sequential, so parallelism is "
            "across pairs. Defaults to `os.cpu_count()`. Lower this if large "
            "attempt egraphs exhaust available RAM."
        ),
    )

    @property
    def limits(self) -> dict[str, int | float]:
        """The configured guide-replay budgets, without the omitted ones."""
        budgets = {
            "max_iters": self.stop_iters,
            "max_nodes": self.stop_nodes,
            "max_time": self.stop_time,
        }
        return {key: value for key, value in budgets.items() if value is not None}

    def base_flags(self, language: str) -> list[str]:
        """Flags shared by every `attempt` process."""
        return cli_flags(language=language, **self.limits)

    def sample_flags(self, language: str) -> list[str]:
        """Flags shared by every `sample` process."""
        return cli_flags(
            language=language,
            seed=self.seed,
            policy=self.sample_policy,
            size_search_steps=self.size_search_steps,
            frontier=self.frontier,
        )


@dataclass(frozen=True)
class Problem:
    """One start/goal problem."""

    start: str
    goal: str


@dataclass(frozen=True)
class SearchNode:
    """Node in the search tree

    `guide` is the node array of the guide as origin_lang; `s_expr` is the same
    term lowered, which is what `samples --start-term` samples from next.
    The root carries the start term and no guide, since it is the unguided
    baseline rather than an attempt.

    `terminal` marks a node drawn from a saturated egraph
    """

    node_id: int
    parent_id: int | None
    depth: int
    guide: list | None
    s_expr: str
    terminal: bool = False


@dataclass(frozen=True)
class Expansion:
    """The outcome of one `samples` process: the pool drawn and its cost."""

    children: list[tuple[list, str]]
    status: Literal["ok", "empty_pool", "no_novel_terms", "rss_killed"]
    meta: dict
    wall_time: float

    @property
    def saturated(self) -> bool:
        return self.meta.get("guide_stop_reason") == SATURATED


@dataclass
class SearchFrontier:
    """The work queue, ordered by the exploration policy.

    `depth` pops the most recently pushed node, so the search follows one chain
    down before trying its siblings; `width` pops the oldest, exhausting a depth
    before descending.
    """

    policy: SearchPolicy
    _items: deque[SearchNode] = field(default_factory=deque)

    def push(self, nodes: list[SearchNode]) -> None:
        self._items.extend(nodes)

    def pop(self) -> SearchNode | None:
        if not self._items:
            return None
        return self._items.pop() if self.policy == "depth" else self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class Budget:
    """A pair's stop conditions, all charged lazily as the search runs."""

    started: float
    deadline: float | None
    max_depth: int
    max_attempts: int | None

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def attempts_left(self, attempts_run: int) -> bool:
        return self.max_attempts is None or attempts_run < self.max_attempts


@dataclass
class PairTrace:
    """Everything one pair's search did, flattened into rows at report time."""

    pair: Problem
    attempts: list[dict] = field(default_factory=list)
    expansions: list[dict] = field(default_factory=list)
    stop_reason: str = "unstarted"
    wall_time: float = 0.0

    @property
    def setup_status(self) -> str:
        """The root expansion's status: whether the pair got a pool at all."""
        return self.expansions[0]["status"] if self.expansions else "unstarted"


@dataclass(frozen=True)
class AttemptResult:
    summary: dict
    peak_rss_bytes: int | None
    wall_time: float


def flatten_problems(args: Args) -> list[Problem]:
    """Group `problems.json`'s pair rows into Problem Pairs."""
    rows = json.loads((args.path / "problems.json").read_text())
    goals: dict[str, list[str]] = {}
    for row in rows:
        goals.setdefault(row["start_term"], []).append(row["goal_term"])
    specs = [(start, goals[start][: args.goal_terms]) for start in sorted(goals)]
    return [Problem(start, goal) for (start, goals) in specs[: args.start_terms] for goal in goals]


async def run_capped(args: Args, cmd: list[str], what: str) -> MeasuredJson | None:
    """Run under the RSS cap, retrying a replay-phase kill up to
    `--sampling-backoff` times: the first retry replays the iterations that
    survived, each further one gives up another iteration."""
    cmd = [*cmd, "--print-success-iters"]
    iters: int | None = None
    attempts = 1
    post_eqsat_kill = 0
    cap = parse_size(args.max_rss)

    for retries_left in range(args.sampling_backoff, -1, -1):
        try:
            # print(f"CMD: {' '.join(cmd)}")
            measured = await run_json_subprocess(cmd, what=what, rss_max_bytes=cap)
            return measured
        except MemoryKilled as killed:
            if eqsat_finished(killed.stderr):
                post_eqsat_kill += 1

            if not retries_left:
                return None

            if iters is None:
                iters = killed.last_iter
                assert iters is not None, "How can it be killed with 0 iters"

            iters -= 1
            attempts += 1

            # Copy `cmd` with its `--max-iters` value replaced.
            try:
                index = cmd.index("--max-iters")
                cmd[index + 1] = str(iters)
            except ValueError:
                cmd.extend(["--max-iters", str(iters)])

    return None


async def draw_expansion(args: Args, sample_flags: list[str], s_expr: str) -> Expansion:
    """Run one `sample` process from `s_expr`. This is called recursively at every depth"""
    cmd = [
        str(args.sample_bin),
        *sample_flags,
        *cli_flags(**args.limits, start_term=s_expr, n_samples=args.n_guides),
    ]

    started = time.monotonic()
    measured = await run_capped(args, cmd, f"sample for term {s_expr!r}")
    wall_time = time.monotonic() - started

    # A capped-out child never printed its `Measured` envelope.
    if measured is None:
        return Expansion([], "rss_killed", dict(EMPTY_GUIDE_META), wall_time)

    # An empty payload is `samples` reporting that construction failed.
    if not measured.payload:
        meta = {**EMPTY_GUIDE_META, "sample_peak_rss_bytes": measured.peak_rss_bytes}
        return Expansion([], "no_novel_terms", meta, wall_time)

    record = measured.payload[0]
    children = list(zip(record["samples"], record["samples_s_expr"], strict=True))
    meta = {
        "guide_nodes": record["guide_nodes"],
        "guide_classes": record["guide_classes"],
        "guide_time": record["guide_time"],
        "guide_memory": record["guide_memory"],
        "guide_peak_live_heap": record["guide_peak_live_heap"],
        "guide_stop_reason": record["stop_reason"],
        "sample_peak_rss_bytes": measured.peak_rss_bytes,
    }
    return Expansion(children, "ok" if children else "empty_pool", meta, wall_time)


class SamplePools:
    """Every `sample` task the run started, keyed by the start term it sampled from.

    This allows lazy sampling! Later callers simply get the cached results

    The sample draws are all in the same `group` so a failed pair cancels
    sample tasks still running instead of leaving them un-awaited
    """

    def __init__(self, args: Args, sample_flags: list[str], group: asyncio.TaskGroup) -> None:
        self.args = args
        self.sample_flags = sample_flags
        self.group = group
        self.tasks: dict[str, asyncio.Task[Expansion]] = {}

    def __len__(self) -> int:
        return len(self.tasks)

    async def draw(self, s_expr: str) -> tuple[Expansion, bool]:
        """That term's pool plus if someone else is the one paying for it."""
        # No `await` before the task finishes, so no race
        cached = s_expr in self.tasks
        if not cached:
            self.tasks[s_expr] = self.group.create_task(
                draw_expansion(self.args, self.sample_flags, s_expr)
            )
        return await self.tasks[s_expr], cached

    def drawn(self) -> dict[str, Expansion]:
        """Serialize finished draws for reporting."""
        return {
            s_expr: task.result()
            for s_expr, task in self.tasks.items()
            if task.done() and not task.cancelled() and task.exception() is None
        }


async def run_attempt(args: Args, base_flags: list[str], goal: str, guide: list) -> AttemptResult:
    """Run one attempt in its own process.

    An attempt killed at the RSS cap comes back as a failed attempt with
    ``stop_reason="rss_killed"`` rather than an exception, since the search
    simply moves on to the next node.
    """
    cmd = [
        str(args.attempt_bin),
        *base_flags,
        *cli_flags(
            goal_term=goal,
            is_guide=True,
            start_term=json.dumps(guide),
            full_union=args.full_union,
        ),
    ]

    started = time.monotonic()
    try:
        measured = await run_json_subprocess(
            cmd, what=f"attempt for goal {goal!r}", rss_max_bytes=parse_size(args.max_rss)
        )
        summary, peak_rss_bytes = attempt_summary(measured.payload), measured.peak_rss_bytes
    except MemoryKilled:
        summary, peak_rss_bytes = rss_killed_summary(), None
    return AttemptResult(summary, peak_rss_bytes, time.monotonic() - started)


async def expand_node(
    pools: SamplePools,
    trace: PairTrace,
    budget: Budget,
    front: SearchFrontier,
    node: SearchNode,
    ids: itertools.count,
    seen: set[str],
) -> None:
    """Draw `node`'s pool, record what it cost, and queue the unseen children.

    Guides already tried on this pair are dropped: sharing pools across nodes
    makes the search tree a DAG, and skips guides already attempted on this
    pair.

    A saturated replay marks its children terminal, so the search attempts them
    but never samples past them.
    """
    started_at = budget.elapsed()
    expansion, cached = await pools.draw(node.s_expr)

    children = []
    for guide, s_expr in expansion.children:
        key = json.dumps(guide)
        if key in seen:
            continue
        seen.add(key)
        children.append(
            SearchNode(
                next(ids), node.node_id, node.depth + 1, guide, s_expr, terminal=expansion.saturated
            )
        )
    front.push(children)

    trace.expansions.append(
        {
            "start_term": trace.pair.start,
            "goal_term": trace.pair.goal,
            "node_id": node.node_id,
            "depth": node.depth,
            "status": expansion.status,
            "cached": cached,
            "saturated": expansion.saturated,
            "drawn": len(expansion.children),
            "pushed": len(children),
            "started_at": started_at,
            # A cached pool cost this pair nothing but the lookup.
            "wall_time": 0.0 if cached else expansion.wall_time,
            **expansion.meta,
        }
    )


async def search_pair(
    args: Args, base_flags: list[str], pools: SamplePools, pair: Problem
) -> PairTrace:
    """Run one pair's sampling/attempt search and return its trace.

    The loop is: pop a node, attempt it, and on failure expand it back onto the
    frontier. Each attempt is a separate `attempt` process, so its
    `attempt_peak_rss_bytes` is that attempt's own peak rather than a high-water mark
    shared across the pair.
    """
    started = time.monotonic()
    budget = Budget(
        started=started,
        deadline=None if args.max_total_time is None else started + args.max_total_time,
        max_depth=args.max_depth,
        max_attempts=args.max_attempts,
    )
    trace = PairTrace(pair)
    frontier = SearchFrontier(args.search_policy)
    ids = itertools.count(1)
    seen: set[str] = set()

    root = SearchNode(node_id=0, parent_id=None, depth=0, guide=None, s_expr=pair.start)
    await expand_node(pools, trace, budget, frontier, root, ids, seen)

    while True:
        if budget.expired():
            trace.stop_reason = "time_exhausted"
            break
        if not budget.attempts_left(len(trace.attempts)):
            trace.stop_reason = "attempt_budget_exhausted"
            break
        node = frontier.pop()
        if node is None:
            # Every node bottomed out at `--max-depth`, hit a saturated egraph,
            # or the pools ran dry; `setup_status` and the expansion rows'
            # `saturated` flag tell those apart.
            trace.stop_reason = "frontier_exhausted"
            break

        assert node.guide is not None, "the root is a baseline, not an attempt"
        started_at = budget.elapsed()
        attempt = await run_attempt(args, base_flags, pair.goal, node.guide)
        trace.attempts.append(
            {
                "start_term": pair.start,
                "goal_term": pair.goal,
                "policy": args.sample_policy,
                "attempt": len(trace.attempts),
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "guide_s_expr": node.s_expr,
                "terminal": node.terminal,
                "started_at": started_at,
                "wall_time": attempt.wall_time,
                "attempt_peak_rss_bytes": attempt.peak_rss_bytes,
                **attempt.summary,
            }
        )

        if attempt.summary["reached"]:
            trace.stop_reason = "reached"
            break

        # A terminal node came out of a saturated egraph, so sampling from it
        # would rebuild that same egraph and redraw that same pool. Leaving it
        # unexpanded sends the search back up to whatever the frontier holds.
        if not node.terminal and node.depth < budget.max_depth and not budget.expired():
            await expand_node(pools, trace, budget, frontier, node, ids, seen)

    trace.wall_time = budget.elapsed()
    return trace


async def run_unguided_pair(args: Args, base_flags: list[str], pair: Problem) -> dict:
    """Run the pair-matched single-start baseline.

    A baseline killed at the RSS cap becomes an ``rss_killed`` failure row.
    """
    cmd = [
        str(args.attempt_bin),
        *base_flags,
        *cli_flags(start_term=pair.start, goal_term=pair.goal),
    ]
    what = f"unguided attempt for goal term {pair.goal!r}"
    try:
        measured = await run_json_subprocess(cmd, what=what, rss_max_bytes=parse_size(args.max_rss))
        summary = attempt_summary(measured.payload)
        peak_rss_bytes = measured.peak_rss_bytes
    except MemoryKilled:
        # print(f"{what}: killed at RSS cap", file=sys.stderr)
        summary, peak_rss_bytes = rss_killed_summary(), None
    return {
        "start_term": pair.start,
        "goal_term": pair.goal,
        "unguided_success": summary["reached"],
        "unguided_stop_reason": summary["stop_reason"],
        "unguided_panic": summary["panic"],
        "unguided_final_live_heap_bytes": summary["memory"],
        "unguided_peak_live_heap_bytes": summary["peak_live_heap"],
        "unguided_peak_rss_bytes": peak_rss_bytes,
    }


# -----------
# Reporting
# -----------


def resolve_output_dir(args: Args) -> Path:
    """Resolve (+ create) folder, auto-numbering `run.N` if unset."""
    out = args.output
    if out is None:
        base = Path("data/guided_search")
        base.mkdir(parents=True, exist_ok=True)
        existing = [int(p.suffix[1:]) for p in base.glob("run.*") if p.suffix[1:].isdigit()]
        out = base / f"run.{max(existing, default=0) + 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def summarize_pair(args: Args, trace: PairTrace) -> dict:
    """Collapse one search into a single guided-workflow row."""
    attempts = trace.attempts
    successes = [attempt for attempt in attempts if attempt["reached"]]

    # Each attempt is its own process, so pick which one's peak to report rather
    # than inheriting a shared high-water mark.
    #
    # `attempt_peak_rss_bytes` is the *decisive* attempt: the one that reached, or
    # the last one tried if none did.
    # `attempt_peak_rss_bytes_max` is the max across every attempt run, which is what
    # the pair cost end to end.
    decisive = successes[0] if successes else (attempts[-1] if attempts else None)
    attempt_peak = decisive["attempt_peak_rss_bytes"] if decisive else None

    attempt_peaks = [
        attempt["attempt_peak_rss_bytes"]
        for attempt in attempts
        if attempt.get("attempt_peak_rss_bytes") is not None
    ]
    attempt_peak_max = max(attempt_peaks) if attempt_peaks else None

    expansion_peaks = [
        exp["sample_peak_rss_bytes"]
        for exp in trace.expansions
        if exp.get("sample_peak_rss_bytes") is not None
    ]
    sample_peak = max(expansion_peaks) if expansion_peaks else None

    rss_peaks = [peak for peak in (sample_peak, attempt_peak_max) if peak is not None]
    live_peaks = [
        peak
        for peak in (
            *(exp.get("guide_peak_live_heap") for exp in trace.expansions),
            *(attempt.get("peak_live_heap") for attempt in attempts),
        )
        if peak is not None
    ]

    # A pool that drew nothing usable is a setup failure; a pool that queued
    # nodes the search never got to is not — that is `search_stop_reason`'s job.
    setup_status = trace.setup_status
    if setup_status == "ok" and not trace.expansions[0]["pushed"]:
        setup_status = "empty_pool"

    return {
        "start_term": trace.pair.start,
        "goal_term": trace.pair.goal,
        "policy": args.sample_policy,
        "exploration_policy": args.search_policy,
        "max_depth": args.max_depth,
        "branching": args.n_guides,
        "attempt_budget": args.max_attempts,
        "time_budget": args.max_total_time,
        "guided_success": bool(successes),
        "search_stop_reason": trace.stop_reason,
        "success_attempt": successes[0]["attempt"] + 1 if successes else None,
        "success_depth": successes[0]["depth"] if successes else None,
        "attempts_run": len(attempts),
        "expansions_run": len(trace.expansions),
        "expansions_paid": sum(1 for exp in trace.expansions if not exp["cached"]),
        "saturated_expansions": sum(1 for exp in trace.expansions if exp["saturated"]),
        # It does not make sense to draw guides from a saturated egraph.
        "root_saturated": bool(trace.expansions and trace.expansions[0]["saturated"]),
        "deepest_attempt": max((attempt["depth"] for attempt in attempts), default=None),
        "pair_wall_time": trace.wall_time,
        # The last attempt's reason when one ran, otherwise why none did.
        "guided_stop_reason": None
        if successes
        else (
            attempts[-1].get("stop_reason")
            if attempts
            else (setup_status if setup_status != "ok" else trace.stop_reason)
        ),
        "guided_panic": any(attempt["panic"] for attempt in attempts),
        "setup_status": setup_status,
        "sample_status": trace.setup_status,
        "attempt_peak_rss_bytes": attempt_peak,
        "attempt_peak_rss_bytes_max": attempt_peak_max,
        "sample_peak_rss_bytes": sample_peak,
        "guided_peak_rss_bytes": max(rss_peaks) if rss_peaks else None,
        "guided_peak_live_heap_bytes": max(live_peaks) if live_peaks else None,
    }


def write_sample_pools(pools: SamplePools, out: Path) -> None:
    """Dump every pool the run drew, keyed by the term it was sampled from."""
    out.mkdir(parents=True, exist_ok=True)
    dumped = {
        s_expr: {
            "status": expansion.status,
            "samples": [guide for guide, _ in expansion.children],
            "sample_s_expr": [child for _, child in expansion.children],
            **expansion.meta,
        }
        for s_expr, expansion in pools.drawn().items()
    }
    (out / "samples.json").write_text(json.dumps(dumped))


def report_results(
    args: Args,
    traces: list[PairTrace],
    unguided_rows: list[dict],
    pools: SamplePools,
) -> None:
    """Write attempt, expansion, pair, baseline, and joined comparison results."""
    out = resolve_output_dir(args)

    attempt_rows = [row for trace in traces for row in trace.attempts]
    expansion_rows = [row for trace in traces for row in trace.expansions]

    attempts = pl.DataFrame(attempt_rows, schema=ATTEMPT_SCHEMA)
    attempts.write_parquet(out / "results.parquet")
    (out / "results.json").write_text(json.dumps(attempt_rows, indent=2))

    expansions = pl.DataFrame(expansion_rows, schema=EXPANSION_SCHEMA)
    expansions.write_parquet(out / "expansions.parquet")

    pairs = pl.DataFrame([summarize_pair(args, trace) for trace in traces], schema=PAIR_SCHEMA)
    unguided = pl.DataFrame(unguided_rows, schema=UNGUIDED_SCHEMA)
    comparison = pairs.join(unguided, on=["start_term", "goal_term"], how="left", validate="1:1")
    pairs.write_parquet(out / "pair_results.parquet")
    unguided.write_parquet(out / "unguided_results.parquet")
    comparison.write_parquet(out / "comparison.parquet")

    write_sample_pools(pools, out / "sample_run")

    config = {
        **args.model_dump(),
        "effective_limits": args.limits,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    reached_pairs = int(pairs["guided_success"].sum())
    total_pairs = len(pairs)
    reach_rate = reached_pairs / total_pairs if total_pairs else 0.0
    attempts_run = int(pairs["attempts_run"].sum())
    print(
        f"\nReached {reached_pairs}/{total_pairs} start/goal pairs "
        f"(reach rate {reach_rate:.2f}) in {attempts_run} attempt(s) and "
        f"{len(pools)} distinct sample pool(s). "
        f"Wrote {out / 'comparison.parquet'}",
        file=sys.stderr,
    )


async def main() -> int:
    args = Args()

    exit_if_missing(args.sample_bin, args.attempt_bin)
    jobs = args.jobs or os.cpu_count() or 1

    cfg = json.loads((args.path / "problem_args.json").read_text())
    base_flags = args.base_flags(str(cfg["language"]))
    sample_flags = args.sample_flags(str(cfg["language"]))

    pairs = flatten_problems(args)
    print(
        f"Searching {len(pairs)} (start, goal) pair(s) "
        f"(policy={args.search_policy}, max_depth={args.max_depth}, "
        f"branching={args.n_guides}, max_attempts={args.max_attempts}, "
        f"max_total_time={args.max_total_time})",
        file=sys.stderr,
    )
    # The pools draw in this scope rather than in detached tasks, so the first
    # failed pair cancels the draws still running instead of leaving orphaned
    # `sample` children behind.
    async with asyncio.TaskGroup() as draws:
        pools = SamplePools(args, sample_flags, draws)
        traces = await fan_out(
            jobs,
            lambda pair: search_pair(args, base_flags, pools, pair),
            pairs,
            "search",
            unit="pair",
        )

    # The baseline is re-run here, not reused from `problems.json`: the search
    # budget is the `--stop-*` one, not the budget generation measured under.
    print(f"Running {len(pairs)} pair-matched unguided baseline(s)", file=sys.stderr)
    unguided_rows = await fan_out(
        jobs, lambda pair: run_unguided_pair(args, base_flags, pair), pairs, "unguided", unit="pair"
    )

    report_results(args, traces, unguided_rows, pools)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
