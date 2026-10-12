"""Drive the guide search from Python.

This driver reads the start/goal pairs in ``problems.json`` (written by
``generate_problems.py``) and runs one guide *search* per pair: a tree of guide
chains explored through a work queue. `experiment.py` drives it.

Every ``sample`` and ``attempt`` process is held to the ``max_rss`` cgroup
RSS cap. A killed ``sample`` is retried up to ``sampling_backoff`` times,
first at the number of iterations that completed, then giving up one more
rewrite-applying iteration per retry. A process that dies of an uncaught panic
is recorded as ``binary_panic``, one that outruns its hard timeout (``max_pair_time``
plus a fixed grace) as ``timeout``; neither is retried. A process whose arguments
exceed the kernel's per-argument limit is never spawned and recorded as
``arg_too_long``; like a saturated attempt, its node is not expanded further.
"""

import asyncio
import itertools
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import Field, model_validator

from baseline import check_baseline
from common import (
    ArgTooLong,
    BinaryPanicked,
    Measured,
    MemoryKilled,
    Pair,
    PrioritySlot,
    SamplePolicy,
    TimeoutKilled,
    cli_flags,
    exit_if_missing,
    fan_out,
    load_pairs,
    log,
    measure_attempt,
    problem_language,
    run_json_subprocess,
)
from replay_args import ReplayArgs
from schemes import ATTEMPT_SCHEMA, EMPTY_SAMPLE_META, EXPANSION_SCHEMA, PAIR_SCHEMA

# How egg's `StopReason::Saturated` renders through `{:?}`
SATURATED = "Saturated"

# Attempt stop reasons whose node is never expanded.
DEAD_ENDS = (SATURATED, "arg_too_long")

# How many seconds past `max_pair_time` a child may run before it is killed.
TIMEOUT_GRACE = 5.0

# What `report_results` writes once the search finished.
RESULT_FILES = (
    "attempts.parquet",
    "attempts.json",
    "expansions.parquet",
    "pairs.parquet",
    "pools.json",
)


class SearchPolicy(StrEnum):
    DFS = "dfs"
    BFS = "bfs"

    @property
    def frontier(self) -> type[SearchFrontier]:
        return DfsFrontier if self is SearchPolicy.DFS else BfsFrontier


class SearchArgs(ReplayArgs):
    """`ReplayArgs` plus the flags of the search itself."""

    output: Path = Field(
        description="Run folder for `attempts.parquet`/`attempts.json`, created if missing."
    )

    baseline: Path = Field(
        description=(
            "Baseline folder written by `baseline.py`. "
            "It must match this run's budget and cover all of its pairs."
        )
    )

    sample_bin: Path = Field(
        default=Path("target/release/sample"), description="Path to the sample-construction binary."
    )

    # Search budget
    max_pair_time: float | None = Field(
        default=None,
        gt=0,
        description=("Budget for one pair's search (wall time of its `sample`/`attempt`)"),
    )

    max_attempts: int | None = Field(
        default=None, gt=0, description=("Cap on `attempt` processes per pair.")
    )

    # Search policy
    search_policy: SearchPolicy = Field(
        default=SearchPolicy.DFS, description=("DFS vs. BFS of the space")
    )

    guides_per_attempt: int = Field(
        default=1, gt=0, description=("Sibling guides unioned per eqsat.")
    )

    branching: int = Field(
        default=5,
        gt=0,
        description=("Guides drawn each time a node is expanded. -> Branching factor"),
    )

    sampling_backoff: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How often a `sample` process killed at `max_rss` is retried, each "
            "retry giving up one more rewrite-applying iteration. 0 disables retries. "
            "Unlimited if omitted."
        ),
    )

    size_search_steps: int = Field(
        default=200, ge=0, description="How many exact-size-search increments to allow."
    )

    sample_policy: SamplePolicy = Field(
        default=SamplePolicy.Count, description="pool sampling policy."
    )

    frontier: bool = Field(
        default=False, description="Sample only the frontier of terms, not the whole egraph"
    )

    full_union: bool = Field(
        default=True, description="Use the full-union add for the attempt egraph."
    )

    seed: int = Field(default=0, description="RNG seed used in Python and Rust.")

    @model_validator(mode="after")
    def validate_search_budget(self) -> SearchArgs:
        if self.max_pair_time is None and self.max_attempts is None:
            raise ValueError("give at least one of max_pair_time and max_attempts")
        return self

    @model_validator(mode="after")
    def validate_guides_branching(self) -> SearchArgs:
        if self.branching % self.guides_per_attempt != 0:
            raise ValueError("branching must be a multiple of guides_per_attempt")
        return self

    @property
    def timeout(self) -> float | None:
        """Hard timeout of every `sample` and `attempt` process"""
        return None if self.max_pair_time is None else self.max_pair_time + TIMEOUT_GRACE

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
class SearchNode:
    """Node in the search tree

    `guide` is the node array of the guide as origin_lang; `s_expr` is the same
    term lowered, which is what `sample` samples from next.
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
    """The outcome of one `sample` process: the pool drawn and its cost."""

    children: tuple[tuple[list, str], ...]
    status: Literal[
        "ok",
        "empty_pool",
        "no_novel_terms",
        "out_of_memory",
        "binary_panic",
        "timeout",
        "arg_too_long",
    ]
    meta: dict
    wall_time: float

    @property
    def saturated(self) -> bool:
        return self.meta.get("stop_reason") == SATURATED


@dataclass
class Budget:
    """A pair's stop conditions, all charged lazily as the search runs.

    A pool drawn by another pair is charged in full as if this pair had drawn it itself.
    """

    max_time: float | None
    max_attempts: int | None
    spent: float = 0.0

    def charge(self, seconds: float) -> None:
        self.spent += seconds

    def expired(self) -> bool:
        return self.max_time is not None and self.spent >= self.max_time


@dataclass
class PairTrace:
    """Everything one pair's search did, flattened into rows at report time."""

    pair: Pair
    attempts: list[dict] = field(default_factory=list)
    expansions: list[dict] = field(default_factory=list)
    drawn: set[str] = field(default_factory=set)
    stop_reason: str = "unstarted"

    @property
    def root_status(self) -> str:
        """The root expansion's status: whether the pair got a pool at all."""
        return self.expansions[0]["status"] if self.expansions else "unstarted"


# --------
# Sampling
# --------


async def run_capped(
    args: SearchArgs, limit: PrioritySlot, cmd: list[str], what: str
) -> tuple[Measured | None, float]:
    """Run under the RSS cap, retrying a killed child up to `sampling_backoff`
    times: the first retry replays the iterations that completed, each further
    one gives up another iteration that applied a rewrite. Gives up early once
    no such iteration is left.

    Also returns the wall time summed over every try, the killed ones included.
    A `BinaryPanicked` or `TimeoutKilled` is not retried and carries that same sum."""
    cmd = [*cmd, "--print-success-iters"]
    cap = args.max_rss_bytes
    timeout = args.timeout
    # Try for the first time, record list of successful productive eqsat iterations
    try:
        measured = await run_json_subprocess(
            cmd, what=what, max_rss_bytes=cap, limit=limit, timeout=timeout
        )
        return measured, measured.wall_time
    except MemoryKilled as killed:
        wall_time = killed.wall_time
        candidates = killed.productive_iters

    for max_iters in itertools.islice(reversed(candidates), args.sampling_backoff):
        # Set or replace `--max-iters`.
        try:
            index = cmd.index("--max-iters")
            cmd[index + 1] = str(max_iters)
        except ValueError:
            cmd.extend(["--max-iters", str(max_iters)])
        try:
            measured = await run_json_subprocess(
                cmd, what=what, max_rss_bytes=cap, limit=limit, timeout=timeout
            )
            return measured, wall_time + measured.wall_time
        except MemoryKilled as killed:
            wall_time += killed.wall_time
        except (BinaryPanicked, TimeoutKilled) as failed:
            failed.wall_time += wall_time
            raise

    return None, wall_time


async def draw_expansion(
    args: SearchArgs, sample_flags: list[str], limit: PrioritySlot, s_expr: str
) -> Expansion:
    """Run one `sample` process from `s_expr`. This is called recursively at every depth"""
    cmd = [
        str(args.sample_bin),
        *sample_flags,
        *cli_flags(**args.limits, start=s_expr, n_samples=args.branching),
    ]

    try:
        measured, wall_time = await run_capped(args, limit, cmd, f"sample for term {s_expr!r}")
    except ArgTooLong as too_long:
        log(f"WARNING: {too_long}")
        return Expansion((), "arg_too_long", dict(EMPTY_SAMPLE_META), 0.0)
    except BinaryPanicked as panicked:
        panicked.warn()
        return Expansion((), "binary_panic", dict(EMPTY_SAMPLE_META), panicked.wall_time)
    except TimeoutKilled as timed_out:
        log(f"WARNING: {timed_out}")
        return Expansion((), "timeout", dict(EMPTY_SAMPLE_META), timed_out.wall_time)

    # A capped-out child never printed its `Measured` envelope.
    if measured is None:
        return Expansion((), "out_of_memory", dict(EMPTY_SAMPLE_META), wall_time)

    # An empty payload is `sample` reporting that construction failed.
    if not measured.payload:
        meta = {**EMPTY_SAMPLE_META, "peak_rss": measured.peak_rss}
        return Expansion((), "no_novel_terms", meta, wall_time)

    record = measured.payload[0]
    children = tuple(zip(record["samples"], record["samples_s_expr"], strict=True))
    meta = {
        "iters": record["iters"],
        "nodes": record["nodes"],
        "classes": record["classes"],
        "total_time": record["time"],
        "final_live_heap": record["final_live_heap"],
        "peak_live_heap": record["peak_live_heap"],
        "stop_reason": record["stop_reason"],
        "peak_rss": measured.peak_rss,
    }
    return Expansion(children, "ok" if children else "empty_pool", meta, wall_time)


class SamplePools:
    """Every `sample` task the run started, keyed by the start term it sampled from.

    This allows lazy sampling! Later callers simply get the cached results

    The sample draws are all in the same `group` so a failed pair cancels
    sample tasks still running instead of leaving them un-awaited
    """

    def __init__(
        self,
        args: SearchArgs,
        sample_flags: list[str],
        limit: PrioritySlot,
        group: asyncio.TaskGroup,
    ) -> None:
        self.args = args
        self.sample_flags = sample_flags
        self.limit = limit
        self.group = group
        self.tasks: dict[str, asyncio.Task[Expansion]] = {}

    def __len__(self) -> int:
        return len(self.tasks)

    async def draw(self, s_expr: str) -> Expansion:
        """That term's pool, drawing it only if this run has not already.

        The Pool may cache for the whole run but it simulates being pair independence.
        """
        # No `await` before the task finishes, so no race
        if s_expr not in self.tasks:
            self.tasks[s_expr] = self.group.create_task(
                draw_expansion(self.args, self.sample_flags, self.limit, s_expr)
            )
        return await self.tasks[s_expr]

    def drawn(self) -> dict[str, Expansion]:
        """Serialize finished draws for reporting."""
        return {
            s_expr: task.result()
            for s_expr, task in self.tasks.items()
            if task.done() and not task.cancelled() and task.exception() is None
        }


# ------
# Search
# ------


class SearchFrontier(ABC):
    """The work queue with the drawing logic once the queue runs empty.

    Children are attempted in batches of `guides_per_attempt` siblings, in the
    order `sample` returned them; a trailing partial batch is dropped.
    Nodes to descend are remembered via `queue_expansion`.
    """

    def __init__(
        self,
        guides_per_attempt: int,
        root: str,
        pools: SamplePools,
        trace: PairTrace,
        budget: Budget,
    ) -> None:
        self.guides_per_attempt = guides_per_attempt
        self.pools = pools
        self.trace = trace
        self.budget = budget
        # The first `pop` draws the root's pool.
        self.root = SearchNode(node_id=0, parent_id=None, depth=0, guide=None, s_expr=root)
        self._ids = itertools.count(1)
        self._seen: set[str] = set()

    @abstractmethod
    def queue_expansion(self, batch: tuple[SearchNode, ...]) -> None:
        """Queue `batch`'s nodes to have their pools drawn later."""

    @abstractmethod
    async def pop(self) -> tuple[SearchNode, ...] | None:
        """The next batch to attempt, or `None` once the frontier or budget ran out."""

    async def _expand(self, node: SearchNode) -> list[tuple[SearchNode, ...]]:
        """Draw `node`'s pool, record what it cost, and return the unseen children batched.

        A saturated replay marks its children terminal, so the search attempts them
        but never samples past them.
        """
        started_at = self.budget.spent
        cached = node.s_expr in self.trace.drawn
        self.trace.drawn.add(node.s_expr)
        expansion = await self.pools.draw(node.s_expr)
        # A pool this pair already drew cost it nothing but the lookup.
        wall_time = 0.0 if cached else expansion.wall_time
        self.budget.charge(wall_time)

        children = []
        for guide, s_expr in expansion.children:
            key = json.dumps(guide)
            if key in self._seen:
                continue
            self._seen.add(key)
            children.append(
                SearchNode(
                    next(self._ids),
                    node.node_id,
                    node.depth + 1,
                    guide,
                    s_expr,
                    terminal=expansion.saturated,
                )
            )
        k = self.guides_per_attempt
        batches = [tuple(children[i : i + k]) for i in range(0, len(children) - k + 1, k)]

        self.trace.expansions.append(
            {
                "start": self.trace.pair.start,
                "goal": self.trace.pair.goal,
                "node_id": node.node_id,
                "depth": node.depth,
                "status": expansion.status,
                "cached": cached,
                "saturated": expansion.saturated,
                "drawn": len(expansion.children),
                "pushed": len(batches) * k,
                "started_at": started_at,
                "wall_time": wall_time,
                **expansion.meta,
            }
        )
        return batches


class DfsFrontier(SearchFrontier):
    """Follows one chain down before trying its siblings.

    The stack holds undrawn nodes and ready batches; a node is drawn only once
    it reaches the top, and its batches replace it.
    """

    def __init__(
        self,
        guides_per_attempt: int,
        root: str,
        pools: SamplePools,
        trace: PairTrace,
        budget: Budget,
    ) -> None:
        super().__init__(guides_per_attempt, root, pools, trace, budget)
        self._stack: deque[SearchNode | tuple[SearchNode, ...]] = deque([self.root])

    def queue_expansion(self, batch: tuple[SearchNode, ...]) -> None:
        # Reversed, so the first sibling ends up on top.
        self._stack.extend(reversed(batch))

    async def pop(self) -> tuple[SearchNode, ...] | None:
        while self._stack and not self.budget.expired():
            top = self._stack.pop()
            if isinstance(top, tuple):
                return top
            # Loop again, so a draw that used up the budget hands out nothing.
            self._stack.extend(reversed(await self._expand(top)))
        return None


class BfsFrontier(SearchFrontier):
    """Exhausts a depth before descending, so it only draws once no batch is ready."""

    def __init__(
        self,
        guides_per_attempt: int,
        root: str,
        pools: SamplePools,
        trace: PairTrace,
        budget: Budget,
    ) -> None:
        super().__init__(guides_per_attempt, root, pools, trace, budget)
        # Nodes whose pool is still to be drawn.
        self._to_expand: deque[SearchNode] = deque([self.root])
        self._ready: deque[tuple[SearchNode, ...]] = deque()

    def queue_expansion(self, batch: tuple[SearchNode, ...]) -> None:
        self._to_expand.extend(batch)

    async def pop(self) -> tuple[SearchNode, ...] | None:
        while self._to_expand and not self._ready and not self.budget.expired():
            self._ready.extend(await self._expand(self._to_expand.popleft()))
        return self._ready.popleft() if self._ready else None


async def search_pair(
    args: SearchArgs, base_flags: list[str], limit: PrioritySlot, pools: SamplePools, pair: Pair
) -> PairTrace:
    """Run one pair's sampling/attempt search and return its trace.

    The loop is: pop a batch of siblings, attempt them together, and on failure
    hand them back to the frontier, which draws their pools once it needs the
    children. A batch whose attempt saturated is never handed back, since its
    subtrees cannot hold the goal. Each attempt is a separate `attempt` process,
    so its `peak_rss` is that attempt's own peak rather than a high-water mark
    shared across the pair.
    """
    budget = Budget(max_time=args.max_pair_time, max_attempts=args.max_attempts)
    trace = PairTrace(pair)
    frontier = args.search_policy.frontier(
        args.guides_per_attempt, pair.start, pools, trace, budget
    )

    while True:
        # Time is checked first, so a pair out of both budgets counts as `time_exhausted`.
        if budget.expired():
            trace.stop_reason = "time_exhausted"
            break
        if budget.max_attempts is not None and len(trace.attempts) >= budget.max_attempts:
            trace.stop_reason = "attempt_budget_exhausted"
            break
        batch = await frontier.pop()
        if batch is None:
            # Every node hit a saturated egraph,
            # was pruned as a dead end, or the pools ran dry; `setup_status`, the
            # expansion rows' `saturated` flag, and the attempt rows'
            # `stop_reason` tell those apart.
            trace.stop_reason = "time_exhausted" if budget.expired() else "frontier_exhausted"
            break

        assert all(node.guide is not None for node in batch), (
            "the root is a baseline, not an attempt"
        )
        first = batch[0]
        started_at = budget.spent
        cmd = [
            str(args.attempt_bin),
            *base_flags,
            *cli_flags(
                goal=pair.goal,
                is_guide=True,
                start=json.dumps([node.guide for node in batch]),
                full_union=args.full_union,
            ),
        ]
        # A killed or panicked attempt comes back as a failed one rather than an
        # exception, since the search simply moves on to the next node.
        attempt = await measure_attempt(
            cmd,
            what=f"attempt for goal {pair.goal!r}",
            max_rss_bytes=args.max_rss_bytes,
            limit=limit,
            timeout=args.timeout,
        )
        budget.charge(attempt.wall_time)
        trace.attempts.append(
            {
                "start": pair.start,
                "goal": pair.goal,
                "sample_policy": args.sample_policy,
                # Counted from 1, so the n-th attempt of a pair has `attempt == n`.
                "attempt": len(trace.attempts) + 1,
                "node_ids": [node.node_id for node in batch],
                "parent_id": first.parent_id,
                "depth": first.depth,
                "guide_s_exprs": [node.s_expr for node in batch],
                "terminal": first.terminal,
                "started_at": started_at,
                "wall_time": attempt.wall_time,
                "peak_rss": attempt.peak_rss,
                **attempt.summary,
            }
        )

        if attempt.summary["reached"]:
            trace.stop_reason = "reached"
            break

        # A terminal batch came out of a saturated egraph, so sampling from it
        # would rebuild that same egraph and redraw that same pool. Not queueing
        # it sends the search back up to whatever the frontier holds.
        # A saturated attempt is a dead end as well, we wont get past that
        # so we don't even queue it for expansion. Same for guides too long
        # to pass to `attempt` at all.
        if attempt.summary["stop_reason"] not in DEAD_ENDS and not first.terminal:
            frontier.queue_expansion(batch)

    return trace


# ---------
# Reporting
# ---------


def summarize_pair(args: SearchArgs, trace: PairTrace) -> dict:
    """Collapse one search into a single guided-workflow row."""
    attempts = trace.attempts
    successes = [attempt for attempt in attempts if attempt["reached"]]

    # `attempt_peak_rss` is the one that reached the goal, or the last one tried if none did.
    # `attempt_peak_rss_max` is the max across every attempt run, which is the pair cost end to end.
    decisive = successes[0] if successes else (attempts[-1] if attempts else None)
    attempt_peak = decisive["peak_rss"] if decisive else None

    attempt_peaks = [attempt["peak_rss"] for attempt in attempts if attempt["peak_rss"] is not None]
    attempt_peak_max = max(attempt_peaks, default=None)

    expansion_peaks = [exp["peak_rss"] for exp in trace.expansions if exp["peak_rss"] is not None]
    sample_peak = max(expansion_peaks, default=None)

    rss_peaks = [peak for peak in (sample_peak, attempt_peak_max) if peak is not None]
    live_peaks = [
        peak
        for peak in (
            *(exp.get("peak_live_heap") for exp in trace.expansions),
            *(attempt.get("peak_live_heap") for attempt in attempts),
        )
        if peak is not None
    ]

    # A pool that drew nothing usable is a setup failure; a pool that queued
    # nodes the search never got to is not — that is `search_stop_reason`'s job.
    setup_status = trace.root_status
    if setup_status == "ok" and not trace.expansions[0]["pushed"]:
        setup_status = "empty_pool"

    return {
        "start": trace.pair.start,
        "goal": trace.pair.goal,
        "sample_policy": args.sample_policy,
        "search_policy": args.search_policy,
        "branching": args.branching,
        "guides_per_attempt": args.guides_per_attempt,
        "max_attempts": args.max_attempts,
        "max_pair_time": args.max_pair_time,
        "guided_success": bool(successes),
        "search_stop_reason": trace.stop_reason,
        "success_attempt": successes[0]["attempt"] if successes else None,
        "success_depth": successes[0]["depth"] if successes else None,
        "attempts_run": len(attempts),
        "expansions_run": len(trace.expansions),
        "expansions_paid": sum(not exp["cached"] for exp in trace.expansions),
        "saturated_expansions": sum(exp["saturated"] for exp in trace.expansions),
        # It does not make sense to draw guides from a saturated egraph.
        "root_saturated": bool(trace.expansions and trace.expansions[0]["saturated"]),
        "deepest_attempt": max((attempt["depth"] for attempt in attempts), default=None),
        "wall_time": sum(exp["wall_time"] for exp in trace.expansions)
        + sum(attempt["wall_time"] for attempt in attempts),
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
        "root_status": trace.root_status,
        "attempt_peak_rss": attempt_peak,
        "attempt_peak_rss_max": attempt_peak_max,
        "sample_peak_rss": sample_peak,
        "guided_peak_rss": max(rss_peaks, default=None),
        "guided_peak_live_heap": max(live_peaks, default=None),
    }


def write_sample_pools(pools: SamplePools, out: Path) -> None:
    """Dump every pool the run drew, keyed by the term it was sampled from."""
    dumped = {
        s_expr: {
            "status": expansion.status,
            "samples": [guide for guide, _ in expansion.children],
            "samples_s_expr": [child for _, child in expansion.children],
            **expansion.meta,
        }
        for s_expr, expansion in pools.drawn().items()
    }
    (out / "pools.json").write_text(json.dumps(dumped))


def report_results(
    args: SearchArgs,
    traces: list[PairTrace],
    pools: SamplePools,
    elapsed: float,
) -> str:
    """Write attempt, expansion, and pair results, and return a one-line summary.

    The baseline stays in its own folder, which `config.json` names under `baseline`.
    """
    out = args.output

    attempt_rows = [row for trace in traces for row in trace.attempts]
    expansion_rows = [row for trace in traces for row in trace.expansions]

    attempts = pl.DataFrame(attempt_rows, schema=ATTEMPT_SCHEMA)
    attempts.write_parquet(out / "attempts.parquet")
    (out / "attempts.json").write_text(json.dumps(attempt_rows, indent=2))

    expansions = pl.DataFrame(expansion_rows, schema=EXPANSION_SCHEMA)
    expansions.write_parquet(out / "expansions.parquet")

    pairs = pl.DataFrame([summarize_pair(args, trace) for trace in traces], schema=PAIR_SCHEMA)
    pairs.write_parquet(out / "pairs.parquet")

    write_sample_pools(pools, out)

    reached_pairs = int(pairs["guided_success"].sum())
    total_pairs = len(pairs)
    reach_rate = reached_pairs / total_pairs if total_pairs else 0.0
    attempts_run = int(pairs["attempts_run"].sum())
    return (
        f"Reached {reached_pairs}/{total_pairs} start/goal pairs "
        f"(reach rate {reach_rate:.2f}) in {attempts_run} attempt(s), "
        f"{len(pools)} distinct sample pool(s) and {timedelta(seconds=round(elapsed))}. "
        f"Wrote {out / 'pairs.parquet'}"
    )


async def search(
    args: SearchArgs, limit: PrioritySlot, desc: str = "search", leave: bool = True
) -> str:
    """Run the whole search, write its results, and return their one-line summary.

    Raises `BaselineMismatch` before any work is spent if the baseline does not fit.
    """
    started = time.monotonic()
    exit_if_missing(args.sample_bin, args.attempt_bin)
    language = problem_language(args.path)
    base_flags = args.base_flags(language)
    sample_flags = args.sample_flags(language)

    pairs = load_pairs(args.path, args.n_starts, args.n_goals)
    check_baseline(args.baseline, args.baseline_key(), pairs)

    args.output.mkdir(parents=True, exist_ok=True)
    # Written first, so `experiment.py --rerun-aborted` can rerun a failed search.
    config = {**args.model_dump(), "effective_limits": args.limits}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str))

    log(f"Searching {len(pairs)} (start, goal) pair(s)")
    # Only the processes hold slots, so a pair waiting on another pair's pool holds none.
    # The pools run in this task group, so a failed pair cancels them too.
    async with asyncio.TaskGroup() as group:
        pools = SamplePools(args, sample_flags, limit, group)
        traces = await fan_out(
            lambda pair: search_pair(args, base_flags, limit, pools, pair),
            pairs,
            desc,
            unit="pair",
            leave=leave,
        )

    summary = report_results(args, traces, pools, time.monotonic() - started)
    log(summary)
    return summary
