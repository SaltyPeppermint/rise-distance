# Low-memory rejection-candidate construction

The rejection-backed path constructs exact-size terms from the current e-graph
without building exact term counts. It then rejects terms that existed at the
previous e-graph boundary and removes duplicates within the candidate pool.
This trades completeness and distribution guarantees for a substantially
smaller construction state.

The relevant implementation is:

- [`src/candidates/rejection.rs`](../../src/candidates/rejection.rs) — proposal
  engines, rejection, scheduling, limits, and telemetry.
- [`src/previous.rs`](../../src/previous.rs) — exact membership lookup at the
  previous boundary.
- [`src/bin/candidates.rs`](../../src/bin/candidates.rs) — CLI pool selection
  and package setup.

For the count-backed alternative, see
[exact novel-candidate construction](exact_novel_candidates.md).

## Pipeline

Both rejection pools use the same outer pipeline:

```text
current root + target size
        │
        ▼
 proposal engine ── construction failure ──▶ retry within limits
        │ exact-size current extraction
        ▼
 previous-boundary lookup ── present ──────▶ reject as non-novel
        │ absent
        ▼
 batch uniqueness set ────── duplicate ────▶ reject as duplicate
        │ new
        ▼
 accepted exact-novel candidate
```

`RejectionCandidatePackage` borrows the final current e-graph, reconstructs a
compact `PrevIndex`, canonicalizes the current root, and computes
root-restricted size budgets through the configured cap. It does not construct
`BigUint` histograms, current/previous joint counts, node-match tables, or
suffix convolutions.

The cap is:

```text
seed AST size + max_retries * retry_step
```

The same cap knobs are used by exact size discovery, but the rejection path
does not perform an exact novel-size scan.

## Exact novelty filtering

Proposal engines may return any extraction of the current root at the requested
size. Novelty is decided afterwards by `PrevIndex::contains_origin_expr`:

1. Traverse the concrete expression bottom-up.
2. Ignore origin annotations and translate each child to its previous canonical
   e-class.
3. Look up the translated node at the previous boundary.
4. Accept the root only when that lookup fails.

Consequently, every accepted candidate is:

- extractable from the current root;
- exactly the requested AST size;
- absent from every e-class at the previous boundary; and
- distinct from every earlier accepted candidate in the same pool.

Novelty is exact even though proposal and size discovery are probabilistic.

## Proposal engines

### `rejection_walk`

`RandomWalkEngine` performs a randomized, bounded top-down construction. For
each `(e-class, size)` state it:

1. Keeps e-nodes whose children can fit at their minimum extraction sizes.
2. Shuffles those nodes.
3. Randomly partitions the remaining size among the children.
4. Recurses until it constructs a term or exhausts
   `--rejection-walk-backtrack` fuel.

Its scheduled size domain is every size from the seed AST size through the
cap. A proposal can return `None` even when a term of that size exists: bounded
backtracking is an operational limit, not a reachability proof. These events
are reported as `construction_failures`.

This engine retains only the current e-graph, previous index, root budgets, and
temporary recursion state. It is the lowest-memory option.

### `rejection_feasible`

`FeasibilityEngine` first builds one reachability bit for each retained
`(current e-class, exact size)` state. The bits are computed in increasing size
order, so cycles in the e-graph do not require a fixpoint: every child of a
size-`s` tree has size below `s`.

The engine schedules exactly the plain-reachable root sizes. During
construction it selects among feasible e-nodes and feasible child-size
partitions, so every proposal for a reported size succeeds. Selection is local
and randomized; it is not uniform over complete terms.

The flat bitset costs approximately:

```text
retained current classes * ceil((cap + 1) / 64) machine words
```

It uses more memory than the random walk but avoids construction failures and
probing sizes having no current-root extraction.

## Size discovery and allocation

Candidate sizes are discovered empirically. The scheduler visits active sizes
in rounds so an impossible or old-only size cannot consume the entire pool
budget before other sizes are tried. A size is *observed novel* after its first
accepted proposal.

Discovery stops when `--novel-size-goal` sizes have been observed or no active
size can be attempted. The selected sizes are the smallest observed sizes, not
necessarily the mathematically smallest novel sizes. Failure to observe a size
does not prove that its novel frontier is empty.

After discovery:

- `greedy` continues filling from the selected sizes in ascending order.
- `uniform` computes an even per-size target and emits accepted terms
  round-robin across the selected sizes.
- `proportional:<min>` is unsupported because it requires exact per-size term
  counts.

Both supported allocations may return fewer candidates than requested when a
limit is reached, proposals repeatedly fail, novelty acceptance is low, or the
available unique frontier is too small.

## Bounds and stop reasons

The rejection path is bounded by independent controls:

| CLI flag | Scope |
| --- | --- |
| `--rejection-walk-backtrack` | Recursive state and partition visits for one random-walk proposal. |
| `--rejection-attempts-per-size` | Proposal attempts for one target size. |
| `--rejection-global-attempts` | Proposal attempts for the whole pool. |
| `--rejection-max-time` | Wall-clock time for the whole pool. |
| `--max-memory` | Absolute live-heap ceiling shared with guide replay. |

Memory is checked before and after material package allocations, after the
feasibility bitset is built, and periodically during proposal attempts. The
periodic check runs every 64 attempts to avoid making jemalloc epoch updates
the dominant cost on small graphs.

Telemetry reports one of these stop reasons:

- `quota_filled` — the requested number of unique candidates was returned;
- `memory_limit` — the live heap exceeded `--max-memory`;
- `time_limit` — the pool wall-clock limit elapsed;
- `global_attempt_limit` — the pool exhausted its global proposal budget;
- `quota_or_size_budget` — per-size budgets or available active sizes were
  exhausted before filling the quota; or
- `no_novel_candidate_observed` — no candidate was accepted before ordinary
  size/quota exhaustion.

Budget exhaustion never proves that the exact novel frontier is empty.

## Determinism and telemetry

Each size receives its own RNG stream derived from:

```text
(candidate seed, pool salt, target size)
```

The walk and feasibility pools have different salts, so they do not share
random streams. A fixed graph, arguments, pool, and candidate seed produce a
stable sequence.

The binary logs aggregate `rejection_stats` and per-size `rejection_size`
records to stderr. Per-size fields are:

- `proposal_attempts`;
- `construction_failures`;
- `non_novel_rejections`;
- `duplicate_rejections`; and
- `accepted_unique_terms` plus its acceptance rate.

The aggregate record also includes elapsed seconds, peak live heap, and the
stop reason.

## CLI and guided-search usage

Build the candidate binary:

```bash
cargo build --release --bin candidates
```

Construct a random-walk pool directly:

```bash
target/release/candidates \
    --language math \
    --seed '(+ x 0)' \
    --max-iters 38 \
    --max-nodes 1000000 \
    --max-time 10 \
    --max-memory 2G \
    --candidate-pool rejection_walk \
    --candidates-per-pool 1000 \
    --candidate-seed 42 \
    --size-allocation uniform \
    --novel-size-goal 5
```

Use either pool through the Python driver:

```bash
uv run scripts/guided_search.py data/seed_terms/example \
    --stop-memory 2G \
    --strategy no_replacement_rejection_walk

uv run scripts/guided_search.py data/seed_terms/example \
    --stop-memory 2G \
    --strategy no_replacement_rejection_feasible
```

The `with_replacement_*` and `no_replacement_*` prefixes describe how Python
chooses guide subsets from the finite constructed pool. They do not change the
Rust proposal or rejection algorithm.

## Comparison with exact construction

| Property | Exact count-backed path | Rejection walk | Rejection feasible |
| --- | --- | --- | --- |
| Accepted candidates are exactly novel | Yes | Yes | Yes |
| Exact-size construction | Yes | Yes, when a proposal succeeds | Yes |
| Exact plain-reachable sizes | Yes | No | Yes |
| Smallest novel sizes known exactly | Yes | No | No |
| Count-proportional selection | Available | Unavailable | Unavailable |
| Construction failure for a reachable size | No | Possible | No |
| Main retained state | Big counts, matches, suffix tables | Root budgets | Root budgets and reachability bits |

When exact and rejection pools are requested together, exact counting still
determines peak candidate-construction memory. Use a rejection-only manifest
when measuring the low-memory path.
