# Root-restricted novel-size search and exact package construction

`ExactCandidatePackage::build_through_novel_sizes` finds the requested novel
root sizes and builds all data needed for exact drawing without enumerating or
counting classes, e-nodes, or current/previous pairs that the selected root
cannot use.

The relevant code is:

- [src/candidates/count/layered.rs](../../src/candidates/count/layered.rs) — shared
  current-root budgets and the generic layered DP.
- [src/candidates/count/novel.rs](../../src/candidates/count/novel.rs) — rooted
  match enumeration, pruning, joint counting, and the exact scan.
- [src/candidates/package.rs](../../src/candidates/package.rs) — phase
  ordering, package retention, and telemetry.

For the counting recurrence, first read
[size-layered term counting](layered_counting.md).

## Control flow

The final candidate size is not known until the exact scan completes. The scan
cap bounds the first rooted match enumeration; the result is tightened after
the final size is selected:

```text
cap = start_size + max_retries * retry_step

prev_lookup = reconstruct_previous_boundary()
cap_budgets = root_budgets(curr, root, cap)
matches = enumerate_matches_rooted(curr, prev_lookup, cap_budgets)
drop(prev_lookup)

novel_sizes = find_novel_root_sizes_rooted(
    curr, root, matches, stop_after=sizes, cap_budgets
)

if fewer than sizes were found:
    return Err(cap)

final_max_size = novel_sizes[sizes - 1]
final_budgets = root_budgets(curr, root, final_max_size)
prune_matches(curr, matches, final_budgets)

plain = count_plain_rooted(curr, final_budgets)
joint = count_joint_rooted(curr, matches, final_budgets)
package = derive_novel_and_retain_package_data(plain, joint, matches)
```

`ExactCandidatePackage::build(result, max_size)` already knows its final
limit. It computes final budgets immediately, enumerates matches inside that
domain, and builds the package without the cap scan.

`start_size`, `max_retries`, and `retry_step` define only the scan cap. There
is no retry schedule. On success, both the returned size and the package limit
are the `sizes`-th smallest novel root size.

## Budget-aware match enumeration

For each relevant current class `c`, rooted enumeration visits only nodes
whose cheapest possible realization fits:

```text
1 + sum(min_size(canonical_child)) <= budget(c)
```

It retains the bottom-up fixpoint used to discover matches. A current node
`n(c_1, ..., c_k)` tries the Cartesian product of the previous classes in each
child's discovered cover. Replacing current children by one such tuple gives a
translated node that can be queried in the complete previous lookup. A
successful lookup records:

```text
(current_class, node_index, previous_class, previous_children)
```

and grows the current class's cover. Passes continue until no match is added.

Previous classes are never filtered by a previous-root notion. A previous
class remains eligible whenever it witnesses a relevant current term.

### Why rooted enumeration is complete

Take a non-novel term `t` of size at most `budget(c)` extractable from relevant
current class `c`.

- If `t` is a leaf, the rooted pass queries that leaf directly.
- Otherwise every child subtree is smaller than `t`, is shared by its current
  and previous child classes, and fits the child current class's propagated
  root budget.
- By induction on term size, every child cover entry needed to translate the
  parent is discovered.
- The Cartesian product therefore visits the parent tuple and discovers its
  previous witness.

Cycles do not affect this argument because a finite extracted tree strictly
decreases in size from parent to child.

## Exact incremental scan

Plain and joint counts advance together one size layer at a time. The joint
key is `(current_class, previous_class)`, and every pair inherits its current
class budget. After layer `s`, the root count is final:

```text
novel(root, s) = plain(root, s)
               - sum_pc joint((root, pc), s)
```

The scan uses `BigUint`, so nonzero detection is exact. It records nonzero
sizes in ascending order and stops at the requested count; no larger scan
layer or exact-drawing suffix cache is constructed.

The final package is a separate pass because drawing needs complete rooted
histograms and plain suffix tables through `final_max_size`, potentially with
the caller's counter type rather than `BigUint`.

## Final-size pruning and retained data

Cap enumeration can retain nodes that fit the scan cap but cannot participate
at the selected final size. Before package counting, `prune_matches` removes:

- entries whose current class is absent from the final budgets; and
- entries whose current e-node minimum exceeds its final class budget.

Every previous witness attached to a surviving current node is retained.
Rooted joint counting decides which current/previous pairs have nonzero cells
within budget. The package retains only:

- rooted plain histograms and exact-drawing suffix tables;
- nonempty rooted joint histograms;
- cover entries derived from nonempty joint keys;
- the final-budget-pruned node-match table; and
- novel histograms derived from those rooted counts.

## Correctness invariant

For a selected root `r` and final limit `M`, every query reachable while
constructing a term of size at most `M` remains available:

- the exact novel histogram at `r` through `M`;
- recursive plain and joint histogram lookups;
- covers and node matches needed by every feasible frontier state; and
- suffix tables needed to split child sizes.

The restriction removes only data no root derivation can query. Every drawn
candidate remains extractable from the current graph and absent from the
previous boundary.

## Diagnostics

`build_through_novel_sizes` writes a diagnostic to its supplied log only when it
cannot build a package with the requested number of novel sizes. A successful
call does not write a structural or live-heap summary. Call `log_root_counts`
explicitly after success when a sorted frontier-size histogram is needed.

## Verification

Tests exercise acyclic and cyclic graphs, empty novel frontiers, merged and
repeated children, a parent match discovered through child cover information,
unreachable matching classes, oversized nodes removed by final pruning, and
multiple previous witness classes for one current class. The end-to-end
backoff fixture has novel sizes `5, 7, 9, ...`; requesting three sizes selects
and builds the package at `9`.
