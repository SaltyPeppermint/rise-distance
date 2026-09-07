# Root-restricted novel-size search and exact package construction

`FrontierPackage::build_through_novel_sizes` finds the requested novel
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
cap = start_size + search_steps

prev_lookup = reconstruct_previous_boundary()
cap_budgets = root_budgets(curr, root, cap)
matches = enumerate_matches_rooted(curr, prev_lookup, cap_budgets)
drop(prev_lookup)

final_max_size = find_novel_root_sizes(
    curr, root, matches, min_extractable, cap_budgets
)

if fewer than min_extractable terms were found:
    return Err(cap)

final_budgets = root_budgets(curr, root, final_max_size)
prune_matches(curr, matches, final_budgets)

plain = count_plain_rooted(curr, final_budgets)
joint = count_joint_rooted(curr, matches, final_budgets)
package = derive_novel_and_retain_package_data(plain, joint, matches)
```

`FrontierPackage::build(result, max_size)` already knows its final
limit. It computes final budgets immediately, enumerates matches inside that
domain, and builds the package without the cap scan.

`start_size` and `search_steps` define only the scan cap. There
is no retry schedule. On success, both the returned size and the package limit
are the smallest novel root size at which the cumulative novel term count
reaches `min_extractable`.

`PlainPackage::build_through_sizes` has the same shape with no previous
boundary to subtract: it skips match enumeration entirely and scans
`find_plain_root_size` over plain root counts, where every extractable term
counts toward `min_extractable`.

## How `search_steps` drives memory

`search_steps` looks like a retry counter, but it is not one: it only ever
enters the code as `cap = start_size + search_steps`, and `cap` is the size
limit that every pre-scan structure is sized against. Raising it costs memory
in four places, none of which depend on the size the scan eventually picks.

**Budgets admit more classes.** `class_budgets` seeds the root with `cap` and
propagates `child_budget = (parent_budget - 1) - (mins_sum - min_size(child))`
downward, keeping a class only when some node fits. A larger `cap` raises every
reachable class's budget and additionally admits classes whose cheapest
realization did not fit the old cap. The budget map is the domain of everything
downstream.

**The DP skeleton is allocated for that whole domain up front.**
`plain_dp_rooted` builds `children_of` — canonical child ids per node — for
every budgeted class, and `LayeredDp::new` allocates one suffix table per node
position for each of them, before a single layer is stepped.

**Every layer touches more classes.** `step` works on
`budgets.iter().filter(|(_, &budget)| size <= budget)`. Higher budgets keep more
classes active at *every* layer, early ones included, so both the retained
histograms and the suffix tables grow denser per layer rather than just taller.

**On the frontier path, match enumeration runs at the cap.**
`enumerate_matches_rooted` visits every node whose minimum realization fits its
cap budget and retains the bottom-up cover fixpoint used to discover matches.
That table is built against `cap_budgets` and is only narrowed by
`prune_matches` *after* the scan has chosen `final_max_size`, so peak RSS
carries the cap-sized match data even when the selected size is small.

### Success stops early; failure pays the full cap

The scan returns as soon as the cumulative term count reaches
`min_extractable`, so the number of layers actually stepped is bounded by the
selected size, not by `cap`. When the search succeeds quickly, `search_steps`
costs only the four items above.

When it cannot be satisfied, the loop runs all `rooted.limit() == cap` layers
over the full admitted class set before returning `Err(cap)`. `search_steps`
therefore bounds the cost of *failing*, and it does so superlinearly: more
layers, each over more classes, each retaining more entries.

This is the practical trap. Under an external RSS cap, raising `search_steps`
converts start terms that used to complete the scan and report "too few terms"
into start terms that are killed before they can report anything — the give-up
path is the most expensive path there is. Widening the search to rescue
borderline start terms costs the most on exactly the start terms it cannot
rescue.

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
