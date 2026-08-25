# Size-layered, root-restricted term counting

Candidate construction counts exact-size terms only for current e-classes and
sizes that can participate in an extraction from the selected root. This
document describes the shared size-layered kernel and the root budgets that
bound every production counting phase.

The relevant code is:

- [src/candidates/count/layered.rs](../../src/candidates/count/layered.rs) —
  `RootBudgets`, `root_budgets`, and `LayeredDp`.
- [src/candidates/draw/plain.rs](../../src/candidates/draw/plain.rs) —
  top-down consumption of rooted plain histograms and suffix tables.
- [src/candidates/count/novel.rs](../../src/candidates/count/novel.rs) — reuse of
  the kernel for rooted joint counting and the novel-size scan.
- [src/analysis/semilattice/ast_size.rs](../../src/analysis/semilattice/ast_size.rs)
  — minimum extraction sizes used by the budget calculation.

## Exact-size recurrence

For a current e-class `c`, the sparse histogram

```text
plain(c, s) = number of terms of exactly size s extractable from c
```

uses AST node count as size. For an e-node `f(c_1, ..., c_k)`, the node costs
one and its children share the remaining `s - 1` nodes:

```text
plain(c, s) = sum over nodes f(c_1, ..., c_k) in c
              sum over s_1 + ... + s_k = s - 1
              product_i plain(c_i, s_i)
```

Every child size is strictly smaller than `s`. Counts at layer `s` therefore
depend only on completed earlier layers, even if the e-graph contains cycles.
For example, `X = {a, f(X)}` unfolds by size:

```text
size 1: a
size 2: f(a)
size 3: f(f(a))
...
```

There is a cycle between e-classes but no cycle between expanded
`(e-class, size)` states.

## The `LayeredDp` kernel

`LayeredDp<K, C>` is generic over a state key `K` and counter type `C`. Plain
counting uses `K = Id`; joint current/previous counting uses
`K = (Id, Id)`.

For each key and node with children `k_1..k_n`, it retains suffix convolution
tables:

```text
S_i(t) = ways to fill children i..n with total size t
S_n    = {0: 1}
```

At size layer `s`, it extends each suffix by the single total `t = s - 1`:

```text
S_i(t) = sum over sigma of histogram(k_i, sigma) * S_(i+1)(t - sigma)
```

Both factors are final. `sigma >= 1`, so even the suffix lookup is at a total
strictly below the entry being added. After all suffixes have been extended,
the key publishes its layer count by summing `S_0(s - 1)` over its nodes.
Separating suffix extension from publication makes the result independent of
hash-map iteration order.

Plain counting retains these suffix tables because top-down drawing uses them
to split a requested size among children without proposing infeasible
remainders.

## Shared root budgets

Production callers first construct one `RootBudgets` value for a canonical
current root and limit. It owns:

- the maximum useful subterm size for each relevant canonical current class;
- the minimum extraction size for every canonical current class; and
- the root limit.

Bundling these values prevents a global size limit from being passed where a
per-class limit is required. The same value is consumed by plain counting,
rooted match enumeration, the exact scan, and rooted joint counting.

Let `min(c)` be the smallest term extractable from `c`. The root starts with
the full limit:

```text
budget(root) = limit
```

For a parent node `f(c_1, ..., c_k)` in class `p`, child `c_i` can use whatever
remains when the node and every sibling take their minimum sizes:

```text
candidate_budget(c_i) = budget(p) - 1 - sum_(j != i) min(c_j)
```

A worklist relaxation retains the largest candidate received by each class.
Nodes whose children cannot fit at their minima do not propagate budgets.
Classes that never receive a budget cannot occur in any root extraction within
the limit and are omitted.

The minimum feasible size of an individual e-node is

```text
1 + sum_i min(c_i)
```

Rooted match enumeration uses this second check to skip oversized nodes even
inside a relevant class.

## Why capping is exact for root queries

Suppose a parent extraction fits within `budget(p)`. After paying for the node
and giving every sibling at least its minimum size, child `c_i` can receive no
more than `budget(c_i)` by construction. Consequently:

- every recursive plain-count lookup requested by the root is retained;
- every suffix-table remainder reachable during drawing is retained;
- every recursive draw call satisfies `size <= budget(child)`; and
- joint terms are also safe because a joint term for `(c, pc)` is a plain term
  of current class `c`.

The restriction removes only states no extraction from the selected root can
query. Direct histogram queries against deeper classes are intentionally
capped; `ExactCandidatePackage` exposes root-driven candidate drawing, not a global
all-class analysis.

## Rooted joint counting

Matches turn the generic kernel into a joint DP. A key `(c, pc)` has one DP
node for each match of an e-node in current class `c` against previous class
`pc`; its child keys are zipped pairs `(current_child, previous_child)`.

Only pairs whose current class appears in `RootBudgets` are created, and every
pair receives exactly `budget(c)`. A child pair absent from that map has no
terms, which is the standard `LayeredDp` contract. Previous classes are not
assigned an independent reachability or size budget.

See [finding the smallest novel sizes](novel_size_search.md) for the complete
exact-package flow.

## Verification

Tests cover:

- cyclic e-graphs with finite witnesses at increasing sizes;
- sibling minima tightening a deep child's budget;
- unreachable classes and oversized e-nodes being excluded;
- suffix tables checked entry-for-entry against direct convolutions;
- scan and final package histograms agreeing for every selected root in small
  acyclic and cyclic fixtures; and
- recursive frontier drawing returning only terms absent from the previous
  boundary.
