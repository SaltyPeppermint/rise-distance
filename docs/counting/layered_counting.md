# Size-layered, root-restricted term counting

Candidate construction counts exact-size terms only for current e-classes and
sizes that can participate in an extraction from the selected root. This
document describes the shared size-layered kernel and the root budgets that
bound every production counting phase.

The relevant code is:

- [src/sampling/count/budgets.rs](../../src/sampling/count/budgets.rs) —
  `RootBudgets` and `MinSizes`.
- [src/sampling/count/mod.rs](../../src/sampling/count/mod.rs) —
  `LayerTable` and per-node size-layered accumulation.
- [src/sampling/count/whole.rs](../../src/sampling/count/whole.rs) —
  whole-graph layer stepping directly over the e-graph.
- [src/sampling/draw/whole.rs](../../src/sampling/draw/whole.rs) —
  top-down consumption of rooted whole histograms.
- [src/sampling/count/novel.rs](../../src/sampling/count/novel.rs) — reuse of
  `LayerTable` for rooted joint counting and the novel-size scan.
- [src/analysis/semilattice/ast_size.rs](../../src/analysis/semilattice/ast_size.rs)
  — minimum extraction sizes used by the budget calculation.

## Exact-size recurrence

For a current e-class `c`, the sparse histogram

```text
whole(c, s) = number of terms of exactly size s extractable from c
```

uses AST node count as size. For an e-node `f(c_1, ..., c_k)`, the node costs
one and its children share the remaining `s - 1` nodes:

```text
whole(c, s) = sum over nodes f(c_1, ..., c_k) in c
              sum over s_1 + ... + s_k = s - 1
              product_i whole(c_i, s_i)
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

## The `LayerTable` kernel

`LayerTable<K, N>` is generic over a state key `K` and suffix-cache node key
`N`. Whole counting uses `K = Id` and steps directly over `egraph[id].nodes`;
joint current/previous counting uses `K = (Id, PrevId)` and steps directly over
`matches`.

Nodes of arity ≤ 2 are evaluated directly without any suffix table. Only for
nodes with arity `n >= 3` does `LayerTable` lazily populate sparse suffix
convolution tables for positions `1..n - 1`:

```text
S_i(t) = ways to fill children i..n with total size t
```

At size layer `s`, it extends each stored suffix by the single total `t = s - 1`:

```text
S_i(t) = sum over sigma of histogram(k_i, sigma) * S_(i+1)(t - sigma)
```

Both factors are final. `sigma >= 1`, so even the suffix lookup is at a total
strictly below the entry being added. After all nodes in a layer have been
evaluated, the layer's counts are published into `data`. Separating evaluation
from publication makes the result independent of hash-map iteration order.

Any suffix cache is dropped once counting finishes; top-down drawing computes
on-demand suffix convolutions only for nodes with arity ≥ 3.

## Shared root budgets

Production callers first construct one `RootBudgets` value for a canonical
current root and limit. It owns:

- the maximum useful subterm size for each relevant canonical current class;
- the minimum extraction size for every canonical current class; and
- the root limit.

Bundling these values prevents a global size limit from being passed where a
per-class limit is required. The same value is consumed by whole counting,
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

Whole counting and rooted match enumeration use this second check to skip
oversized nodes even inside a relevant class.

## Why capping is exact for root queries

Suppose a parent extraction fits within `budget(p)`. After paying for the node
and giving every sibling at least its minimum size, child `c_i` can receive no
more than `budget(c_i)` by construction. Consequently:

- every recursive whole-count lookup requested by the root is retained;
- every suffix-convolution remainder reachable during drawing is covered by the
  retained child histograms;
- every recursive draw call satisfies `size <= budget(child)`; and
- joint terms are also safe because a joint term for `(c, pc)` is a whole term
  of current class `c`.

The restriction removes only states no extraction from the selected root can
query. Direct histogram queries against deeper classes are intentionally
capped; `FrontierPackage` exposes root-driven candidate drawing, not a global
all-class analysis.

## Rooted joint counting

Joint counting steps directly over the rooted `matches` table using
`LayerTable<(Id, PrevId), (Id, usize, usize)>`. For each matched node in current
class `c` within `budget(c)`, each match `m` contributes to `(c, m.prev_class)`
with child keys zipped from `(current_child, previous_child)`. Previous classes
are not assigned an independent reachability or size budget; missing child
pairs contribute no terms.

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
