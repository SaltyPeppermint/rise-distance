# Size-layered, root-restricted term counting

To sample terms from an e-graph without bias or wasted work, we first count
how many terms of each exact AST size `s` can be extracted from a chosen root
e-class.

Doing this naively runs into two problems:

1. **Cycles:** E-graphs often contain cycles (like `X = {a, f(X)}`), where
   unbounded extraction is infinite.
2. **Scale:** An e-graph can contain many e-classes and large subterms that can
   never appear in a size-bounded extraction from `root`.

The counting layer solves both with two shared primitives:

- **Size-layered dynamic programming (`LayerTable`)** counts terms size by size
  (`s = 1, 2, 3, ...`), unfolding cycles into an acyclic DAG.
- **Root budgets (`MinSizes` and `RootBudgets`)** compute the largest subterm
  size each e-class could possibly need for `root`, skipping every unreachable
  class and oversized e-node.

> **Related docs:** For how these primitives are combined with match
> enumeration and geometric cap doubling to build a `FrontierPackage`, read
> [novel_size_search.md](novel_size_search.md) next. For the high-level
> novel-candidate pipeline, see
> [novel_candidates.md](../candidates/novel_candidates.md).

## Where the code lives

- [src/sampling/count/budgets.rs](../../src/sampling/count/budgets.rs) —
  `MinSizes`, `RootBudgets`, and `doubling_caps`.
- [src/sampling/count/mod.rs](../../src/sampling/count/mod.rs) —
  generic `LayerTable` kernel and arity-specialized convolution.
- [src/sampling/count/whole.rs](../../src/sampling/count/whole.rs) —
  whole-graph counting directly over the e-graph.
- [src/sampling/count/novel.rs](../../src/sampling/count/novel.rs) —
  reuse of `LayerTable` for joint `curr`/`prev` counting and novel-size search.
- [src/sampling/draw/whole.rs](../../src/sampling/draw/whole.rs) —
  top-down drawing from whole-graph histograms.
- [src/analysis/semilattice/ast_size.rs](../../src/analysis/semilattice/ast_size.rs) —
  minimum extraction sizes (`AstSize`) used to compute root budgets.

---

## 1. Counting by size layer (handling cycles)

Size `s` is AST node count. For a canonical e-class `c`, we want the sparse
histogram:

```text
whole(c, s) = |{ terms t of exact size s extractable from c }|
```

An e-node `f(c_1, ..., c_k)` contributes `1` for itself, leaving `s - 1` nodes
to distribute among its `k` children:

```text
whole(c, s) = sum over nodes f(c_1, ..., c_k) in c
              sum over s_1 + ... + s_k = s - 1
              product_i whole(c_i, s_i)
```

Because every child subterm has size at least `1`, every `s_i` is strictly less
than `s`. Layer `s` therefore depends **only on already-finished layers
`1..s - 1`**.

Even when e-classes form a cycle, the expanded `(e-class, s)` states are
strictly acyclic. For example, `X = {a, f(X)}` unfolds cleanly one size at a
time:

```text
s = 1: a          (count = 1)
s = 2: f(a)       (count = 1)
s = 3: f(f(a))    (count = 1)
...
```

No fixpoint iteration is needed: each `(c, s)` entry is computed once.

---

## 2. The `LayerTable` kernel

`LayerTable<K, N>` is a generic size-layered counter parameterized by:

- **`K` (state key):** what is being counted.
- **`N` (node key):** identifies a specific production for optional suffix
  caching.

| Counting mode | State key `K` | Node key `N` | Steps over |
| :--- | :--- | :--- | :--- |
| Whole counting | `Id` (canonical `curr` class `c`) | `(Id, node_idx)` | `egraph[c].nodes` |
| Joint counting | `(Id, PrevId)` (pair `(c, pc)`) | `(Id, node_idx, match_idx)` | rooted `matches` |

### Fast paths by arity

When computing how many ways a node's children can sum to `total = s - 1`,
`LayerTable::node_count` avoids unnecessary allocations by specializing on
arity:

- **Arity 0 (leaf):** returns `1` when `total == 0` (`s == 1`), else `None`.
- **Arity 1 (unary):** direct histogram lookup at `total`.
- **Arity 2 (binary):** single pointwise convolution of the two child
  histograms at `total`.
- **Arity ≥ 3:** lazily populates sparse **suffix convolution tables** for
  positions `1..n - 1`:

  ```text
  S_i(t) = ways to fill children i..n with total size t
  ```

  At each size layer `s`, `node_count` extends each stored suffix table by just
  the single new total `t = s - 1`:

  ```text
  S_i(t) = sum over sigma of histogram(k_i, sigma) * S_(i+1)(t - sigma)
  ```

  Because every child size `sigma >= 1`, both factors are already finalized.
  This requires callers to step `total` in ascending order without gaps for
  each active node key. Once counting finishes, `into_data()` drops the entire
  suffix cache; top-down drawing computes suffix convolutions on demand only
  for the arity ≥ 3 nodes it actually visits.

### Order-independent layer updates

Within layer `s`, all budgeted states are evaluated against the finalized
`1..s - 1` tables first, and only then are the new counts inserted into `data`.
Separating evaluation from publication ensures the counts never depend on
hash-map iteration order.

---

## 3. Root budgets (`MinSizes` and `RootBudgets`)

Counting every class up to a global size limit wastes time and memory on
subterms that can never fit inside a root extraction. Instead, counting is
scoped to one canonical `root` and size `limit`.

### Step 1: Precompute minimum sizes (`MinSizes`)

`MinSizes::new(egraph)` runs a single bottom-up `AstSize` analysis to find
`min(c)`, the size of the smallest term extractable from each canonical class
`c`. Because `MinSizes` depends only on the e-graph (not on `limit`), callers
compute it once and reuse it across multiple limits (for example, during
geometric cap search and final pruning).

### Step 2: Propagate maximum useful sizes (`RootBudgets`)

`min_sizes.budgets_for(egraph, root, limit)` computes `budget(c)`, the largest
subterm size class `c` can ever take inside a size-`limit` extraction from
`root`:

1. Seed `root` with the full limit:
   ```text
   budget(root) = limit
   ```
2. For any parent e-node `f(c_1, ..., c_k)` in class `p`, child `c_i` can use
   whatever remains after paying `1` for `f` and giving every sibling its
   minimum possible size:
   ```text
   candidate_budget(c_i) = budget(p) - 1 - sum_(j != i) min(c_j)
   ```
3. Run a worklist relaxation that keeps the **largest** candidate budget any
   parent path gives to `c_i`.

**Example:** Suppose `root` has `budget = 6` and contains `f(A, B)` where
`min(A) = 1` and `min(B) = 3`.
- For `A`: `6 - 1 - 3 = 2` (so `A` never needs terms larger than size 2).
- For `B`: `6 - 1 - 1 = 4` (so `B` never needs terms larger than size 4).

Two additional filters tighten the work further:

- **Unreachable / over-budget classes** (those that never receive a budget
  `>= min(c)`) are omitted from `RootBudgets` entirely.
- **Oversized e-nodes (`node_fits`)** inside an otherwise relevant class are
  skipped whenever their cheapest possible realization exceeds the class budget:
  ```text
  1 + sum_i min(c_i) > budget(c)
  ```

---

## 4. Why capping is exact for root queries

Suppose a parent extraction from `p` has size `<= budget(p)`. After paying `1`
for the parent node and at least `min(c_j)` for every sibling `j != i`, child
`c_i` cannot possibly exceed `budget(c_i)` by construction.

Consequently, restricting counting and matching to `RootBudgets` drops **only**
states that no extraction from `root` within `limit` can ever query:

- Every whole-count lookup needed by `root` is present.
- Every suffix-convolution remainder needed during top-down drawing is covered.
- Every recursive draw call satisfies `s <= budget(child)`.
- Every joint `(c, pc)` term is also a whole term of `c`, so `budget(c)` is a
  sound bound for joint counting too.

> **Note:** Non-root classes are intentionally capped to what `root` can use.
> `WholePackage` and `FrontierPackage` are built for root-driven drawing, not
> as global all-class tables.

---

## 5. Next: driving whole and joint counts together

Joint counting reuses `LayerTable` with key `(Id, PrevId)` over the rooted
`matches` table, giving each pair `(c, pc)` the budget `budget(c)` of its
current class. See [novel_size_search.md](novel_size_search.md) for how
`FrontierPackage::build` combines `MinSizes`, `RootBudgets`, match enumeration,
and simultaneous whole/joint layer stepping to find the smallest usable novel
size.

---

## Verification

Tests cover:

- cyclic e-graphs with finite witnesses at increasing sizes;
- sibling minima tightening a deep child's budget;
- unreachable classes and oversized e-nodes being excluded;
- arity ≥ 3 suffix tables matching direct convolutions entry-for-entry;
- scan and final package histograms agreeing across acyclic and cyclic
  fixtures; and
- recursive frontier drawing returning only terms absent from the previous
  boundary.
