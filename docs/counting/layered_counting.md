# Size-layered term counting

How the per-class term-size histograms (`plain(c)(s)` = number of distinct
terms of size `s` extractable from e-class `c`) are computed by walking
upward through size layers, optionally restricted to what a root can reach.

The relevant code lives in:

- [src/sampling/count/layered.rs](../../src/sampling/count/layered.rs) — `count_terms`, `class_budgets`.
- [src/sampling/count/plain.rs](../../src/sampling/count/plain.rs) — `PlainTermCount::{new, rooted}`.
- [src/sampling/count/novel.rs](../../src/sampling/count/novel.rs) — the exact,
  root-restricted novel-size scan.
- [src/analysis/semilattice/ast_size.rs](../../src/analysis/semilattice/ast_size.rs) — the min-size analysis the budget pass reuses.

---

## 1. What is counted

For every e-class `c`, the result is a sparse histogram:

```
plain(c)(s) = number of distinct terms of exactly size s
              extractable from c
```

Term size is AST node count. An e-node contributes 1 for itself, so a node
`f(c_1, ..., c_k)` constructs a term of size `s` exactly when its children
construct subterms whose sizes sum to the remaining budget `s - 1`.

At each size layer, the question for every e-class is therefore:

> What can the e-nodes in this class construct from their children using
> exactly the remaining budget?

Leaves initialize the process: a leaf has no children, so it constructs one
term of size 1. Larger terms then become available layer by layer.

The resulting histograms are used to answer which root sizes exist and to
weight the choices made by the top-down samplers. Counting also produces the
per-node suffix tables those samplers need.

## 2. The layer-walking model

A term of size `s` rooted at an e-node spends 1 on the node itself, so its
child subterms have sizes `>= 1` summing to `s - 1` — every one of them is
**strictly smaller than `s`**:

```
plain(c, s) = sum over e-nodes n in c of
              |{ (s_1, ..., s_k) : s_i >= 1, sum = s - 1 }|   weighted by
              product over i of plain(child_i, s_i)           with all s_i < s
```

So `plain(·, s)` depends only on `plain(·, < s)`. The computation walks:

```
size 1  ->  size 2  ->  size 3  ->  ...  ->  limit
```

You can picture it as a table with one row per e-class and one column per
term size:

```
              size 1   size 2   size 3   ...   size limit
class A          ✓        ✓        ✓
class B          ✓        ✓        ✓
class C          ✓        ✓        ✓
```

The algorithm completes an entire column before moving to the next one.
When it is filling column `s`, every child count it can consult lies in a
column strictly to the left and is already final.

This remains true for cyclic e-graphs. For example:

```
X = { a, f(X) }
```

The e-class points to itself, but one size layer lower:

```
(X, size s) depends on (X, size s - 1)
```

The layers therefore unfold the cycle into an ordinary progression:

```
(X, 1)        (X, 2)        (X, 3)        ...
   a            f(a)          f(f(a))
```

The cycle is still present between e-classes, but there is no cycle between
the expanded `(e-class, size)` states. Leaves seed size 1, and each completed
layer makes the next layer computable.

## 3. The layered DP

`count_terms` keeps, per `(class, node)` with children `c_1..c_k`, the suffix
tables

```
S_i(t) = number of ways to fill children i..k with subterm sizes summing to t
S_k    = {0: 1}    (empty product)
```

and processes `size = 1..=limit`. Each layer does two things:

**Extend the suffix tables by the single new total `t = size - 1`** (from
`i = k-1` down to `0`):

```
S_i(t) = sum over sigma of  plain(c_i, sigma) * S_{i+1}(t - sigma)
```

Both factors are already final: histogram entries are at sizes `<= size - 1 = t`
(earlier layers), and since `sigma >= 1`, the lookup `S_{i+1}(t - sigma)`
only touches totals `<= t - 1` from earlier layers — the `t`-entry that
`S_{i+1}` gained moments ago in this same loop is provably never read.
The two-map sum iterates whichever of the two maps is smaller
(`convolve_entry`).

**Read off the layer**: `plain(c, size) = sum over nodes of S_0(size - 1)`
(for leaves `S_0 = {0: 1}`, contributing exactly at `size = 1`).

The two phases of a layer are deliberately separate: every suffix entry for
total `s - 1` is computed before any class publishes its count at size `s`.
The result therefore cannot depend on hash-map iteration order.

The suffix tables are also the cache used by the sampler. To choose child
`i`, the sampler combines the number of terms available from that child with
`S_{i+1}`, the number of ways the remaining children can consume the
remaining budget. The DP therefore returns the histograms and suffix tables
together in `CountData`.

## 4. Root restriction: budgets

The histograms are only ever consumed from very few classes — the roots —
plus whatever top-down sampling reaches from them, and sampling can never ask
a deep class for a large size: the size budget shrinks on the way down.
`PlainTermCount::rooted` exploits that.

With `min(c)` = smallest extractable term size (the existing `AstSize`
semilattice analysis — a cheap fixpoint over small integers), define the
**budget** as the largest size a subterm rooted at `c` can take in *any*
extraction of size `<= limit` from a root:

```
budget(root) = limit
budget(c_i)  = max over parent positions (node n in class p, slot i) of
               budget(p) - 1 - sum over j != i of min(c_j)
```

computed by worklist relaxation. Budgets only grow, are bounded by `limit`,
and strictly shrink along any cycle (each level costs the node plus the
siblings' minima), so the relaxation terminates quickly. The DP then simply
skips layers above a class's budget; classes that never receive a budget
(unreachable, or unusable within `limit`) are skipped entirely. A deep class
ends up with a handful of histogram entries instead of ~`limit`, which
shrinks every convolution that touches it — including in the joint/novel
counting built on top.

**Why capped data is still exact for every root-driven query.** Capping never
alters a kept count, it only drops sizes a root extraction cannot use:

- In any decomposition of a total `t <= budget(p) - 1` across *all* children
  of a node, child `i` gets at most `t - sum(min of siblings)
  <= budget(c_i)` — so `S_0`, the table both the layer read-off and the
  sampler's initial node pick consult, never misses a contribution.
- Mid-recursion the sampler looks up `S_{i+1}(r)` only at remainders `r`
  left after children `0..i` drew real sizes (each `>= min`), and the same
  bound applies to every child in the suffix. Inner tables can undercount
  only at totals no reachable sampling state can produce, and the DP itself
  reads inner tables only at reachable totals (same induction).
- Recursive `sample(child, s)` calls therefore always satisfy
  `s <= budget(child)`, where the child's histogram is complete. The min
  sizes are plain minima, which lower-bound novel/joint terms too, so the
  argument carries over to the novel sampler sitting on top of a rooted
  plain count.

Everything answered *from a root* — histograms, `possible_size`, sampling,
enumeration — is bit-identical to the unrooted analysis (tested). Direct
queries against deeper classes see the capped data; that is the documented
contract of `rooted`.

## 5. Reuse for joint and novel counting

`LayeredDp` is generic over its key type. Plain counting uses an e-class ID
as the key. Joint counting uses `(current_class, previous_class)` pairs, but
walks the same size layers with the same recurrence. The exact novel-size
scan advances the plain and joint DPs together and can inspect a root as soon
as each layer becomes final; see
[smallest novel-size search](novel_size_search.md).

## 6. Verification

- Cycle tests cover `union(a, a+b)` graphs with exact counts at size 5
  (1 and 16 respectively).
- `rooted` ≡ unrooted at the root — histogram *and* suffix tables — on a
  cyclic graph.
- Budget caps: a cyclic class under a unary root loses exactly its
  unreachable top size; sibling minima tighten budgets by the expected
  amount (`+(x, f(f(b)))` caps `x` at `limit - 4`); unreachable classes are
  absent.
- Layered suffix tables compared entry-for-entry against
  `suffix_convolutions` over full histograms.
