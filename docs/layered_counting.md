# Size-layered term counting

How the per-class term-size histograms (`plain(c)(s)` = number of distinct
terms of size `s` extractable from e-class `c`) are computed — one pass per
size layer instead of a worklist fixpoint, optionally restricted to what a
root can reach.

The relevant code lives in:

- [src/sampling/count/layered.rs](../src/sampling/count/layered.rs) — `count_terms`, `class_budgets`.
- [src/sampling/count/plain.rs](../src/sampling/count/plain.rs) — `PlainTermCount::{new, rooted}`.
- [src/sampling/count/novel.rs](../src/sampling/count/novel.rs) — the root-restricted `Mod61` probe.
- [src/analysis/semilattice/ast_size.rs](../src/analysis/semilattice/ast_size.rs) — the min-size analysis the budget pass reuses.

---

## 1. The problem

Three consumers need `plain`:

1. the root's histogram (which sizes exist, `min_size`/`max_size`, sample
   allocation per size),
2. the top-down samplers, which weigh child-size splits by counts,
3. the per-node **suffix convolution cache** the plain sampler reads
   (`suffix[class][node][i]` = ways children `i..` sum to a given total).

The old implementation (`ExprCount`, an implementor of the now-removed
`CommutativeSemigroupAnalysis` trait) treated the histogram map as a general
lattice value and iterated a worklist to a fixpoint:

- pop a class, run `make` for every e-node — a **full convolution of all
  child histograms up to `limit`**, with `BigUint` cells;
- if the merged result differs from the stored one, re-enqueue all parents.

On a cyclic e-graph this is where the money went. A class on a cycle gains
roughly **one new size per trip around the cycle**, and every gain re-runs
the full convolution for every node of every parent on the cycle. That is up
to `limit` full recomputations per class — ~cubic in `limit` per node where
one convolution (~quadratic) suffices. On top of that, the fixpoint computed
every class in the graph (reachable from a root or not), cloned all child
histograms per `make`, compared whole `BigUint` histograms for the
convergence check, and `PlainTermCount::new` then rebuilt the suffix cache
from scratch in a second pass over every node.

## 2. The key observation: counting is stratified by size

A term of size `s` rooted at an e-node spends 1 on the node itself, so its
child subterms have sizes `>= 1` summing to `s - 1` — every one of them is
**strictly smaller than `s`**:

```
plain(c, s) = sum over e-nodes n in c of
              |{ (s_1, ..., s_k) : s_i >= 1, sum = s - 1 }|   weighted by
              product over i of plain(child_i, s_i)           with all s_i < s
```

So `plain(·, s)` depends only on `plain(·, < s)`. Indexed by `(class, size)`,
the dependency relation is a DAG with `limit` layers **even when the e-graph
is cyclic** — the cycles the old fixpoint fought are an artifact of computing
whole histograms at once. Induction over `s` is simultaneously the
termination argument and the correctness proof; no convergence check exists
because there is nothing to converge.

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

Two structural wins fall out:

- **Total work per e-node = one full convolution.** Each `(i, t)` cell is
  computed exactly once; summed over layers that is the same arithmetic a
  single `suffix_convolutions` call performs — the old version paid it up to
  `limit` times.
- **The DP state *is* the suffix cache.** The tables the sampler wants are
  exactly the `S_i`, so the separate cache-building pass in
  `PlainTermCount::new` is gone; the DP returns histograms and cache
  together (`CountData`).

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

## 5. Old vs. new

|                                | old (`ExprCount` fixpoint)               | new (`count_terms`)                          |
| ------------------------------ | ---------------------------------------- | -------------------------------------------- |
| cycles                         | re-convolve until fixpoint               | non-issue: DP is stratified by size          |
| convolutions per e-node        | up to `limit` full ones                  | exactly one (spread over layers)             |
| classes computed               | all                                      | all, or root-reachable within budget         |
| per-class size cap             | global `limit`                           | per-class budget (`rooted`)                  |
| suffix cache                   | second full pass (rayon)                 | byproduct of the DP                          |
| convergence check              | whole-histogram `BigUint` equality       | none                                         |
| empty histograms               | stored as empty maps                     | absent from `data`                           |

Measured on a depth-30 chain of self-cyclic classes at `limit = 150`
(release): `PlainTermCount` construction went from **214 ms to 3.9 ms**
unrooted and **2.6 ms** rooted, with identical histograms. The gap widens
with `limit` (cubic vs. quadratic); the rooted gain grows with how much of
the graph a root cannot reach.

## 6. Observable behavior changes

- `CommutativeSemigroupAnalysis` and `ExprCount` are gone; `ExprCount`'s
  cycle tests moved to `layered.rs`.
- `PlainTermCount::data()` no longer contains classes whose histogram is
  empty (`enumerate` handles the absence; `size_histogram` was `Option`
  already). The suffix cache is likewise trimmed to classes with terms.
- `PrecomputePackage` builds its plain count with
  `PlainTermCount::rooted(max_size, curr, &[root])`, and
  `probe_novel_root_sizes` runs its `Mod61` plain pass root-restricted as
  well — the probe inside `backoff_precompute` gets the same speedup.
  (`compute_joint` has since been stratified by size too, on the same
  generic `LayeredDp` kernel; see
  [incremental_probe.md](incremental_probe.md).)
- For `Mod61`, entries whose *residue* is zero are now dropped eagerly
  instead of at the end. Downstream products with them were `0 mod p`
  anyway, so residues are unchanged; a dropped size is exactly the
  documented one-sided fingerprint miss.
- `Sampler::min_size` returned the smallest size with *more than one* term
  (`possible_size(id, size, 1)`) and spun forever when no size had two; it
  now returns the smallest size with any term, matching its docstring and
  `smallest()`.
- `PlainTermCount::new` lost its `Sync` bounds (the DP is single-threaded;
  the rayon suffix pass no longer exists).

## 7. Verification

- Ported cycle tests: the `union(a, a+b)` graphs with exact counts at size 5
  (1 and 16 respectively).
- `rooted` ≡ unrooted at the root — histogram *and* suffix tables — on a
  cyclic graph.
- Budget caps: a cyclic class under a unary root loses exactly its
  unreachable top size; sibling minima tighten budgets by the expected
  amount (`+(x, f(f(b)))` caps `x` at `limit - 4`); unreachable classes are
  absent.
- Layered suffix tables compared entry-for-entry against
  `suffix_convolutions` over full histograms.
- Full suite: 146/146 with `cargo nextest`; `cargo clippy --all-targets`
  clean.
