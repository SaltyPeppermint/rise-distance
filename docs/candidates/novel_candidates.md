# Exact novel-candidate construction

This document is the high-level guide to how the codebase constructs candidate
terms from a current e-graph (`curr`) that are **not** extractable from any
e-class in an earlier snapshot (`prev`)—terms that reflect new equalities
learned between `prev` and `curr`.

Instead of sampling terms at random and rejecting ones that already existed in
`prev`, the pipeline has two stages:

1. **Count** the exact number of novel terms per size within the root's size
   budget (`FrontierPackage::build`).
2. **Draw** directly from the novel subset without replacement
   (`FrontierDrawer`).

> **How the docs fit together:**
> - **This doc:** core definitions, end-to-end pipeline, correctness proof,
>   worked example, and public API.
> - **Drawing & policies:** [frontier_drawing.md](frontier_drawing.md) —
>   the two-state frontier automaton, agreement profiles, `Weigher` policies,
>   and `SampledTrie`.
> - **Counting kernel:** [layered_counting.md](../counting/layered_counting.md) —
>   `LayerTable` (handling cycles) and `MinSizes` / `RootBudgets`.
> - **Search & package build:** [novel_size_search.md](../counting/novel_size_search.md) —
>   geometric cap doubling, match enumeration, incremental scanning, and pruning.
> - **Seed terms (no e-graph):** [random_terms.md](../generation/random_terms.md).

## Where the code lives

- [src/sampling/count/budgets.rs](../../src/sampling/count/budgets.rs) —
  `MinSizes`, `RootBudgets`, and `doubling_caps`.
- [src/sampling/count/mod.rs](../../src/sampling/count/mod.rs) —
  generic `LayerTable` size-layered DP kernel.
- [src/sampling/count/novel.rs](../../src/sampling/count/novel.rs) —
  `NovelTermCount`, match enumeration, joint counting, and novel derivation.
- [src/sampling/draw/frontier.rs](../../src/sampling/draw/frontier.rs) —
  `FrontierPackage` and `FrontierDrawer`.
- [src/sampling/draw/mod.rs](../../src/sampling/draw/mod.rs) —
  shared `Drawer` trait, `SamplePackage`, and `draw_children`.

---

## 1. Core definitions

Let `curr` and `prev` be two rebuilt e-graphs, and let size `s` be AST node
count. In any rebuilt e-graph, a concrete term `t` (a `RecExpr`) belongs to
**at most one** e-class: `prev.lookup(t)` returns `Some(pc)` for at most one
canonical `prev` class `pc`, or `None` if `t` cannot be extracted from `prev`.

For a term `t` extracted from canonical `curr` class `c`, we call `t`:

- **non-novel** if `prev.lookup(t) = Some(pc)` for some `prev` class `pc`;
- **novel** if `prev.lookup(t) = None`.

Because two `prev` classes can never share the same concrete term, partitioning
non-novel terms by their `prev` class `pc` **never double-counts**.

This defines the three count tables used across the counting and drawing docs:

```text
whole(c, s)     = |{ terms t of exact size s extractable from curr class c }|

joint(c, pc, s) = |{ terms t of exact size s extractable from curr class c
                     such that prev.lookup(t) = Some(pc) }|

non_novel(c, s) = sum over pc of joint(c, pc, s)
novel(c, s)     = whole(c, s) - non_novel(c, s)
```

`novel[root]` is what `FrontierPackage::root_histogram` exposes and what the
drawer uses to allocate draws across sizes.

---

## 2. Pipeline overview

### Stage 1: Build the count package (`FrontierPackage::build`)

Rather than counting the entire e-graph up to a large global bound, building a
package scopes all work to `root` and finds the smallest size limit that
provides at least `min_extractable` novel terms:

1. **Precompute budgets:** compute `MinSizes` once, then grow size caps
   geometrically from `min(root) + search_step` up to
   `cap = start_size + search_steps`, deriving per-class `RootBudgets` for each
   attempt ([layered_counting.md](../counting/layered_counting.md)).
2. **Enumerate matches:** find every budgeted `curr` e-node whose operator and
   child classes match an e-node in `prev` (`NodeMatch`), building each class's
   `cover[c]` of shared `prev` classes, then drop `prev`'s index before the DP
   tables allocate ([novel_size_search.md](../counting/novel_size_search.md)).
3. **Scan layer by layer:** step `whole` and `joint` `LayerTable`s together
   size by size (`s = 1, 2, ...`) until the cumulative novel count at `root`
   reaches `min_extractable`.
4. **Prune and assemble:** tighten `matches`, `whole`, and `joint` in place to
   the winning size's `final_budgets`, derive `cover` from nonempty `joint`
   keys, and subtract `joint` from `whole` to obtain `novel`
   ([novel_size_search.md](../counting/novel_size_search.md)).

### Stage 2: Draw novel terms (`FrontierDrawer`)

Given the assembled `NovelTermCount`, `FrontierDrawer` constructs terms
top-down using a two-state tree automaton:

- **`State::Novel`** — the subtree must not exist in any `prev` class (always
  the starting state at `root`).
- **`State::SharedWith(pc)`** — the subtree must extract into `prev` class `pc`.

At each step, `FrontierDrawer` enumerates only feasible branches and child-size
splits (rejecting any `Novel` child profile that completes a `NodeMatch`),
`Weigher` selects among them according to `Policy` (`Count`, `Uniform`, or
`Smallest`), and `SampledTrie` tracks prior draws for zero-retry
without-replacement sampling. See [frontier_drawing.md](frontier_drawing.md) for
the full automaton rules, branch pseudocode, and policy details.

---

## 3. Correctness

**Claim.** Whenever `novel(root, s) > 0` and fewer than `novel(root, s)` terms
have already been sampled, `draw_next(s, sampled, rng)` returns a size-`s` term
`t` with `prev.lookup(t) = None`.

We prove two properties jointly by induction on `s`:

- **(A) `SharedWith(pc)`:** If `joint(c, pc, s) > 0`, constructing at
  `(c, s, SharedWith(pc))` returns a size-`s` term `t` with
  `prev.lookup(t) = Some(pc)`.
- **(N) `Novel`:** If `novel(c, s) > 0`, constructing at `(c, s, Novel)`
  returns a size-`s` term `t` with `prev.lookup(t) = None`.

### Base case (`s = 1`, leaves)

- **(A):** `shared_branches` picks a leaf `n` with a match `m.prev_class == pc`.
  By `enumerate_matches`, such a match exists iff `prev.lookup(n) = Some(pc)`. ✓
- **(N):** `novel_branches` picks a leaf `n` whose empty profile `[]` completes
  no match—meaning `prev.lookup(n) = None`. ✓

### Inductive step (`s > 1`)

- **(A):** `shared_branches` picks `(n, m)` with `m.prev_class == pc` and child
  sizes `s_i < s` summing to `s - 1` with
  `joint(c_i, m.prev_children[i], s_i) > 0`. By IH (A), each child returns
  `t_i` with `prev.lookup(t_i) = Some(m.prev_children[i])`. Looking up
  `n(t_1, ..., t_k)` in `prev` finds the matched node in `pc`. ✓

- **(N):** Suppose for contradiction that `construct` returns
  `t = n(t_1, ..., t_k)` with `prev.lookup(t) = Some(P)`. Then `prev` contains
  an e-node `n'` in `P` with canonical children `[q_1, ..., q_k]` where
  `prev.lookup(t_i) = Some(q_i)` for every `i`.
  - If the chosen profile had `profile[i] == Novel`, then IH (N) gives
    `prev.lookup(t_i) = None`, contradicting `Some(q_i)`.
  - Therefore every slot must be `profile[i] == SharedWith(pc_i)`, and IH (A)
    forces `pc_i = q_i`.
  - Since `t` and all `t_i` fit the root budgets, `enumerate_matches` is
    complete ([why rooted enumeration is complete](../counting/novel_size_search.md#why-rooted-enumeration-is-complete))
    and recorded `{ prev_class: P, prev_children: [q_1, ..., q_k] }`.
  - But then `completes_some_match` would have rejected this profile.
    Contradiction. ✓

### No double-counting

Because `prev.lookup(t_i)` is unique for every child `t_i`, a non-novel term
`n(t_1, ..., t_k)` matches **at most one** `NodeMatch` (distinct matches of `n`
must differ in at least one `prev_children[i]`). Thus
`non_novel(c, s) = sum_pc joint(c, pc, s)` counts every non-novel term exactly
once, and the `Novel` agreement profiles partition the remaining novel terms
with zero overlap.

### Termination

- **Match enumeration:** monotonically adds entries to a finite set of
  `(curr_node, prev_class, prev_children)` tuples, so the fixpoint terminates.
- **Counting:** `LayerTable` runs one pass per size `1..=try_cap` (no fixpoint,
  even on cyclic e-graphs).
- **Drawing:** every recursive call strictly decreases `s` (`s_i <= s - 1`).

---

## 4. Worked example: `Add(a, b)` after `union(a, b)`

Suppose `prev` contains `a`, `b`, and `Add(a, b)`, and `curr` merges `a` and
`b`:

```text
prev classes:  A = {a},  B = {b},  R_prev = {Add(A, B)}
curr classes:  M = {a, b},         R      = {Add(M, M)}
```

1. **Match enumeration:**
   - `a` in `M` → match `{ prev_class: A, prev_children: [] }` (`cover[M]` gets `A`).
   - `b` in `M` → match `{ prev_class: B, prev_children: [] }` (`cover[M]` gets `B`).
   - `Add(M, M)` in `R` tries all pairs in `{A, B} × {A, B}`: only `(A, B)`
     exists in `prev`, giving match
     `{ prev_class: R_prev, prev_children: [A, B] }`.

2. **Joint counts:**
   - `joint(M, A, 1) = 1` (`a`)
   - `joint(M, B, 1) = 1` (`b`)
   - `joint(R, R_prev, 3) = joint(M, A, 1) * joint(M, B, 1) = 1` (`Add(a, b)`)

3. **Novel histogram:**
   - `M` at size 1: `whole(M, 1) = 2`, `non_novel(M, 1) = 1 + 1 = 2` → `novel[M]` is empty.
   - `R` at size 3: `whole(R, 3) = 4`, `non_novel(R, 3) = 1` → `novel(R, 3) = 3`.

4. **Drawing at `(R, s = 3, Novel)`:**
   Each child of `Add(M, M)` has options `[Novel, SharedWith(A), SharedWith(B)]`
   (9 profiles total):

   | Profile | Status | Count at budget 2 | Drawn term |
   | :--- | :--- | :---: | :--- |
   | `(SharedWith(A), SharedWith(B))` | Rejected (`completes_some_match`) | — | — |
   | Any profile with `Novel` | Kept, but `novel[M]` is empty | 0 | — |
   | `(SharedWith(A), SharedWith(A))` | Kept | 1 | `Add(a, a)` |
   | `(SharedWith(B), SharedWith(A))` | Kept | 1 | `Add(b, a)` |
   | `(SharedWith(B), SharedWith(B))` | Kept | 1 | `Add(b, b)` |

   Under `Policy::Count`, the drawer picks uniformly among the three novel
   terms `Add(a, a)`, `Add(b, a)`, and `Add(b, b)`, and never produces
   `Add(a, b)`. This is verified by
   `sampling::draw::frontier::tests::independent_frontier_draw_union_diagonal`.

---

## 5. API summary

```rust
// Grow budgets geometrically from `min(root) + search_step` up to
// `start_size + search_steps` until at least `min_extractable` novel terms
// are available.
let (max_size, package) = FrontierPackage::build(
    result,
    start_size,
    search_step,
    search_steps,
    min_extractable,
)?;

// `SamplePackage` trait: allocate `count` draws across root sizes, then draw.
let terms = package.draw_samples(count, policy, seed)?;
```

`FrontierPackage` ([src/sampling/draw/frontier.rs](../../src/sampling/draw/frontier.rs))
owns the full build lifecycle: computing `MinSizes` once, running the geometric
cap search with non-overlapping `prev` lookup and DP memory, pruning to the
final size, and dropping the previous boundary before returning.
