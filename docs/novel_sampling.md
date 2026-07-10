# Novel sampling

End-to-end walkthrough of how the codebase samples terms from `curr` that are *not* extractable from any e-class in `prev` — i.e., terms that carry information learned in the iteration that produced `curr` from `prev`.

The relevant code lives in:

- [src/count/novel.rs](src/count/novel.rs) — counting & match enumeration.
- [src/sampling/novel.rs](src/sampling/novel.rs) — the sampler.

---

## 1. Definitions

Let `curr` and `prev` be two e-graphs (`prev` is some earlier snapshot of `curr`). Every concrete term `t` (a `RecExpr`) has a unique e-class in any rebuilt e-graph it exists in: `prev.lookup(t)` returns `Some(pc)` for at most one `pc`, and similarly for `curr`.

We call a term `t` extracted from curr's class `c`:

- **non-novel** if `prev.lookup(t)` is `Some(_)`, i.e. `t` is also some extraction of some prev class,
- **novel** otherwise.

Two prev classes cannot share a term (`prev.lookup` is unique), so when we count non-novel terms by partitioning over prev classes, there is no double-counting.

The core data structure is the **joint extractability table**:

```
joint(c, pc)(s) =
    | { extractions t of size s rooted at curr's c
        such that prev.lookup(t) = Some(pc) } |
```

From `joint` and the standard plain count `plain(c)(s)` we derive the **novel histogram**:

```
non_novel(c, s) = sum over pc of joint(c, pc)(s)
novel(c, s)     = plain(c, s) - non_novel(c, s)
```

`novel(c, s)` is what `possible_size` consults and what the sampler weighs.

---

## 2. Building `joint` — three phases

### Phase 1: Match enumeration (no counts yet)

A **match** of a curr e-node `n` (in curr-class `c`) is a prev e-node `n'` (in prev-class `pc`) with the same operator/arity such that `n'`'s child classes can be picked consistently from each curr-child's `cover` set.

Concretely:

```
NodeMatch {
    prev_class:    pc,           // the prev e-class containing n'
    prev_children: [pc_1, ..., pc_k]   // n''s child classes (canonical, in prev)
}
```

Where `pc_i` is some prev class that shares an extraction with curr's `child_i`. The set of such candidate prev classes per curr class is called `cover[c]`:

```
cover[c] = { pc : prev.lookup(some extraction of c) = Some(pc) }
```

`cover` and `matches` mutually depend on each other (a match contributes to its parent's `cover`). They are computed jointly via a fixpoint.

#### Pseudocode

```
function enumerate_matches(curr, prev):
    cover   : Map<curr_class, Set<prev_class>> = empty
    matches : Map<(curr_class, node_idx), List<NodeMatch>> = empty

    repeat:
        changed = false
        for each curr e-class c, e-node n at index idx in c:
            children = canonical curr child class ids of n   // [c_1, ..., c_k]
            for each combo (pc_1, ..., pc_k) in cover[c_1] x ... x cover[c_k]:
                // For zero-arity nodes, the cartesian product is a single
                // empty tuple, so the loop runs once with combo = [].
                translated_node = n with children replaced by [pc_1, ..., pc_k]
                if prev.lookup(translated_node) = Some(pc_class):
                    pc_canon = prev.find(pc_class)
                    nm = NodeMatch { prev_class: pc_canon,
                                     prev_children: combo }
                    if nm not already in matches[(c, idx)]:
                        matches[(c, idx)].push(nm)
                        cover[c].insert(pc_canon)
                        changed = true
    until not changed

    return matches
```

The implementation also builds `cover` from `joint` afterwards ([`build_cover`](src/count/novel.rs)) so the sampler can look it up by curr class.

### Phase 2: Joint counts, layered by size

Once matches are known, compute `joint(c, pc)(s)` bottom-up. The recurrence:

```
joint(c, pc)(s) =
    sum over (n at idx in c, m in matches[(c, idx)] with m.prev_class = pc) of:
        sum over (s_1 + ... + s_k) = s - 1 of:
            prod_i  joint(c_i, m.prev_children[i])(s_i)
```

Where `c_i` is the i-th canonical curr child class of `n`. For zero-arity nodes the inner sum is `1` at `s = 1` (one extraction = the node itself).

The matched e-node costs 1, so every child pair is evaluated at a size
strictly below `s`: the recurrence is stratified by size exactly like the
plain counts ([layered_counting.md](layered_counting.md) §2), and the
implementation reuses the same generic layer-by-layer kernel (`LayeredDp`,
one pass per size, each `(pair, size)` cell computed once — no fixpoint,
even on cyclic e-graphs). `joint_children_of` translates the match table
into the kernel's shape: key = `(c, pc)` pair, one node per match, child
keys = `zip(n.children, m.prev_children)`. See
[incremental_probe.md](incremental_probe.md) for the full argument (this
replaced a worklist fixpoint over whole per-pair histograms).

### Phase 3: Derive `novel`

Pointwise subtraction:

```
function derive_novel(plain, joint):
    non_novel : Map<curr_class, Histogram> = sum_pc joint[(c, pc)]
    novel     : Map<curr_class, Histogram>
    for each (c, plain_hist) in plain:
        for each size, total in plain_hist:
            n = total - non_novel[c].get(size, 0)
            if n > 0: novel[c][size] = n
    return novel
```

`novel(c, s)` is the count the sampler uses for both `Sampler::possible_size` and as the outer weighing distribution at the sampling root.

---

## 3. Sampling

### Recursion modes

Once we know we want a novel-rooted term, the recursion threads a `Mode` through every subtree:

```
enum Mode {
    Novel,            // subtree must not be extractable from any prev class
    AgreeWith(pc),    // subtree must be extractable from prev's pc
}
```

- The public `sample(root, size, rng)` always starts in `Mode::Novel`.
- A `Mode::Novel` recursion may *choose* to make some children agree with specific prev classes — those children recurse in `Mode::AgreeWith(pc)`.
- A `Mode::AgreeWith(pc)` recursion is fully determined by the joint table: it just picks compatible (node, match) pairs and recurses.

There is no "Free" mode — once we're committed to producing a novel term, every subtree is constrained either to be novel or to agree with a specific prev class.

### Mode::AgreeWith — straightforward weighted sampling

When sampling at curr-class `c`, target size `s`, and committed prev class `pc`:

```
function sample_agree(c, s, pc, rng):
    candidates = []
    for (idx, n) in eclass(c).nodes:
        for m in matches_of(c, idx) with m.prev_class = pc:
            child_hists = [ joint(child_i, m.prev_children[i])  for i in 0..k ]
            count       = convolve(child_hists, s - 1)[s - 1]
            if count > 0:
                candidates.push((idx, m, weigher.node_weight(count), child_hists))

    pick (idx, m, _, child_hists) ~ WeightedIndex over candidates
    n = eclass(c).nodes[idx]

    children = sample_children(
        children_ids = n.children,
        child_hists  = child_hists,
        modes        = [Mode::AgreeWith(m.prev_children[i])  for i in 0..k],
        child_budget = s - 1,
        rng,
    )

    return stack_children(children, OriginLang(n, c))
```

`AgreeWith` cannot fail in a healthy state: by the time we recurse into it, the parent's profile picked `pc` from `cover[c]`, so at least one match in `c` has `prev_class = pc`.

### Mode::Novel — agreement profiles

This is the heart of the refined sampler. At curr-class `c`, target size `s`:

We need to pick a curr e-node `n` and a way to satisfy "the resulting term is not extractable from any prev class" while using one of `n`'s shapes. For each child slot `i`, the child's extraction has some extractability fingerprint:

- **NOVEL** — the child extraction is not in any prev class, or
- **`Some(pc)`** — the child extraction lives in prev's `pc`.

A full **agreement profile** is `(a_1, ..., a_k)` with each `a_i` drawn from `{None} ∪ { Some(pc) : pc in cover[child_i] }`.

The term is novel-via-`n` iff its profile does not equal `m.prev_children` for any match `m` of `n`. (If the profile equals some `m.prev_children`, the term is exactly extractable from prev's `m.prev_class`.)

#### Why this enumeration is correct

Two prev classes can't share a term, so any non-novel term rooted at `n` is non-novel via *exactly one* match `m`. The profiles `{m.prev_children : m in matches(n)}` partition the non-novel-via-n extractions. Everything else is novel-via-n, with no risk of double-counting. (Distinct matches must differ in at least one child class after rebuild.)

#### Pseudocode

```
function sample_novel(c, s, rng):
    candidates = []

    for (idx, n) in eclass(c).nodes:
        children = canonical curr child class ids of n   // [c_1, ..., c_k]

        slot_options[i] = [None] ++ [Some(pc) for pc in cover[c_i]]

        for profile in cartesian_product(slot_options):
            if completes_some_match(profile, matches_of(c, idx)):
                continue
            child_hists = [
                if profile[i] = None:        novel_histogram(c_i)
                else (profile[i] = Some(pc)): joint(c_i, pc)
                for i in 0..k
            ]
            count = convolve(child_hists, s - 1)[s - 1]
            if count > 0:
                candidates.push((idx, profile,
                                 weigher.node_weight(count),
                                 child_hists))

    pick (idx, profile, _, child_hists) ~ WeightedIndex over candidates
    n = eclass(c).nodes[idx]

    modes[i] =
        if profile[i] = None:        Mode::Novel
        else (profile[i] = Some(pc)): Mode::AgreeWith(pc)

    children = sample_children(
        children_ids = n.children,
        child_hists  = child_hists,
        modes        = modes,
        child_budget = s - 1,
        rng,
    )

    return stack_children(children, OriginLang(n, c))
```

```
function completes_some_match(profile, matches):
    return any m in matches such that
        len(profile) = len(m.prev_children)
        and forall i: profile[i] = Some(m.prev_children[i])
```

Profiles containing any `None` cannot complete any match, since `None` never equals a `Some(pc)`. Profiles with all `Some(...)` complete a match iff their tuple equals some `m.prev_children` exactly.

### Picking child sizes

Once a candidate `(idx, profile-or-match, child_hists)` is chosen, we need to split `child_budget = s - 1` into `(s_1, ..., s_k)` summing to `child_budget`, weighted by per-child counts and a suffix convolution.

This is identical to the plain sampler's child-size loop, just using the mode-specific `child_hists`:

```
function sample_children(children_ids, child_hists, modes, child_budget, rng):
    suffix = right-to-left convolution of child_hists  // suffix[i+1] = conv of i+1..k
    remaining = child_budget
    sampled = []

    for i in 0..k:
        candidates = [
            (s, weigher.child_weight(child_hists[i][s], suffix[i+1][remaining - s]))
            for s in keys(child_hists[i]) if remaining - s ∈ keys(suffix[i+1])
              and suffix[i+1][remaining - s] > 0
        ]
        s_i ~ WeightedIndex over candidates
        remaining -= s_i
        sampled.push(sample_with_mode(children_ids[i], s_i, modes[i], rng))

    return sampled
```

The suffix convolution is built once per call from the chosen `child_hists`, so each child's size is sampled with awareness of the remaining children's joint feasibility. (Analogous to what `PlainTermCount::suffix_cache` precomputes for the plain sampler.)

---

## 4. Correctness

**Claim.** `sample(root, size, rng)` returns a term `t` with `prev.lookup(t) = None` whenever `possible_size(root, size, 0)` holds.

We prove two statements together by induction on `size`:

- **(A)** Sampling under `Mode::AgreeWith(pc)` at curr-class `c` and size `s` returns `t` with `prev.lookup(t) = Some(pc)` (assuming the precondition that `joint[(c, pc)](s) > 0`).
- **(N)** Sampling under `Mode::Novel` at curr-class `c` and size `s` returns `t` with `prev.lookup(t) = None` (assuming `novel(c)(s) > 0`).

### Base case (size 1, leaves)

**(A)** A size-1 candidate in `sample_agree(c, 1, pc)` is a leaf node `n ∈ c` with a match `m` such that `m.prev_class = pc`. By construction of `enumerate_matches`, such a match exists iff `prev.lookup(n) = pc`. So the sampled term is a leaf `n` with `prev.lookup(n) = pc`. ✓

**(N)** A size-1 candidate in `sample_novel(c, 1)` is a leaf node `n ∈ c` whose empty profile `[]` does not complete any match — i.e., no leaf match exists for `n`. By the same construction, this means `prev.lookup(n) = None`. ✓

### Inductive step (size > 1)

**(A) at `(c, s, pc)`.** The sampler picks `(idx, m)` with `m.prev_class = pc` and a child-size split `(s_1, …, s_k)` summing to `s - 1` with each `joint(c_i, m.prev_children[i])(s_i) > 0`. Each child recurses under `Mode::AgreeWith(m.prev_children[i])`; by IH (A), child `i` returns `t_i` with `prev.lookup(t_i) = m.prev_children[i]`. Then `prev.lookup(n(t_1, …, t_k))` looks up the prev e-node with operator `n` and child classes `m.prev_children`, which by definition of the match sits in `m.prev_class = pc`. So `prev.lookup(t) = pc`. ✓

**(N) at `(c, s)`.** Suppose for contradiction that the sampler returns `t = n(t_1, …, t_k)` with `prev.lookup(t) = Some(P)` for some prev class `P`. Then prev contains an e-node `n'` in `P` with canonical child classes `[q_1, …, q_k]` and `prev.lookup(t_i) = q_i` for each `i`.

The chosen profile `(a_1, …, a_k)` and the IHs determine each `t_i`:

- If `a_i = None`: by IH (N), `prev.lookup(t_i) = None`. But we just said `prev.lookup(t_i) = q_i`, a concrete prev class. Contradiction unless every `a_i ≠ None`.
- If `a_i = Some(pc_i)`: by IH (A), `prev.lookup(t_i) = pc_i`. So `q_i = pc_i`.

Hence the profile is `(Some(q_1), …, Some(q_k))`. Each `q_i` is shared between curr's `c_i` and prev's `q_i` (witnessed by `t_i`), so `q_i ∈ cover[c_i]` in match enumeration's internal cover. The cartesian product loop in [`enumerate_matches`](src/count/novel.rs) therefore visits the combo `(q_1, …, q_k)`, calls `prev.lookup` on the translated node, finds `Some(P)`, and records the match `{ prev_class: P, prev_children: [q_1, …, q_k] }` in `matches[(c, idx)]`. But then `completes_some_match` would have rejected the chosen profile. Contradiction.

(The `cover_of` exposed to the sampler is a subset of match enumeration's internal cover — `compute_joint` drops `(c, pc)` pairs whose histogram is empty within `max_size`. That's harmless here: any `q_i` reached by a real sampled child `t_i` is witnessed by `t_i`'s size, which is within budget, so `joint[(c_i, q_i)]` is non-empty and `q_i ∈ cover_of(c_i)`. The slot-options enumeration in `sample_novel` sees it; either way, the `completes_some_match` check sees the match in `matches[(c, idx)]`, which is the only thing that matters.)

So no `t` produced by `Mode::Novel` can have `prev.lookup(t) = Some(_)`, i.e., every produced term is novel. ✓

### Why the joint sum doesn't double-count

The argument above also shows why `non_novel(c)(s) = sum_pc joint[(c, pc)](s)` doesn't double-count: if `t = n(t_1, …, t_k)` is non-novel, the unique prev class `P` containing it together with the unique tuple `[q_1, …, q_k] = [prev.lookup(t_i)]` nails down a unique match `m`. So `t` is counted exactly once across all `(c, pc)` pairs — under `pc = P`, by the match `m`. Two different matches at the same node would have to disagree on at least one `prev_children[i]`, but `prev.lookup(t_i)` is a single value, so at most one match's `prev_children` can equal the tuple of child lookups.

This is the inclusion-exclusion argument referenced in `sample_novel`: the partition is by the unique witnessing match, with no overlap.

### Termination

The match-enumeration fixpoint adds entries monotonically to a finite set (bounded by `(curr classes) × (node indices) × (prev classes) × (prev class tuples)`), so it terminates. The joint counts need no convergence argument: the layered DP runs exactly one pass per size, `1..=max_size`. Sampling is non-recursive in `(c, s)` past the first call: every recursive call has strictly smaller `s` (children get `s_i ≤ s - 1`).

---

## 5. Worked example: `Add(a, b)` after `union(a, b)`

Setup:

```
prev: a, b, Add(a, b)
curr: prev + union(a, b)
```

After the union, curr has:

- merged class `M` containing the e-nodes `Symbol("a")`, `Symbol("b")`.
- root class `R` containing one e-node `Add(M, M)`.

Prev has classes `A` (for `a`), `B` (for `b`), `R_prev` (for `Add(a, b)`).

### Match enumeration

- `Symbol("a")` in `M`: `prev.lookup(Symbol("a")) = Some(A)`. Match
  `{ prev_class: A, prev_children: [] }`.
- `Symbol("b")` in `M`: similar, match with `prev_class: B`.
- `Add(M, M)` in `R`: cartesian product over `cover[M] = {A, B}` for both children:
  - `(A, A)` → `prev.lookup(Add(A, A))` = `None`.
  - `(A, B)` → `prev.lookup(Add(A, B))` = `Some(R_prev)`. Match `{ prev_class: R_prev, prev_children: [A, B] }`.
  - `(B, A)` → `None`.
  - `(B, B)` → `None`.

So `matches[(R, 0)]` has exactly one entry.

### Joint counts

- `joint(M, A)(1) = 1` (just `a`).
- `joint(M, B)(1) = 1` (just `b`).
- `joint(R, R_prev)(3) = joint(M, A)(1) * joint(M, B)(1) = 1`. (One extraction: `Add(a, b)`.)

### Novel histogram

- `plain(M)(1) = 2`, `non_novel(M)(1) = 1 + 1 = 2`, so `novel(M)(1)` is empty (every leaf of `M` is in some prev class).
- `plain(R)(3) = 4` (the four ordered pairs over `{a, b}`), `non_novel(R)(3) = 1`, so `novel(R)(3) = 3`.

### Sampling at `(R, size = 3, Mode::Novel)`

Slot options for `Add(M, M)`'s two children:

- `slot_options[0] = [None, Some(A), Some(B)]`
- `slot_options[1] = [None, Some(A), Some(B)]`

Nine profiles total. The single match `(R_prev, [A, B])` is completed by the profile `(Some(A), Some(B))` only, so eight profiles are novel-via-`n`. Of those, the ones with non-zero count at total budget 2:

- `(Some(A), Some(A))`: child_hists = `[joint(M,A), joint(M,A)] = [{1:1}, {1:1}]`, conv at 2 = 1. Recurse children with `Mode::AgreeWith(A)`. Result: `Add(a, a)`.
- `(Some(B), Some(A))`: similarly → `Add(b, a)`.
- `(Some(B), Some(B))`: similarly → `Add(b, b)`.

Profiles with any `None` have count 0 because `novel_histogram(M)` is empty. The remaining profiles also have count 0 in this graph.

So the sampler picks uniformly (under `CountWeigher`) from the three non-novel-via-prev terms `Add(a,a)`, `Add(b,a)`, `Add(b,b)` but never `Add(a,b)`. This is exactly what `sampling::novel::tests::novel_sample_union_diagonal` asserts.

---

## 5. Public API summary

```rust
// Counting (in src/count/novel.rs):
let plain = PlainTermCount::<C>::new(max_size, &curr);
let novel = NovelTermCount::new(max_size, &curr, &prev, &plain);

// Sampling (in src/sampling/novel.rs):
let sampler = NovelSampler::new(&novel, root, weigher);

if sampler.possible_size(root, size, /* samples */ 0) {
    let term = sampler.sample(root, size, &mut rng);
    // term is novel: not extractable from any e-class in `prev`.
}
```

`NovelSampler` implements the [`Sampler`](src/sampling/mod.rs) trait, so it gets `sample_batch` / `sample_batch_root` for free.
