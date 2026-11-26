# Exact frontier drawing

A core design principle of frontier sampling is that **what is valid** and
**how we pick among valid choices** are completely separate:

- **`FrontierDrawer`** enforces correctness: it enumerates only e-nodes,
  child-state profiles, and child-size splits that are guaranteed to produce a
  novel term of the exact requested size.
- **`Weigher`** controls the probability distribution (`Count` vs. `Uniform`)
  over those already-valid choices, while **`SampledTrie`** ensures no term is
  drawn twice.

Because of this boundary, changing the sampling distribution only means
swapping the `Weigher`—never touching the frontier membership logic.

> **Prerequisites & related docs:**
> - For definitions (`novel`, `joint`), the end-to-end correctness proof, and a
>   full worked example, read [novel_candidates.md](novel_candidates.md).
> - For how the count tables and `matches` are built, see
>   [layered_counting.md](../counting/layered_counting.md) and
>   [novel_size_search.md](../counting/novel_size_search.md).
> - For direct grammar generation of seed terms, see
>   [random_terms.md](../generation/random_terms.md).

## Where the code lives

- [src/sampling/draw/frontier.rs](../../src/sampling/draw/frontier.rs) —
  `FrontierDrawer`, `State`, and feasible branch enumeration.
- [src/sampling/draw/mod.rs](../../src/sampling/draw/mod.rs) —
  shared `Drawer` trait (`draw_batch`, `smallest`, `draw_for_policy`) and
  `draw_children`.
- [src/sampling/draw/trie.rs](../../src/sampling/draw/trie.rs) —
  sparse pre-order derivation trie (`SampledTrie`) and `DrawCursor`.
- [src/sampling/weigher.rs](../../src/sampling/weigher.rs) —
  `Weigher` (`Count` and `Uniform` selection policies).

---

## 1. Design goals

Frontier drawing guarantees four properties **directly by construction**,
without generating unconstrained terms and rejecting ones that already existed
in `prev`:

1. **Frontier membership:** every drawn term is extractable from `curr` and
   absent from *every* e-class in `prev`.
2. **Exact size:** every drawn term has the exact requested AST node count `s`.
3. **Termination on cycles:** drawing terminates for any finite `s`, even when
   `curr` contains e-class cycles.
4. **Pluggable distributions:** different weighting policies can be swapped in
   without duplicating correctness logic.

---

## 2. Frontier membership as a tree automaton

Every recursive subtree is constructed under one of two states:

```rust
enum State {
    Novel,              // subtree must NOT exist in any prev e-class
    SharedWith(PrevId), // subtree MUST extract into this exact prev e-class
}
```

Bottom-up, a parent e-node's state is completely determined by its children's
states:

1. **Any child is `Novel`** → the parent is `Novel` (if a subterm doesn't exist
   in `prev`, no larger term containing it can exist in `prev`).
2. **All children are `SharedWith(pc_i)`** → replace the node's children with
   `[pc_1, ..., pc_k]` and look up the translated node in `prev`:
   - Found in `parent_pc` → the parent is `SharedWith(parent_pc)`.
   - Not found → the parent is `Novel`.

### `Novel` as a proof obligation

Drawing runs **top-down**, starting at `(root, s, State::Novel)`. Think of
`Novel` as a proof obligation ("guarantee this subtree cannot be reconstructed
in `prev`"). At any e-node, you can satisfy it in two ways:

- **Discharge it here:** pick `SharedWith(pc_i)` for every child such that the
  translated parent has **no** match in `prev`.
- **Delegate it downward:** mark at least one child as `Novel`; its failure to
  reconstruct in `prev` automatically makes the parent and every ancestor novel
  too.

**Example:** Suppose `prev` contains `F(A, B)` but *not* `F(B, B)`. At an
e-node `F(c_1, c_2)` under `State::Novel`:

| Child profile | Accepted? | Why |
| :--- | :---: | :--- |
| `[SharedWith(A), SharedWith(B)]` | No | Reconstructs `F(A, B)` in `prev` |
| `[SharedWith(B), SharedWith(B)]` | Yes | Discharges obligation at `F` (`F(B, B)` is absent from `prev`) |
| `[Novel, SharedWith(B)]` | Yes | Delegates obligation to the first child |

Every child is assigned either `Novel` or one specific `SharedWith(pc)`—there
is no "free / unconstrained" state. Because any concrete term belongs to at
most one rebuilt `prev` e-class, these states are **strictly disjoint**,
keeping the counts exact and preventing double-counting.

> **Why not just require a newly added e-node?** A novel term does not need any
> brand-new e-node—merging two existing e-classes can create new combinations
> of old e-nodes (such as `F(B, B)` above). The two-state automaton captures
> both new nodes and new combinations seamlessly.

---

## 3. Enumerating feasible branches and child sizes

The precomputed histograms from `NovelTermCount` map directly onto the two
automaton states:

```text
histogram(c, Novel)          = novel[c]
histogram(c, SharedWith(pc)) = joint[(c, pc)]
```

### Branches in `State::SharedWith(pc)`

When a subtree must land in `prev` class `pc`, only `(node, match)` pairs with
`m.prev_class == pc` are eligible (`shared_branches`):

```text
function shared_branches(c, s, pc):
    branches = []
    for (idx, n) in eclass(c).nodes:
        for m in matches[(c, idx)] with m.prev_class == pc:
            child_states = [SharedWith(pc_i) for pc_i in m.prev_children]
            child_hists  = [joint[(c_i, pc_i)] for (c_i, pc_i) in zip(n.children, m.prev_children)]
            count        = convolve_at(child_hists, s - 1)
            if count > 0:
                branches.push(Branch { idx, child_states, count, child_hists })
    return branches
```

### Branches in `State::Novel` (agreement profiles)

When a subtree must be novel, we pair each e-node `n` in `c` with every
**agreement profile** `(state_1, ..., state_k)` over its children, where each
slot `i` picks from `[Novel] ++ [SharedWith(pc) for pc in cover[c_i]]`.

A profile produces a non-novel term iff every slot is `SharedWith(pc_i)` and
`(pc_1, ..., pc_k)` equals `m.prev_children` for some match `m` of `n`.
Rejecting those profiles leaves only novel productions (`novel_branches`):

```text
function novel_branches(c, s):
    branches = []
    for (idx, n) in eclass(c).nodes:
        slot_options = [ [Novel] ++ [SharedWith(pc) for pc in cover[c_i]]
                         for c_i in n.children ]
        for profile in cartesian_product(slot_options):
            if completes_some_match(profile, matches[(c, idx)]):
                continue
            child_hists = [ histogram(c_i, profile[i]) for i in 0..k ]
            count       = convolve_at(child_hists, s - 1)
            if count > 0:
                branches.push(Branch { idx, child_states: profile, count, child_hists })
    return branches

function completes_some_match(profile, matches):
    return any m in matches where
        profile == [SharedWith(pc) for pc in m.prev_children]
```

### Splitting child sizes (`draw_children`)

Once `Weigher` picks a branch, `draw::draw_children` splits `child_budget = s - 1`
into `(s_1, ..., s_k)` using a right-to-left suffix convolution built on demand
from `branch.child_hists`:

```text
function draw_children(children_ids, child_hists, child_states, child_budget, cursor):
    suffix = right-to-left convolution of child_hists   // suffix[i+1] covers children i+1..k
    remaining = child_budget
    drawn = []

    for i in 0..k:
        // Only sizes s_i where both child_hists[i][s_i] and suffix[i+1][remaining - s_i]
        // are positive are offered to `weigher`.
        s_i = weigher.pick_size(child_hists[i], suffix[i+1], remaining, cursor)
        remaining -= s_i
        drawn.push(construct(children_ids[i], s_i, child_states[i], cursor))

    return drawn
```

As a result, the `Weigher` **only ever sees valid branches and valid size
splits**.

---

## 4. Selection policies (`Weigher`)

While `FrontierDrawer` determines *which* choices are valid, `Weigher` picks
among them using the remaining unsampled capacity tracked by `DrawCursor`:

| Policy | `Weigher` mode | Local weight of a choice | Resulting distribution |
| :--- | :--- | :--- | :--- |
| `Policy::Count` | `Weigher::Count` | Exact remaining unsampled completions (`local_count * rest_total - used`) | Exact **uniform** sampling without replacement over all remaining frontier terms of that size |
| `Policy::Uniform` | `Weigher::Uniform` | Equal weight (`1`) for every choice with positive remaining capacity | Uniform over local branching decisions (exhausted subtrees excluded) |
| `Policy::Smallest` | `Weigher::Uniform` | Picks the single smallest available size (`Drawer::smallest`) with a fixed seed | One deterministic smallest novel term |

Both `Count` and `Uniform` operate over the exact same feasible productions:
they change *which* valid terms are more likely, never whether a term is novel
or has the right size.

---

## 5. Without-replacement sampling (`SampledTrie`)

Every concrete term corresponds to a **unique** pre-order sequence of branch
and child-size decisions (derivations are unambiguous). `Drawer::draw_batch`
exploits this to sample without replacement in **one pass per requested term,
with zero rejection retries**:

1. **Feasibility check up front:** `draw_batch` checks `root_histogram()` to
   verify that each requested size bucket has enough terms (returning
   `Err(DrawingError::InsufficientTerms)` otherwise).
2. **One `SampledTrie` per size bucket:** Each node in the sparse trie records
   how many previously drawn root terms passed through that decision prefix.
3. **Subtracting used capacity:** While walking an already-visited prefix,
   `Weigher` subtracts `cursor.used(key)` from each choice's capacity so
   exhausted subtrees are never revisited.
4. **Fast path off-trie:** As soon as a draw takes an unvisited edge, no prior
   draw can collide with the remaining suffix. Residual-capacity tracking
   (`rest_total`) switches off (`None`) for the rest of that draw while still
   recording its path in the trie.
5. **Deterministic ordering:** Drawn expressions in each size bucket are sorted
   before returning, keeping output order stable for a fixed seed.

For the formal induction proof that every drawn term has exact size `s` and
`prev.lookup(t) = None`, see
[novel_candidates.md §3](novel_candidates.md#3-correctness).

---

## 6. Possible follow-up work

Future optimizations should keep the boundary between feasibility and selection
intact:

1. **Lazy profile enumeration:** stream child-state profiles instead of
   materializing the full Cartesian product for high-arity operators.
2. **Branch / suffix caching:** cache feasible branches or suffix convolutions
   by `(c, s, state)` if large batches spend noticeable time recomputing them.

Neither optimization needs to change how frontier membership or exact size is
proven—they only speed up enumerating the same feasible productions.
