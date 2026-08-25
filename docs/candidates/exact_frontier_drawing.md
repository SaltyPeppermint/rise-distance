# Exact frontier drawing

Frontier membership and random selection are separate responsibilities.
`FrontierSpace` owns the correctness-critical constrained derivation space;
`IndependentFrontierDrawer` chooses among the feasible productions it exposes.
Changing the probability distribution therefore does not require
reimplementing the frontier test.

The relevant implementation is:

- [`src/candidates/count/novel.rs`](../../src/candidates/count/novel.rs):
  previous-node matches plus plain, joint, and novel histograms;
- [`src/candidates/draw/frontier/space.rs`](../../src/candidates/draw/frontier/space.rs):
  frontier states and feasible derivations; and
- [`src/candidates/draw/frontier/independent.rs`](../../src/candidates/draw/frontier/independent.rs):
  independent weighted drawing over that space.

The detailed counting argument is in
[`exact_novel_candidates.md`](exact_novel_candidates.md). Direct grammar
drawing, including binder handling, is documented separately in
[`../generation/random_terms.md`](../generation/random_terms.md).

## Responsibilities

The frontier implementation must:

1. return only current-graph terms that are absent from every previous
   e-class;
2. construct terms of exactly the requested size;
3. terminate on cyclic e-graphs for every finite requested size; and
4. permit different random distributions over the valid derivations without
   duplicating the correctness logic.

Construction is direct. It does not generate an unconstrained term and reject
it after checking previous membership.

## Frontier membership as a tree automaton

For a current e-class, a concrete subtree is constructed under one of two
states:

```rust
enum FrontierState {
    OutsidePrev,
    InsidePrev(Id),
}
```

- `InsidePrev(pc)` means that the concrete subtree must also be extractable
  from previous e-class `pc`.
- `OutsidePrev` means that the concrete subtree must not be extractable from
  any previous e-class.

The transition at a current e-node is determined by its child states:

1. If any child is `OutsidePrev`, the parent is also `OutsidePrev`.
2. Otherwise every child is `InsidePrev(pc_i)`. Replace the current e-node's
   children with those previous-class ids and look up the translated node in
   the previous graph.
   - A successful lookup places the parent in `InsidePrev(parent_pc)`.
   - A failed lookup places the parent in `OutsidePrev`.

### `OutsidePrev` as a proof obligation

Construction runs top-down, so `OutsidePrev` is an obligation that a selected
production must discharge or delegate:

- An all-`InsidePrev` child profile discharges the obligation at the current
  node when the translated parent has no previous-node match.
- A profile containing an `OutsidePrev` child delegates the obligation to that
  child. Its eventual failure to reconstruct also makes every ancestor absent
  from the previous graph.

For example, if the previous graph contains `F(A, B)` but not `F(B, B)`, then:

```text
[InsidePrev(A), InsidePrev(B)]  rejected: reconstructs F(A, B)
[InsidePrev(B), InsidePrev(B)]  accepted: steps outside at F
[OutsidePrev,  InsidePrev(B)]   accepted: delegates to the first child
```

Every child remains classified as either `OutsidePrev` or one particular
`InsidePrev(pc)`. These cases are disjoint because a concrete previous term
belongs to a unique rebuilt previous e-class.

Drawing a frontier term starts at:

```text
(current root class, requested size, OutsidePrev)
```

This is more precise than requiring the term to contain a newly added e-node.
A novel term can arise solely from a new combination created by merging child
classes.

## Count tables and feasible productions

The counted implementation maps its histograms onto the automaton states:

```text
histogram(current_class, OutsidePrev)
    = novel[current_class]

histogram(current_class, InsidePrev(previous_class))
    = joint[(current_class, previous_class)]
```

For `InsidePrev(pc)`, `FrontierSpace` considers current e-node matches whose
previous parent is `pc`. Each child receives the corresponding
`InsidePrev(previous_child)` state.

For `OutsidePrev`, each child slot receives these possible states:

```text
OutsidePrev
InsidePrev(pc) for each previous class in the child's match cover
```

`FrontierSpace` enumerates child-state profiles and rejects every profile that
exactly completes a known previous-node match. It then uses the state
histograms and exact convolution to retain only profiles whose children can
fill the requested parent size.

After choosing a production, suffix convolutions restrict each child-size
choice to values that leave a feasible exact-size remainder for the later
children. Consequently, the random selection layer receives only feasible
branches and size splits.

## Selection distributions

`IndependentFrontierDrawer` draws each complete expression without reference
to earlier expressions in the batch. A `Weigher` controls its local random
choices:

- `CountWeigher` weights a branch by its number of complete expressions and a
  child-size split by `child_count * rest_count`.
- `NaiveWeigher` assigns equal weight to every feasible local branch and
  child-size choice.

Both distributions operate over the same `FrontierSpace`; they can affect
which valid expression is likely, but not its size or frontier membership.

## Correctness argument

The recursive invariants are:

- constructing `(c, s, InsidePrev(pc))` returns a size-`s` extraction from
  current class `c` whose lookup in the previous graph is `pc`; and
- constructing `(c, s, OutsidePrev)` returns a size-`s` extraction from current
  class `c` whose lookup in the previous graph fails.

For `InsidePrev(pc)`, every exposed current-node/previous-match pair has
previous parent `pc`, and every child receives the matched previous-child
state. The induction hypothesis establishes the child lookups, so the
translated parent exists in `pc`.

For `OutsidePrev`, `FrontierSpace` rejects every child-state profile equal to a
known previous-node match. If the selected profile contains `OutsidePrev`, the
induction hypothesis makes reconstruction of a previous parent impossible. If
all children are `InsidePrev`, the rejected-match check establishes that the
translated parent is absent.

For both states, suffix feasibility makes the selected child sizes sum to
`s - 1`; adding the current e-node gives total size `s`.

Every recursive child size is strictly smaller than its parent size. The proof
and construction therefore remain well-founded even when the e-graph contains
cycles.

## Duplicate collection and work bound

Repeated direct draws can produce the same complete expression. The shared
`Drawer::draw_size` implementation inserts completed expressions into a set
and retries until it reaches the requested distinct count or exhausts its
fixed draw budget.

The exact histogram caps the target to the known number of available terms.
The retry phase is additionally bounded by
`MAX_DRAW_ATTEMPTS_PER_CANDIDATE * requested_count`. Duplicate handling is
therefore finite and separate from frontier correctness: every attempted draw
is already a valid exact-size frontier term.

The returned expressions are sorted after collection, making the result stable
for a fixed seed and deterministic traversal order.

## Possible follow-up work

Potential optimizations should preserve the boundary between feasibility and
selection:

1. Enumerate child-state profiles lazily so high-arity operators do not require
   materializing their full Cartesian product.
2. Cache feasible branches or suffix convolutions by `(class, size, state)` if
   repeated batch construction makes recomputation significant.
3. Implement exact rank/unrank over counted frontier derivations if guaranteed
   drawing without replacement becomes valuable.

None of these changes should participate in proving size or frontier status;
they operate only on productions already established as feasible.
