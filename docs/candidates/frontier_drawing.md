# Exact frontier drawing

Frontier membership and random selection are separate responsibilities.
`FrontierDrawer` enumerates the correctness-critical feasible productions,
and its `Weigher` chooses among them. Changing the probability distribution
therefore only means swapping the `Weigher`, not reimplementing the frontier
test.

The relevant implementation is:

- [`src/sampling/count/novel.rs`](../../src/sampling/count/novel.rs):
  previous-node matches plus whole, joint, and novel histograms;
- [`src/sampling/draw/frontier.rs`](../../src/sampling/draw/frontier.rs):
  frontier states, feasible derivations, and independent weighted drawing
  over them; and
- [`src/sampling/draw/weigher.rs`](../../src/sampling/draw/weigher.rs):
  the `CountWeigher` and `UniformWeigher` selection policies.

The detailed counting argument is in
[`novel_candidates.md`](novel_candidates.md). Direct grammar
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
enum State {
    Novel,
    SharedWith(Id),
}
```

- `SharedWith(pc)` means that the concrete subtree must also be extractable
  from previous e-class `pc`.
- `Novel` means that the concrete subtree must not be extractable from
  any previous e-class.

The transition at a current e-node is determined by its child states:

1. If any child is `Novel`, the parent is also `Novel`.
2. Otherwise every child is `SharedWith(pc_i)`. Replace the current e-node's
   children with those previous-class ids and look up the translated node in
   the previous graph.
   - A successful lookup places the parent in `SharedWith(parent_pc)`.
   - A failed lookup places the parent in `Novel`.

### `Novel` as a proof obligation

Construction runs top-down, so `Novel` is an obligation that a selected
production must discharge or delegate:

- An all-`SharedWith` child profile discharges the obligation at the current
  node when the translated parent has no previous-node match.
- A profile containing a `Novel` child delegates the obligation to that
  child. Its eventual failure to reconstruct also makes every ancestor absent
  from the previous graph.

For example, if the previous graph contains `F(A, B)` but not `F(B, B)`, then:

```text
[SharedWith(A), SharedWith(B)]  rejected: reconstructs F(A, B)
[SharedWith(B), SharedWith(B)]  accepted: steps outside at F
[Novel,  SharedWith(B)]   accepted: delegates to the first child
```

Every child remains classified as either `Novel` or one particular
`SharedWith(pc)`. These cases are disjoint because a concrete previous term
belongs to a unique rebuilt previous e-class.

Drawing a frontier term starts at:

```text
(current root class, requested size, Novel)
```

This is more precise than requiring the term to contain a newly added e-node.
A novel term can arise solely from a new combination created by merging child
classes.

## Count tables and feasible productions

The counted implementation maps its histograms onto the automaton states:

```text
histogram(current_class, Novel)
    = novel[current_class]

histogram(current_class, SharedWith(previous_class))
    = joint[(current_class, previous_class)]
```

For `SharedWith(pc)`, `FrontierDrawer` considers current e-node matches whose
previous parent is `pc`. Each child receives the corresponding
`SharedWith(previous_child)` state.

For `Novel`, each child slot receives these possible states:

```text
Novel
SharedWith(pc) for each previous class in the child's match cover
```

`FrontierDrawer` enumerates child-state profiles and rejects every profile that
exactly completes a known previous-node match. It then uses the state
histograms and exact convolution to retain only profiles whose children can
fill the requested parent size.

After choosing a production, suffix convolutions restrict each child-size
choice to values that leave a feasible exact-size remainder for the later
children. Consequently, the random selection layer receives only feasible
branches and size splits.

## Selection distributions

`FrontierDrawer` draws each complete expression without reference
to earlier expressions in the batch. A `Weigher` controls its local random
choices:

- `CountWeigher` weights a branch by its number of complete expressions and a
  child-size split by `child_count * rest_count`.
- `UniformWeigher` assigns equal weight to every feasible local branch and
  child-size choice.

Both distributions operate over the same feasible productions; they can affect
which valid expression is likely, but not its size or frontier membership.

## Correctness argument

The recursive invariants are:

- constructing `(c, s, SharedWith(pc))` returns a size-`s` extraction from
  current class `c` whose lookup in the previous graph is `pc`; and
- constructing `(c, s, Novel)` returns a size-`s` extraction from current
  class `c` whose lookup in the previous graph fails.

For `SharedWith(pc)`, every exposed current-node/previous-match pair has
previous parent `pc`, and every child receives the matched previous-child
state. The induction hypothesis establishes the child lookups, so the
translated parent exists in `pc`.

For `Novel`, `FrontierDrawer` rejects every child-state profile equal to a
known previous-node match. If the selected profile contains `Novel`, the
induction hypothesis makes reconstruction of a previous parent impossible. If
all children are `SharedWith`, the rejected-match check establishes that the
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
