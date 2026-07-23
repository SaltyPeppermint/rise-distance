# Frontier sampling policies

This document records why frontier membership was separated from sampling
policy, how the shared constrained derivation space works, and how the
coverage-balanced sampler increases structural variety without tree-distance
comparisons or unbounded rejection sampling.

The relevant implementation is:

- [`src/sampling/count/novel.rs`](../src/sampling/count/novel.rs): match,
  joint-count, and outside-`prev` histograms.
- [`src/sampling/sampler/frontier/space.rs`](../src/sampling/sampler/frontier/space.rs):
  shared frontier states and feasible derivations.
- [`src/sampling/sampler/frontier/independent.rs`](../src/sampling/sampler/frontier/independent.rs):
  independent weighted sampling over that space.
- [`src/sampling/sampler/frontier/balanced.rs`](../src/sampling/sampler/frontier/balanced.rs):
  batch-local coverage-balanced sampling over the same space.

The detailed counting argument remains in
[`novel_sampling.md`](novel_sampling.md).

## Motivation

The original `NovelSampler` had two responsibilities:

1. Preserve the frontier invariant: return only current-graph terms which are
   not extractable from any e-class in the previous graph.
2. Choose a probability distribution over those terms.

Those concerns need not be coupled. In particular, count-proportional sampling
can be correct and uniform over individual terms while still putting almost
all probability mass in one enormous family of structurally similar terms.
Changing that distribution should not require reimplementing the
correctness-critical frontier test.

Post-hoc alternatives have unattractive costs:

- Rejection sampling can take unbounded work when the proposal distribution is
  concentrated.
- Zhang-Shasha tree distance requires dynamic programming for many pairs of
  completed trees.
- A bounded candidate pool has predictable cost, but cannot cover a rare
  structural family it never proposes.

The chosen design instead balances decisions while a valid frontier term is
being constructed. Every partial derivation is feasible, every completed term
is on the frontier, and the amount of construction work is bounded by a fixed
multiple of the requested batch size.

## Frontier membership as a two-state tree automaton

For a current e-class `c`, a subtree has one of these states:

```rust
enum FrontierState {
    OutsidePrev,
    InsidePrev(Id),
}
```

- `InsidePrev(pc)` means the concrete subtree is also extractable from
  previous e-class `pc`.
- `OutsidePrev` means it is not extractable from any previous e-class.

The transition at a current e-node is deterministic once the child states are
known:

1. If any child is `OutsidePrev`, the parent is `OutsidePrev`.
2. Otherwise every child is `InsidePrev(pc_i)`. Replace the current e-node's
   children with the `pc_i` values and look it up in `prev`.
   - A successful lookup gives `InsidePrev(parent_pc)`.
   - A failed lookup gives `OutsidePrev`.

Sampling a frontier term means constructing a derivation rooted at:

```text
(current root class, requested size, OutsidePrev)
```

This definition is stronger and more accurate than requiring a term to contain
a "new e-node." Equality saturation can create a new combination solely by
merging child classes; all operators in the resulting frontier term may
already have appeared in `prev`.

The existing counting tables map directly onto the automaton:

```text
histogram(c, OutsidePrev)   = novel[c]
histogram(c, InsidePrev(p)) = joint[(c, p)]
```

`FrontierSpace` uses these histograms plus the precomputed node matches to
enumerate only feasible `(e-node, child-state profile)` branches. It also
enumerates only child sizes that leave a feasible budget for the remaining
children.

Consequently, a sampling policy never receives an invalid choice. Frontier
correctness belongs to `FrontierSpace`, not to an individual random policy.

## Sampler names and responsibilities

The original `NovelSampler` is now named `IndependentFrontierSampler`.

"Independent" describes its important behavior: every complete term is drawn
without knowledge of earlier terms in the batch. It still supports both
existing weighers:

- `CountWeigher`: weight a derivation by the number of complete terms beneath
  it.
- `NaiveWeigher`: give feasible local derivation choices equal weight.

Both use `FrontierSpace`, so the refactor does not change their frontier
semantics.

`BalancedFrontierSampler` also uses `FrontierSpace`, but shares a coverage map
across all constructions requested from one `sample_size` call.

## Coverage-balanced construction

The balanced policy records three local feature families:

1. **Node coverage**

   ```text
   (current class, frontier state, e-node index)
   ```

2. **Profile coverage**

   ```text
   (node key, child frontier states)
   ```

   This captures the reason a term is on the frontier: novelty introduced at
   the current node, propagated through one or more children, or produced by a
   particular combination of agreements with previous e-classes.

3. **Child-size coverage**

   ```text
   (profile key, child position, chosen child size)
   ```

For each feasible choice, the current implementation computes an integer
penalty from its usage counts. It selects a minimum-penalty choice and breaks
ties with the caller's seeded RNG:

```text
branch penalty =
    node_penalty    * node_usage
  + profile_penalty * profile_usage

size penalty =
    child_size_penalty * child_size_usage
```

The defaults are:

```text
node_penalty       = 2
profile_penalty    = 1
child_size_penalty = 1
```

Favoring node coverage first prevents an e-node with many profiles from
immediately dominating all other e-nodes. Profile and child-size coverage then
spread samples within each structural family.

Coverage is local to a `sample_size` batch. `sample()` creates an empty
coverage map because one isolated draw has no earlier terms to diversify
against. Keeping the state local avoids interior mutability, call-order
dependence between unrelated batches, and synchronization requirements.

## Work bound and duplicate behavior

`BalancedFrontierSampler::sample_size` targets the requested number of distinct
outputs, capped by the exact number of available frontier terms reported by the
histogram. It retains one coverage map while refilling duplicates, with the
same `32 × requested` construction budget as the independent sampler.

Completed terms are placed in a set before return. An exact duplicate is
collapsed and another construction is attempted while budget remains.
Therefore:

- Runtime remains bounded when the frontier is dominated by similar or
  identical derivations.
- The method usually fills duplicate-induced shortfalls, while retaining the
  coverage policy across refill draws.
- It can still return fewer than requested after exhausting the bounded work
  budget, consistent with the `Sampler::sample_size` "up to" contract.

`PrecomputePackage::sample_balanced_frontier_terms` is the production entry
point. It uses the same size-distribution logic as `sample_frontier_terms`, but
always samples the frontier and applies the balanced policy.
`sample_balanced_frontier_terms_with_config` exposes the three penalties when
an experiment needs to tune them.

## Correctness argument

The two recursive claims are:

- Constructing `(c, s, InsidePrev(pc))` returns a size-`s` extraction from
  current class `c` whose lookup in `prev` is `pc`.
- Constructing `(c, s, OutsidePrev)` returns a size-`s` extraction from current
  class `c` whose lookup in `prev` fails.

For `InsidePrev(pc)`, `FrontierSpace` exposes only current-node/previous-match
pairs whose previous parent is `pc`, and gives every child the corresponding
`InsidePrev(prev_child)` state. The induction hypothesis establishes every
child lookup, so the translated parent node exists in `pc`.

For `OutsidePrev`, `FrontierSpace` rejects every child-state profile equal to a
known previous-node match. If a selected profile contains `OutsidePrev`, the
induction hypothesis makes a previous parent lookup impossible. If all
children are `InsidePrev`, the rejected-match check ensures that their
translated parent node does not exist in `prev`.

Both independent and balanced policies choose exclusively from these branches,
so their different selection distributions cannot affect this proof.

## Future optimizations

The current version deliberately starts with a small, auditable policy.
Promising follow-up work is:

1. **Capacity-tempered balancing.** Incorporate a branch's exact capacity using
   a weight such as `capacity^alpha / (1 + usage)^beta`. `alpha = 0` recovers
   structural balancing; `alpha = 1` approaches count-proportional sampling.
   This needs a stable logarithmic conversion for very large `BigUint` counts.

2. **Exact sampling without replacement.** Implement rank/unrank over the
   frontier derivation grammar. Production blocks have known counts, and child
   products can use mixed-radix ranks. Distinct ranks would guarantee distinct
   terms with no rejection.

3. **Hierarchical quota allocation.** Allocate a batch quota among root
   branches, profiles, and size splits before constructing terms. Concave
   capacity weights would provide explicit stratification rather than the
   current online least-used heuristic.

4. **Cross-size coverage.** Override `sample_batch` and share selected coverage
   features across size buckets. At present each `sample_size` call starts a
   fresh coverage map.

5. **Lazy profile generation.** The Cartesian product of child-state options
   is harmless for the usual arity-zero-to-two languages, but a factored
   dynamic program would avoid materializing every profile for high-arity
   operators.

6. **Branch and convolution caches.** Cache feasible branches and suffix
   convolutions by `(class, size, state)` when repeated batch construction
   makes their recomputation measurable.

7. **Diversity measurements.** Evaluate node/profile coverage, structural
   shingle Jaccard distance, and downstream search success. Tree-edit distance
   can remain an offline evaluation metric without entering the sampling hot
   path.

8. **CLI tuning.** The balanced policy is available as the `balanced`
   `SampleStrategy`, as `sample_balanced` in the guide menu, and through the
   Python driver's `no_replacement_balanced` / `with_replacement_balanced`
   strategies. A future CLI can expose the individual `BalanceConfig`
   penalties when experiments need values other than the defaults.
