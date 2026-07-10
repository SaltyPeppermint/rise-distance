# Size-incremental exact search

`backoff_precompute` finds the `sizes`-th smallest novel term size at the
root in one exact, size-incremental scan. It then builds the sampling package
at exactly that size.

The relevant code is:

- [src/sampling/count/layered.rs](../src/sampling/count/layered.rs) — the
  generic `LayeredDp` kernel shared by plain and joint counting.
- [src/sampling/count/novel.rs](../src/sampling/count/novel.rs) — layered
  joint counting and `find_novel_root_sizes`.
- [src/sampling/precompute.rs](../src/sampling/precompute.rs) — cap handling
  and package construction.

## Why the search is incremental

For both plain and joint counting, a term of size `s` depends only on child
terms whose sizes are strictly below `s`: the root e-node itself contributes
one to the size. Counts are therefore stratified by size even when the
e-graph contains cycles.

The joint recurrence uses `(curr_class, prev_class)` as its key. For a match
of a current e-node against a previous class, its child keys are the zipped
current and previous children. Apart from that different key shape, it is
the same layered recurrence as plain counting. `LayeredDp` evaluates each
`(key, size)` cell once, in size order.

After layer `s` completes, the root's exact novel count at that size is final:

```text
novel(root, s) = plain(root, s)
               - sum over pc of joint((root, pc), s)
```

Later layers cannot change it. `find_novel_root_sizes` can record nonzero
sizes in ascending order and stop as soon as it has recorded `sizes` of them.
Nothing above the answer is computed during the scan.

## Exact arithmetic

The scan uses `BigUint` throughout. A size is reported exactly when its novel
count is nonzero; there are no modular fingerprints, collisions, or fallback
verification paths.

The scan is root-restricted. The plain DP computes only classes reachable
from the root within the global size cap. Each joint pair `(c, pc)` inherits
the budget of its current class `c`; pairs whose current class is unreachable
are omitted. This is sound because every joint term of `(c, pc)` is also a
plain term of `c`, so the plain subterm budget bounds every joint dependency
needed by a root query.

The package build remains a separate pass because it produces the complete
novel histograms and suffix caches needed by the sampler. Match enumeration
is independent of the size cap and is reused by both passes.

## `backoff_precompute` control flow

```text
matches = enumerate_matches(curr, prev)
cap = start_size + max_retries * retry_step
novel_sizes = find_novel_root_sizes(cap, ..., stop_after=sizes)

if fewer than `sizes` were found:
    return Err(cap)

max_size = novel_sizes[sizes - 1]
build and return the package at max_size
```

`start_size`, `max_retries`, and `retry_step` retain their command-line/API
meaning only through the cap. There is no sequence of retries at those
bounds. On success, both the returned `usize` and the package's `max_size`
are the `sizes`-th smallest novel size.

## Verification

Tests compare the root-restricted scan with the full exact novel histogram
for every class in several graph shapes, including a cyclic graph. The
end-to-end backoff test has novel sizes `5, 7, 9, ...`; requesting three
sizes returns and builds the package at `9`.
