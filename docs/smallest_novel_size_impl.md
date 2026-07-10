# Smallest novel size: implementation notes

Short description of the change that removed `backoff_precompute`'s probe
retry schedule. Full motivation and correctness arguments live in
[incremental_probe.md](incremental_probe.md); background in
[layered_counting.md](layered_counting.md) and
[fingerprint_probe.md](fingerprint_probe.md).

## What was slow

`backoff_precompute` had to discover where the frontier begins (the smallest
novel term size at the root) by trial: fingerprint-probe at `start_size`,
`start_size + retry_step`, …, each probe rerunning the full `Mod61`
plain+joint pipeline from scratch. Every bound below the smallest novel size
was a complete, wasted run, and the joint side of each probe was still a
whole-histogram worklist fixpoint (quadratic re-convolutions on cycles).

## The change

One probe, layered by size, that stops itself:

1. **`src/sampling/count/layered.rs`** — the size-layered DP loop of
   `count_terms` moved into a generic kernel `LayeredDp<K, C>` (per-key node
   lists, budgets, suffix tables, histograms, and a `step()` that completes
   one size layer). `count_terms` is now a thin wrapper: build class
   children/budgets via `plain_dp`, step to the limit, repackage as
   `CountData`. No behavior change.

2. **`src/sampling/count/novel.rs`** — `compute_joint` no longer runs a
   worklist fixpoint over whole histograms. `joint_children_of` translates
   the match table into the kernel's shape (key = `(curr_class, prev_class)`
   pair; one node per match; child keys =
   `zip(node children, match prev_children)`), and the same `LayeredDp`
   computes each `(pair, size)` cell exactly once — valid because a match's
   e-node costs 1, so joint counts stratify by size just like plain counts.
   The old fixpoint (`compute_pair_histogram`, `PairDeps`, `PairMatches`,
   its `UniqueQueue`) is deleted.

3. **`src/sampling/count/novel.rs`** — `probe_novel_root_sizes` now builds
   the rooted plain DP and the pair DP (pair budgets inherited from the
   curr class's rooted budget; pairs of unreachable classes skipped) and
   steps both **in lockstep**. After each layer `s`,
   `novel(s) = plain[root](s) − Σ_pc joint[(root, pc)](s)` is final, so the
   probe records novel sizes in ascending order and exits as soon as
   `stop_after` of them are found.

4. **`src/sampling/mod.rs`** — `backoff_precompute` calls the probe once
   with `cap = start_size + max_retries * retry_step` and `stop_after =
   sizes`. The retry loop over bounds is gone; the smallest novel size is
   found at exactly its own layer, and the layer the probe stops at *is* the
   `max_size` for the single exact `BigUint` run. The verified exact run and
   the exact fallback loop are unchanged, as is the `Err` value.

## Effect

- Finding the initial starting value is no longer a search: the first novel
  size falls out of the one incremental run at exactly its layer, sizes
  below it are processed once (any successful probe needed them anyway), and
  nothing beyond the `sizes`-th novel size is computed.
- The exact analysis also got cheaper: its joint table now comes from the
  layered kernel (one convolution cell per pair and size) instead of the
  fixpoint.
- Public behavior of `backoff_precompute` is unchanged (same result value,
  same `Err`, same fallback); `probe_novel_root_sizes` gained a `stop_after`
  parameter.

## Verification

- Existing tests cover the rewrite: probe-vs-exact agreement over every
  class as root (including a cyclic graph), the novel-sampler invariants on
  top of the layered joint table, and the end-to-end `backoff_precompute`
  test (novel sizes 5, 7, 9 → `max_size = 9`).
- `cargo nextest run`: 146/146 passed; `cargo clippy --all-targets` clean.
