# Smallest novel sizes: implementation notes

`backoff_precompute` used to search a retry schedule by rerunning the whole
counting pipeline at successively larger bounds. It now performs one exact,
layer-by-layer root scan up to the old schedule's final cap.

The implementation has three pieces:

1. `LayeredDp<K, C>` owns the per-key node lists, budgets, suffix tables, and
   histograms. `step()` completes one size layer. Plain classes and joint
   `(curr_class, prev_class)` pairs use the same kernel.
2. `find_novel_root_sizes` advances exact `BigUint` plain and joint DPs in
   lockstep. Novelty at size `s` is final after layer `s`, so it stops at the
   requested number of novel sizes.
3. `backoff_precompute` uses
   `start_size + max_retries * retry_step` as a cap, returns `Err(cap)` when
   the scan finds too few sizes, and otherwise builds the package at the
   exact last size found.

The package build is still necessary after the scan: the scan keeps only the
root-reachable counting state needed to find sizes, whereas sampling needs
complete novel histograms and suffix caches. Match enumeration is shared
between the two passes.

See [incremental_probe.md](incremental_probe.md) for the recurrence and
correctness argument and [layered_counting.md](layered_counting.md) for the
generic layered counter.
