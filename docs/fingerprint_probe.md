# Fingerprint probes for `backoff_precompute`

How `backoff_precompute` finds the smallest `max_size` with at least `sizes`
distinct novel-term sizes at the root — while running the expensive exact
counting analysis only once.

The relevant code lives in:

- [src/sampling/count/mod61.rs](../src/sampling/count/mod61.rs) — the `Mod61` fingerprint counter.
- [src/sampling/count/novel.rs](../src/sampling/count/novel.rs) — `probe_novel_root_sizes`, match-enumeration reuse.
- [src/sampling/mod.rs](../src/sampling/mod.rs) — the rewritten `backoff_precompute`.

---

## 1. The problem

`backoff_precompute` needs the smallest `max_size` such that the root e-class
has at least `sizes` distinct sizes with novel terms. The old loop found it by
running the *full* precompute at `start_size + i * retry_step` until one
attempt succeeded. Each attempt paid for four expensive things, all discarded
on failure:

1. the curr↔prev **match enumeration** (`enumerate_matches`),
2. the **`ExprCount` fixpoint** over every class, with `BigUint` counts,
3. the **suffix-convolution cache** over every node (only ever needed by the
   final package the sampler uses, never by the "how many novel sizes?"
   question),
4. the **joint-count fixpoint** (`compute_joint`), again with `BigUint`.

The dominant cost in 2–4 is bignum arithmetic: counts grow combinatorially
with term size, so every convolution cell is a multi-limb multiply plus heap
traffic.

## 2. The two observations

**The match enumeration is independent of `max_size`.** It is now computed
once per `backoff_precompute` call and shared by every probe and by the final
exact run (`NovelTermCount::with_matches`, `precompute_with_matches`).

**The search needs existence, not counts.** The retry loop only asks "is the
novel count at size `s` nonzero?". A plain boolean feasibility DP cannot
answer that, because novelty is defined by subtraction
(`novel = plain − Σ_pc joint`, and a size can hold both novel and old terms —
booleans don't subtract). But the *homomorphic image* of the counts works:
run the identical counting pipeline with all arithmetic done modulo the
Mersenne prime `p = 2^61 − 1` in a single `u64`. That is the `Mod61` type; it
satisfies the existing `Counter` trait, so `ExprCount` and `compute_joint`
run on it unchanged — orders of magnitude cheaper than `BigUint`, with no
suffix cache built at all.

## 3. The new control flow

```
matches = enumerate_matches(curr, prev)              // once
for bound in start_size, start_size + step, ...:     // cheap probes
    novel_sizes = probe_novel_root_sizes(bound, curr, root, &matches)
    if novel_sizes.len() >= sizes:
        max_size = novel_sizes[sizes - 1]            // k-th smallest novel size
        break
run the exact BigUint precompute ONCE at max_size    // verified; clamp as before
// fallback: the old exact backoff loop (see §5)
```

`probe_novel_root_sizes` runs the plain and joint DPs at `Mod61` and derives
novelty **at the root only** (`novel[s] = plain[root][s] − Σ_pc
joint[(root, pc)][s]`); per-class novel histograms, `derive_novel`, and the
suffix cache are all skipped.

## 4. Why it's correct

**One-sided error.** Reduction mod `p` is a ring homomorphism and the
counting DP is built from `+` and `×`, so the probe computes exactly
`(exact novel count) mod p`. A nonzero residue *proves* the exact count is
nonzero — the probe cannot invent a novel size. It can only *miss* one, when
an exact count is a nonzero multiple of `2^61 − 1` (~`2^-61` per size; these
are natural combinatorial counts, not adversarial ones).

**Misses are harmless or caught.**

- Missed size *below* the chosen one: the exact run finds an extra histogram
  key, and the existing `select_nth_unstable(sizes - 1)` clamp still picks
  the true `sizes`-th smallest.
- Chosen size itself wrong (astronomically unlikely): the exact run's
  `keys().len() >= sizes` check fails and the code falls back to the old
  exact backoff loop. The final package is therefore *always* built from
  exact `BigUint` counts; fingerprints only ever steer the search.

**Same final package, cheaper.** Previously the exact analysis ran at an
overshot bound (`start + i * step`) and was then clamped down. Now it runs at
the clamp target itself, so the single exact run is smaller than the old
successful attempt was.

## 5. Why `u64` cannot overflow

`Mod61` maintains the invariant *residue < 2^61 − 1* through every operation:

- **Addition:** two residues sum to `< 2^62`, comfortably inside `u64`; one
  conditional subtraction renormalizes.
- **Subtraction:** computed as `a + p − b ≤ 2^62` before reduction — no
  underflow, no wraparound.
- **Multiplication:** done in `u128` (product `< 2^122`), then reduced with
  the Mersenne identity `2^61 ≡ 1 (mod p)`: split into `hi·2^61 + lo` with
  both halves `≤ p`, sum, and reduce (the `== 2p` edge is covered
  explicitly).

There is no wrapping arithmetic anywhere — overflow is structurally
impossible, not merely unlikely. Plain wrapping `u64` (counting mod `2^64`)
was deliberately rejected: combinatorial counts are frequently divisible by
large powers of two, so false zeros would be realistic rather than
astronomical; divisibility by a fixed 61-bit prime is not a pattern these
counts exhibit.

The trait plumbing that only exists to satisfy `Counter` is inert:
`Ord` compares residues (meaningless on fingerprints, never consulted on the
probe path), division is honest field division via Fermat inverse (so it can
never silently corrupt anything), and the uniform-sampling impl samples
residues (never used).

## 6. Observable behavior changes

- The `usize` returned by `backoff_precompute` is now the `max_size` the
  exact analysis actually ran at (the `sizes`-th smallest novel size), not
  the overshot retry bound.
- Consequently `root_histogram()` — which feeds `frontier_histogram` and
  `log_root` in `bin/goal.rs` — ends at exactly `sizes` keys instead of
  extending past the clamp.
- Failed retries log `probe found N of K novel sizes (max_size=B), retrying`
  instead of the old exact-attempt message (which still appears in the
  fallback loop).
- The `Err` value on total failure is unchanged
  (`start_size + max_retries * retry_step`).

## 7. Verification

- `Mod61` arithmetic tested against `u128`/`BigUint` references, including
  `p − 1` / `p − 2` edge values, Fermat's little theorem for the inverse, and
  `From<u64::MAX>` reduction.
- Probe-vs-exact agreement tested over *every* class as root on three graph
  shapes, including a cyclic one where novel sizes are unbounded.
- End-to-end `backoff_precompute` test on a cyclic curr/prev pair whose novel
  sizes are provably 5, 7, 9, …: asking for 3 sizes yields `max_size = 9`,
  `min_size = 5`, keys `[5, 7, 9]` with the correct exact counts (built on a
  `#[cfg(test)]` constructor for `EqsatResult`).
- Full suite: 136/136 with `cargo nextest`; `cargo clippy --all-targets`
  clean.
