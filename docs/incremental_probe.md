# Size-incremental fingerprint probing

How `backoff_precompute` finds the `sizes`-th smallest novel term size at the
root in a **single** fingerprint run that stops at exactly that size — no
retry schedule, no wasted probes — by stratifying the joint counts by size
the same way the plain counts already are (see
[layered_counting.md](layered_counting.md)) and reading off the root's
novelty one size layer at a time.

The relevant code lives in:

- [src/sampling/count/layered.rs](../src/sampling/count/layered.rs) — the
  generic `LayeredDp` kernel shared by plain and joint counting;
  `count_terms` / `plain_dp`.
- [src/sampling/count/novel.rs](../src/sampling/count/novel.rs) — the layered
  `compute_joint`, `joint_children_of`, and the incremental
  `probe_novel_root_sizes`.
- [src/sampling/mod.rs](../src/sampling/mod.rs) — the simplified
  `backoff_precompute`.

---

## 1. The problem

[fingerprint_probe.md](fingerprint_probe.md) replaced the exact `BigUint`
retries with cheap `Mod61` probes, but kept the *shape* of the search: probe
at `start_size`, `start_size + retry_step`, … until a bound holds `sizes`
novel sizes, each probe recomputing everything from scratch. Two costs
remained:

1. **Every probe below the smallest novel size is fully wasted.** Where the
   frontier begins is unknown a priori. If the smallest novel term has size
   60 and the schedule starts at 3 with step 2, that is ~29 complete
   plain+joint pipeline runs that were doomed before they started — and the
   per-probe cost *grows* with the bound, so the useless runs are not even
   the cheap ones. This is where `backoff_precompute` spent most of its time.

2. **`compute_joint` was still a whole-histogram worklist fixpoint.** On a
   cyclic e-graph a `(curr, prev)` pair gains roughly one size per trip
   around the cycle, and each gain re-convolved entire histograms of every
   dependent pair — up to `bound` full recomputations per pair, exactly the
   pathology the plain counts already escaped in
   [layered_counting.md](layered_counting.md) §1. And this price was paid
   once per probe.

There is also overshoot at the end: the successful probe ran both DPs up to
its schedule bound even when the `sizes`-th smallest novel size lies below
it.

## 2. Observation 1: joint counts stratify by size too

The joint recurrence (novel.rs, "Phase 2") is

```
joint[(c, pc)](s) = sum over matches (n, m) of c with m.prev_class = pc of
                    |{ (s_1, …, s_k) : s_i >= 1, sum = s - 1 }|  weighted by
                    product over i of joint[(child_i, m.prev_children[i])](s_i)
```

The match's e-node costs 1, so every child pair is evaluated at a size
**strictly below `s`** — the identical argument
[layered_counting.md](layered_counting.md) §2 makes for the plain counts.
Indexed by `(pair, size)`, the dependency relation is a DAG with one layer
per size *even when the e-graph is cyclic*; the fixpoint iteration was an
artifact of computing whole histograms at once.

Structurally, the joint DP **is** the plain DP with different keys:

| plain counting            | joint counting                                      |
| ------------------------- | --------------------------------------------------- |
| key: e-class `c`          | key: pair `(c, pc)`                                 |
| choices: e-nodes of `c`   | choices: matches of `c`'s nodes with prev class `pc`|
| child keys: child classes | child keys: `zip(node children, m.prev_children)`   |
| zero-arity e-node → 1 at size 1 | zero-arity match → 1 at size 1                |

So the layer-by-layer kernel is now generic: `LayeredDp<K, C>` owns the
per-key node lists (`children_of: K -> Vec<Vec<K>>`), the per-key budgets,
the suffix tables, and the histograms, and exposes `step()` — complete the
next size layer. `count_terms` instantiates it with `K = Id` from the
e-graph; `compute_joint` instantiates it with `K = (Id, Id)` from the match
table (`joint_children_of`). One implementation, two callers — the old pair
fixpoint (`compute_pair_histogram`, `PairDeps`, the `UniqueQueue` worklist)
is deleted.

## 3. Observation 2: layer results are final ⇒ stop at the answer

After layer `s` completes, `plain[root](s)` and every `joint[(root, pc)](s)`
are **final** — later layers only add entries at larger sizes. Hence

```
novel(s) = plain[root](s) − sum over pc of joint[(root, pc)](s)
```

is final at the end of layer `s` as well. The probe therefore needs no bound
schedule at all: run both DPs *in lockstep, one layer at a time*, check the
root's novelty after each layer, and stop the moment `sizes` novel sizes have
been seen. The layer at which it stops is simultaneously

1. the proof that a `sizes`-th novel size exists,
2. its exact value, and
3. the `max_size` the single exact `BigUint` run needs.

```
matches = enumerate_matches(curr, prev)            // once, as before
cap     = start_size + max_retries * retry_step    // old schedule's last bound
plain   = LayeredDp over root-reachable classes    // Mod61, rooted budgets
joint   = LayeredDp over (curr, prev) pairs        // Mod61, budgets inherited (§4)
for size = 1, 2, …, cap:
    step both DPs by one layer
    novel = plain[root](size) − Σ_pc joint[(root, pc)](size)   // final now
    if novel ≠ 0: record size
    if `sizes` sizes recorded: stop                // size IS the k-th smallest
run the exact BigUint precompute ONCE at that size // verified, as before
// on probe failure or fingerprint mismatch: old exact backoff loop, unchanged
```

The smallest frontier term is found at exactly its own layer; sizes below it
are processed once (they would be part of any successful run anyway), and
nothing above the `sizes`-th novel size is ever computed.

## 4. Pair budgets: inheriting the root restriction

The plain side of the probe was already root-restricted
([layered_counting.md](layered_counting.md) §4); the joint side now is too.
Each pair inherits its curr class's budget:

```
budget(c, pc) = budget(c)        pairs whose class has no budget are skipped
```

**Why this is sound.** Every term counted by `joint[(c, pc)]` is in
particular a plain term of `c`. The budget recurrence
`budget(child) >= budget(parent) − 1 − Σ sibling plain-minima` is closed
under descending into *any* term of `c`: a joint subterm's siblings are plain
terms, so their sizes are lower-bounded by the same plain minima the budgets
were relaxed with. Inductively, every `(pair, size)` cell that a root query
(`joint[(root, pc)](s)`, `s <= cap`) transitively depends on lies within its
pair's budget. Pairs of unreachable classes can never be depended on and are
dropped entirely — a real saving, since `enumerate_matches` spans the whole
graph while the probe only asks about one root.

The **exact** path keeps uniform budgets (`max_size` for every pair):
`NovelTermCount` has no root parameter and its joint table is a public,
per-class API, so its output must not depend on any particular root. It
still gets the layering win (§5).

## 5. Why it's correct

**Same solution as the old fixpoint.** Old and new compute the same
recurrence; the recurrence has a unique solution because it is well-founded
by size (each cell depends only on strictly smaller sizes). The old worklist
iterated until it stopped changing; the new kernel evaluates each cell once,
in dependency order. Base case: a zero-arity match contributes the
empty-product suffix entry `{0: 1}`, read at children-total `0` — i.e. 1 at
size 1, exactly the old special case.

**Fingerprint semantics unchanged.** Everything is still computed modulo
`p = 2^61 − 1`, so `novel(s) ≡ (exact novel count at s) (mod p)`: a nonzero
residue *proves* a real novel size; a zero residue can only falsely hide one
with probability ~`2^-61` per size. One wrinkle: the kernel drops
zero-residue entries eagerly (as the plain layered DP already did), so the
root read-off treats missing entries as residue 0 — which they are. The
subtraction can even go "below zero" (plain residue 0, joint residues
nonzero); modular arithmetic keeps the congruence, and the one-sidedness
argument is untouched.

**Early exit is exact (up to fingerprints).** Because layer `s` finalizes
`novel(s)`, the recorded sizes are precisely the fingerprint-novel sizes in
ascending order; stopping after `sizes` of them returns the `sizes`-th
smallest. The exact run then re-verifies `>= sizes` histogram keys and falls
back to the exact backoff loop on a mismatch, exactly as before
([fingerprint_probe.md](fingerprint_probe.md) §4) — the final package is
always built from exact `BigUint` counts.

## 6. Old vs. new

|                                     | probe schedule (old)                       | incremental probe (new)                    |
| ----------------------------------- | ------------------------------------------ | ------------------------------------------ |
| probes below the smallest novel size| one full pipeline run per bound            | none — layers 1..m−1 run once, inside the one run |
| work above the `sizes`-th novel size| up to `retry_step` sizes of overshoot      | none — exit mid-run at the answer          |
| `compute_joint` on cycles           | whole-histogram re-convolutions to fixpoint| one convolution cell per `(pair, size)`    |
| joint pairs computed                | all pairs from `matches`                   | probe: root-reachable, budget-capped; exact: all |
| exact run's joint table             | worklist fixpoint                          | same layered kernel                        |
| no-frontier-at-all case             | `max_retries + 1` full probes, then `Err`  | one run to `cap`, then `Err`               |
| answer returned                     | `sizes`-th smallest novel size             | unchanged                                  |

## 7. Observable behavior changes

- `probe_novel_root_sizes` takes a `stop_after` argument and returns at most
  that many (ascending) novel sizes; tests that want the full spectrum pass
  `usize::MAX`.
- `start_size`, `max_retries` and `retry_step` no longer shape the probe;
  they only define the give-up cap `start_size + max_retries * retry_step`
  (unchanged as the `Err` value) and still drive the exact fallback loop.
- On probe failure the log gets a single
  `probe found N of K novel sizes (max_size=<cap>)` line instead of one line
  per retry. Fallback-loop messages are unchanged.
- `compute_joint` results are identical for exact counters; for `Mod61`,
  zero-*residue* entries are now dropped eagerly (the documented one-sided
  miss, same policy as the plain layered DP).

## 8. Verification

- All existing probe-vs-exact agreement tests (every class as root, three
  graph shapes including a cyclic one) run against the layered joint +
  incremental probe unchanged.
- The layered `compute_joint` feeds the exact `NovelTermCount` used by every
  novel-sampling test, so the exact-vs-sampler invariants
  (`SOMEHOW THE NOVEL SAMPLER PRODUCED A NON-NOVEL TERM` debug assert, joint
  ≤ plain in `derive_novel`) all exercise it.
- End-to-end `backoff_precompute` test on the cyclic pair with novel sizes
  5, 7, 9, …: asking for 3 sizes still yields `max_size = 9`,
  `min_size = 5`, keys `[5, 7, 9]` — and now stops layering at 9 instead of
  probing bounds 3, 5, 7, 9.
- Full suite: 146/146 with `cargo nextest`; `cargo clippy --all-targets`
  clean.
