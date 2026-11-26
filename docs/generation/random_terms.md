# Random term generation

Seed terms are sampled directly from a language grammar **before any e-graph
exists**. Given a target AST node count `n`, `SizeUniformSampler` draws a
well-scoped term of exact size `n` uniformly at random—every valid grammar term
of size `n` has the exact same probability, even when the grammar includes
binder operators.

> **Scope:** This document covers direct grammar sampling for seed terms. For
> extracting novel terms from an e-graph, see
> [exact novel-candidate construction](../candidates/novel_candidates.md).

## Where the code lives

- [src/generator.rs](../../src/generator.rs) — `Grammar`, `Samplable`, and
  `SizeUniformSampler`.
- [src/langs/math/generate.rs](../../src/langs/math/generate.rs) — Math grammar
  (includes binders `Diff` and `Integral`).
- [src/langs/prop/generate.rs](../../src/langs/prop/generate.rs) — Propositional
  logic grammar (binder-free).
- [src/bin/start.rs](../../src/bin/start.rs) — seed generation and eqsat
  filtering.

---

## 1. Why binders need special care

A binder `(op body var)` uses 2 nodes for `op` and `var` plus `n - 2` nodes for
`body`, and is **well-scoped** only when `var` occurs free in `body`. Binding
removes `var` from the resulting term's free-variable set (nested binders follow
the same rule).

If you sampled `body` uniformly and then picked one of its free variables, you
would **under-sample** bodies with more free variables: a body with 2 free
variables admits 2 valid `(body, var)` pairs, while a body with 1 free variable
admits only 1.

To make every valid `(body, var)` pair equally likely, `SizeUniformSampler`
instead samples `body` with probability **proportional to `|free(body)|`**, and
then picks `var` uniformly from `free(body)`.

---

## 2. Stage 1: Bottom-up counting

Before drawing any terms, `SizeUniformSampler::new(target, ...)` precomputes
how many terms exist at every size up to `target` using `f64` counts (only
ratios are needed for weighted sampling):

- **`C[n, S]`** — number of size-`n` terms whose exact set of free variables is
  `S`.
- **`counts[n]`** — total number of size-`n` terms (`sum_S C[n, S]`).
- **`fv_totals[n]`** — sum of `|free(t)|` over all size-`n` terms
  (`sum_S |S| * C[n, S]`), which is the exact number of valid `(body, var)`
  pairs for a body of size `n`.
- **`comps[k][m]`** — number of ordered `k`-tuples of terms whose sizes sum to
  `m`:
  ```text
  comps[0][0] = 1
  comps[0][m] = 0                                      (m != 0)
  comps[k][m] = sum_(j=1..m) counts[j] * comps[k-1][m-j]
  ```

### Recurrences

An ordinary arity-`k` operator unions its children's free-variable sets, while
a binder removes one free variable `v` from its body's set `T`:

```text
C[n, S] =
    (ordinary arity-k terms whose children's free-variable sets union to S)
  + |binder| * sum over (T, v) with v in T and T \ {v} = S of C[n - 2, T]

counts[n] =
    sum_k |ops[k]| * comps[k][n - 1]
  + |binder| * fv_totals[n - 2]
```

### Why exact subsets `S` are tracked (and when they aren't)

Tracking only the *count* `|S|` of free variables is not enough, because two
children's free-variable sets can overlap when unioned.

With `V = |vars|`, target size `N`, and maximum arity `K`, the subset DP uses:
- **Memory:** `O(K N 2^V)`
- **Time:** `O(K N^2 4^V + |binder| N V 2^V)`

**Binder-free fast path:** Grammars with no binders track only a single empty
subset (`2^0 = 1`), regardless of how many variable-like leaves they have.

**Grammar invariants:** When `Grammar::binder` is non-empty, every entry in
`Grammar::vars` must appear **exactly once** in the leaf pool `ops[0]`. Because
exact subset counting is exponential in `|vars|`, it is designed for small
fixed variable pools (such as Math's `x` and `y`).

---

## 3. Stage 2: Top-down sampling

To draw a term of size `n`, `gen_sized(rng, n)` works recursively:

1. **Pick a root shape** (`Arity(k)` or `Binder`), weighted by its exact
   contribution to `counts[n]`.
2. **Pick an operator** uniformly from that shape's pool (`ops[k]` or `binder`).
3. **Fill the children:**
   - **Ordinary `Arity(k)`:** split the remaining `n - 1` nodes across the `k`
     children (each split weighted by `counts[j] * comps[remaining - 1][left - j]`)
     and recurse on each child.
   - **Binder:** sample a `(body, var)` pair of body size `m = n - 2` by
     rejection:
     1. Draw a candidate `body` uniformly at size `m` via `gen_sized(rng, m)`.
     2. Accept `body` with probability `|free(body)| / |vars|`; otherwise retry.
     3. Pick `var` uniformly from `free(body)`.

### Why binder sampling is exact and terminates

Multiplying the three probabilities shows that `|free(body)|` cancels out:

```text
Pr(propose body) * Pr(accept body) * Pr(choose var)
    = (1 / counts[m]) * (|free(body)| / |vars|) * (1 / |free(body)|)
    = 1 / (counts[m] * |vars|)
```

Every valid `(body, var)` pair has the exact same probability, so conditioning
on acceptance is **strictly uniform**.

Each proposal succeeds with probability `fv_totals[m] / (|vars| * counts[m])`.
Because `Shape::Binder` only receives positive weight at size `n` when
`fv_totals[n - 2] > 0`, the rejection loop is never entered unless at least one
valid body exists.

---

## 4. Filtering in `start`

`SizeUniformSampler` draws independently with replacement from the full
grammar. When building a benchmark corpus, two downstream filters narrow the
saved terms:

- The [`start` binary](../../src/bin/start.rs) rejects candidates that saturate
  before hitting an eqsat resource limit, bounded by `--retry-limit` draws per
  term.
- [`generate_problems.py`](../../scripts/generate_problems.py) drops duplicate
  terms within each size bucket.

The saved corpus is therefore conditioned on non-saturation and uniqueness—it
is not a uniform sample of the unfiltered grammar.
