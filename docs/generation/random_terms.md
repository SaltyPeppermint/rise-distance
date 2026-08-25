# Random term generation

Seed terms are sampled directly from a language grammar, before an e-graph
exists. This differs from
[exact novel-candidate construction](../candidates/exact_novel_candidates.md),
which draws terms from an e-graph.

The implementation is in
[`src/generator.rs`](../../src/generator.rs), with grammars in
[`src/langs/math/generate.rs`](../../src/langs/math/generate.rs) and
[`src/langs/prop/generate.rs`](../../src/langs/prop/generate.rs).

## Guarantees

For a requested size `n`, `SizeUniformSampler` always returns a term with
exactly `n` AST nodes. Every grammar term of that size is equally likely,
including terms containing binders.

Math has binder operators (`Diff` and `Integral`). Binder bodies are sampled
with probability proportional to their number of free variables, then the
bound variable is chosen uniformly from that set. This is uniform over all
well-scoped `(body, variable)` pairs.

Here a binder `(op body var)` is well-scoped when `var` occurs free in `body`.
Binding removes that variable from the resulting term's free-variable set.
Nested binders are counted and sampled under the same rule.

## Counting and sampling

Let:

```text
C[n,S]       = number of size-n terms with exact free-variable set S
counts[n]    = sum_S C[n,S]
fv_total[n]  = sum_S |S| C[n,S]
```

For an ordinary arity-`k` operator, the result's free-variable set is the union
of its children's sets. For a binder, every free variable in the body provides
one valid bound-variable choice:

```text
C[n,S] =
    ordinary terms whose child free-set union is S
  + |binder| * sum C[n-2,T]
      over T and v where v in T and T \ {v} = S

counts[n] =
    sum_k |ops[k]| * comps[k][n-1]
  + |binder| * fv_total[n-2]
```

`comps[k][m]` counts ordered `k`-tuples of terms with total size `m`:

```text
comps[0][0] = 1
comps[0][m] = 0                       when m != 0
comps[k][m] = sum_(j=1..m) counts[j] * comps[k-1][m-j]
```

The implementation retains the exact subset state because a free-variable
cardinality alone cannot determine the size of a union: two child sets may
overlap. With `V = |vars|`, target `N`, and maximum arity `K`, the current
subset DP uses `O(K N 2^V)` memory and
`O(K N^2 4^V + |binder| N V 2^V)` time. Binder-free grammars use only the
single empty subset, regardless of whether their language has variable-like
leaves.

It then builds a term top-down:

1. Choose a root shape, weighted by its contribution to `counts[n]`.
2. Choose an operator uniformly from that shape's pool.
3. Split the remaining size among the children and recurse.

Binary splits are weighted by the number of child pairs they admit. Counts use
`f64` because only their ratios are needed.

For a binder, the sampler accepts a generated body with probability
`|free(body)| / |vars|`, then chooses one of its free variables. The two steps
cancel:

```text
Pr(propose body) * Pr(accept body) * Pr(choose variable)
    = (1 / counts[m]) * (|free(body)| / |vars|) * (1 / |free(body)|)
    = 1 / (counts[m] * |vars|)
```

Conditioning on acceptance therefore makes every valid body-variable pair
equiprobable. The acceptance probability at body size `m` is
`fv_total[m] / (|vars| * counts[m])`. A binder shape has positive weight only
when this value is positive, so the rejection loop cannot be entered for a
size having no valid body.

Every entry in `Grammar::vars` must occur exactly once in the leaf pool when
the grammar has binders. The exact subset representation is exponential in the
number of such variables and is intended for small fixed variable pools such
as Math's `x` and `y`.

## Relationship to e-graph candidate construction

`SizeUniformSampler` samples seed terms directly from a language grammar. It
does not inspect an e-graph.

The candidate documents describe a separate pipeline:

- [Exact novel-candidate construction](../candidates/exact_novel_candidates.md)
  counts terms extractable from a current e-graph but absent from a previous
  e-graph.

## Filtering in `generate`

The [`generate` binary](../../src/bin/generate.rs) further filters samples:

- Candidates that saturate before hitting an eqsat resource limit are rejected.
- Duplicate terms within a size bucket are rejected.

`--retry-limit` bounds the draws spent finding each distinct, validated term.
The saved corpus is therefore conditioned on validation and uniqueness; it is
not a uniform sample of the full grammar.
