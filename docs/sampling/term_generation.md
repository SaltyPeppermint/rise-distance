# Random term generation

Seed terms are sampled directly from a language grammar, before an e-graph
exists. This differs from [novel-term sampling](novel_sampling.md), which draws
terms from an e-graph.

The implementation is in
[`src/generator.rs`](../../src/generator.rs), with grammars in
[`src/langs/math/generate.rs`](../../src/langs/math/generate.rs) and
[`src/langs/prop/generate.rs`](../../src/langs/prop/generate.rs).

## Guarantees

For a requested size `n`, `SizeUniformSampler` always returns a term with
exactly `n` AST nodes. For languages without binders, every term of that size is
equally likely.

Math has binder operators (`Diff` and `Integral`). Their bodies are sampled
uniformly, then the bound variable is chosen uniformly from the body's free
variables. This guarantees valid scoping, but is not uniform over all
well-scoped `(body, variable)` pairs.

## Counting and sampling

The sampler first counts terms at each size:

```text
counts[1] = |leaves|
counts[n] = |unary|  * counts[n-1]
          + |binary| * sum counts[k] * counts[n-1-k]
          + |binder| * counts[n-2]
```

It then builds a term top-down:

1. Choose a root shape, weighted by its contribution to `counts[n]`.
2. Choose an operator uniformly from that shape's pool.
3. Split the remaining size among the children and recurse.

Binary splits are weighted by the number of child pairs they admit. This makes
all binder-free terms of the requested size equally likely. Counts use `f64`
because only their ratios are needed.

For a binder, the sampler generates a body of size `n - 2` and chooses one of
its free variables. Bodies without a free variable are redrawn.

## Filtering in `generate`

The [`generate` binary](../../src/bin/generate.rs) further filters samples:

- Candidates that saturate before hitting an eqsat resource limit are rejected.
- Duplicate terms within a size bucket are rejected.

`--retry-limit` bounds the draws spent finding each distinct, validated term.
The saved corpus is therefore conditioned on validation and uniqueness; it is
not a uniform sample of the full grammar.
