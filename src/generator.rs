use egg::RecExpr;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::MyLanguage;
use crate::utils::stack_children;

/// How an operator's children are filled when building a term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// `n` independently generated children.
    Arity(usize),
    /// A body and a variable that occurs free in it.
    Binder,
}

impl Shape {
    /// Smallest term this shape can produce.
    const fn min_size(self) -> usize {
        match self {
            Shape::Arity(k) => k + 1,
            Shape::Binder => 3,
        }
    }
}

/// Operator pools grouped by shape.
pub struct Grammar<L> {
    /// `ops[k]` holds the operators of arity `k`; `ops[0]` are the leaves.
    pub ops: Vec<Vec<L>>,
    /// Leaves usable as bound variables.
    pub vars: Vec<L>,
    /// Operators whose children are a body and bound variable.
    pub binder: Vec<L>,
}

impl<L> Grammar<L> {
    /// Build a grammar from its operator pools.
    ///
    /// `ops[k]` holds the operators of arity `k`, so `ops[0]` are the leaves.
    #[must_use]
    pub fn new(ops: Vec<Vec<L>>, vars: Vec<L>, binder: Vec<L>) -> Self {
        Self { ops, vars, binder }
    }

    /// Largest arity with at least one operator, or 0 if the grammar is empty.
    fn max_arity(&self) -> usize {
        self.ops
            .iter()
            .rposition(|pool| !pool.is_empty())
            .unwrap_or(0)
    }

    /// Operators of arity `k`, or an empty slice if the grammar has none.
    fn pool(&self, k: usize) -> &[L] {
        self.ops.get(k).map_or(&[], Vec::as_slice)
    }

    /// Number of distinct labels available for a given shape.
    fn label_count(&self, shape: Shape) -> usize {
        match shape {
            Shape::Arity(k) => self.pool(k).len(),
            Shape::Binder => self.binder.len(),
        }
    }
}

/// Languages that can be sampled by [`SizeUniformSampler`].
pub trait Samplable: MyLanguage + Sized {
    /// Return the grammar, optionally replacing its default leaves.
    fn grammar(leaf_symbols: Option<Vec<Self>>) -> Grammar<Self>;

    /// Indices in `grammar.vars` that occur free in `expr`.
    #[must_use]
    fn free_var_indices(_grammar: &Grammar<Self>, _expr: &RecExpr<Self>) -> Vec<usize> {
        Vec::new()
    }
}

/// Samples terms uniformly at an exact AST size.
///
/// Binder variables are chosen from the generated body's free variables.
pub struct SizeUniformSampler<L: MyLanguage> {
    grammar: Grammar<L>,
    /// Number of terms at each size; index 0 is unused.
    counts: Vec<f64>,
    /// `comps[k][m]` counts ordered `k`-tuples of terms with total size `m`.
    comps: Vec<Vec<f64>>,
    /// Every shape the grammar can build, in weighting order.
    shapes: Vec<Shape>,
    target: usize,
}

impl<L: Samplable> SizeUniformSampler<L> {
    /// Build a sampler for terms of size exactly `target`.
    ///
    /// # Panics
    ///
    /// Panics if the grammar admits no term of size `target` — for instance an
    /// even target in a grammar whose only operators are binary, or a target
    /// below the smallest constructible term.
    #[must_use]
    pub fn new(target: usize, leaf_symbols: Option<Vec<L>>) -> Self {
        let grammar = L::grammar(leaf_symbols);
        let (counts, comps) = count_terms(&grammar, target);

        assert!(
            target >= 1 && counts[target] > 0.0,
            "grammar admits no term of size {target}"
        );

        let shapes = (0..=grammar.max_arity())
            .map(Shape::Arity)
            .chain(std::iter::once(Shape::Binder))
            .collect();

        Self {
            grammar,
            counts,
            comps,
            shapes,
            target,
        }
    }

    /// The exact size of every term this sampler produces.
    #[must_use]
    pub const fn target(&self) -> usize {
        self.target
    }

    /// Approximate number of terms of size `target`.
    #[must_use]
    pub fn space_size(&self) -> f64 {
        self.counts[self.target]
    }

    /// Draw one term of size exactly [`target`](Self::target), uniformly at
    /// random among all terms of that size.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> RecExpr<L> {
        self.gen_sized(rng, self.target)
    }

    /// Draw `count` terms. Terms may repeat; de-duplication is the caller's job.
    pub fn sample_many<R: Rng>(&self, rng: &mut R, count: usize) -> Vec<RecExpr<L>> {
        (0..count).map(|_| self.sample(rng)).collect()
    }

    /// Generate a term of size `n`.
    fn gen_sized<R: Rng>(&self, rng: &mut R, n: usize) -> RecExpr<L> {
        debug_assert!(n >= 1, "cannot build a term of size 0");

        let shape = *self
            .shapes
            .choose_weighted(rng, |shape| self.shape_count(*shape, n))
            .expect("counts[n] > 0 guarantees at least one shape is possible");

        match shape {
            Shape::Arity(k) => {
                let op = pick(rng, self.grammar.pool(k));
                let children = self.gen_children(rng, k, n - 1);
                stack_children(&children, op)
            }
            Shape::Binder => {
                let op = pick(rng, &self.grammar.binder);
                let body = self.gen_body_with_free_var(rng, n - 2);
                let free = L::free_var_indices(&self.grammar, &body);
                let var_idx = *free
                    .choose(rng)
                    .expect("gen_body_with_free_var guarantees a free variable");
                let var = stack_children(&[], self.grammar.vars[var_idx].clone());
                stack_children(&[body, var], op)
            }
        }
    }

    /// Generate `k` children with total size `rest`, uniformly over all such tuples.
    fn gen_children<R: Rng>(&self, rng: &mut R, k: usize, rest: usize) -> Vec<RecExpr<L>> {
        let mut children = Vec::with_capacity(k);
        let mut left = rest;
        // The last child takes whatever remains, so only k-1 sizes are drawn.
        for remaining in (1..k).rev() {
            let size = self.pick_child_size(rng, remaining, left);
            children.push(self.gen_sized(rng, size));
            left -= size;
        }
        if k > 0 {
            children.push(self.gen_sized(rng, left));
        }
        children
    }

    /// Pick one child's size, weighted by how many tuples each choice leaves open.
    ///
    /// `remaining` is the number of children still to be sized after this one,
    /// and `left` is the total size they must share with it.
    fn pick_child_size<R: Rng>(&self, rng: &mut R, remaining: usize, left: usize) -> usize {
        let sizes = (1..=left - remaining).collect::<Vec<_>>();
        *sizes
            .choose_weighted(rng, |size| {
                self.counts[*size] * self.comps[remaining][left - *size]
            })
            .expect("the shape was chosen with non-zero weight")
    }

    /// Generate a binder body with at least one free variable.
    fn gen_body_with_free_var<R: Rng>(&self, rng: &mut R, n: usize) -> RecExpr<L> {
        if n == 1 {
            let label = pick(rng, &self.grammar.vars).clone();
            return stack_children(&[], label);
        }
        loop {
            let body = self.gen_sized(rng, n);
            if !L::free_var_indices(&self.grammar, &body).is_empty() {
                return body;
            }
        }
    }

    /// Number of size-`n` terms with this root shape.
    #[expect(clippy::cast_precision_loss)]
    fn shape_count(&self, shape: Shape, n: usize) -> f64 {
        let labels = self.grammar.label_count(shape) as f64;
        if labels == 0.0 || n < shape.min_size() {
            return 0.0;
        }
        match shape {
            Shape::Arity(k) => labels * self.comps[k][n - 1],
            // Count bodies, then choose among each body's free variables.
            Shape::Binder => labels * self.counts[n - 2],
        }
    }
}

/// Clone a uniformly chosen pool entry.
fn pick<R: Rng, T: Clone>(rng: &mut R, pool: &[T]) -> T {
    pool.choose(rng)
        .expect("shape was chosen with non-zero weight, so its pool is non-empty")
        .clone()
}

/// Count terms of each exact size through `limit`, and the child tuples that
/// build them.
///
/// ```text
/// comps[0][m] = 1 if m == 0 else 0
/// comps[k][m] = sum_{j=1}^{m-1} counts[j] * comps[k-1][m-j]
/// counts[n]   = sum_k |ops[k]| * comps[k][n-1]
///             + |binder| * counts[n-2]
/// ```
///
/// Sizes are strictly increasing in `n`, so `comps[k][m]` for `m < n` is already
/// final by the time `counts[n]` reads it.
///
/// Binder counts represent bodies; their free-variable choices are made during
/// sampling. Counts use `f64` because sampling needs ratios, not exact bigints.
#[expect(clippy::cast_precision_loss)]
fn count_terms<L>(grammar: &Grammar<L>, limit: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let max_arity = grammar.max_arity();
    let n_binder = grammar.binder.len() as f64;

    let mut counts = vec![0.0; limit + 1];
    // comps[0] is the empty tuple: the only way to total 0 is to take no children.
    let mut comps = vec![vec![0.0; limit + 1]; max_arity + 1];
    comps[0][0] = 1.0;

    for n in 1..=limit {
        // A k-tuple totalling n-1 uses only counts below n, already computed.
        for k in 1..=max_arity {
            comps[k][n - 1] = (1..n)
                .map(|j| counts[j] * comps[k - 1][n - 1 - j])
                .sum::<f64>();
        }

        let arities: f64 = (0..=max_arity)
            .map(|k| grammar.pool(k).len() as f64 * comps[k][n - 1])
            .sum();
        let binder = if n >= 3 {
            n_binder * counts[n - 2]
        } else {
            0.0
        };
        counts[n] = arities + binder;
    }

    // comps[k][limit] is never read by counts, but pick_child_size can ask for it.
    for k in 1..=max_arity {
        comps[k][limit] = (1..=limit)
            .map(|j| counts[j] * comps[k - 1][limit - j])
            .sum::<f64>();
    }

    (counts, comps)
}
