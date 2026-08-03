use egg::RecExpr;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::MyLanguage;
use crate::utils::stack_children;

/// How an operator's children are filled when building a term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// A leaf with size 1.
    Leaf,
    /// One term child.
    Unary,
    /// Two term children.
    Binary,
    /// A body and a variable that occurs free in it.
    Binder,
}

impl Shape {
    /// Smallest term this shape can produce.
    const fn min_size(self) -> usize {
        match self {
            Shape::Leaf => 1,
            Shape::Unary => 2,
            Shape::Binary | Shape::Binder => 3,
        }
    }
}

/// Operator pools grouped by shape.
pub struct Grammar<L> {
    /// Variables and constants.
    pub leaves: Vec<L>,
    /// Leaves usable as bound variables.
    pub vars: Vec<L>,
    pub unary: Vec<L>,
    pub binary: Vec<L>,
    /// Operators whose children are a body and bound variable.
    pub binder: Vec<L>,
}

impl<L> Grammar<L> {
    /// Number of distinct labels available for a given shape.
    fn label_count(&self, shape: Shape) -> usize {
        match shape {
            Shape::Leaf => self.leaves.len(),
            Shape::Unary => self.unary.len(),
            Shape::Binary => self.binary.len(),
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
        let counts = count_terms(&grammar, target);

        assert!(
            target >= 1 && counts[target] > 0.0,
            "grammar admits no term of size {target}"
        );

        Self {
            grammar,
            counts,
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

        let shapes = [Shape::Leaf, Shape::Unary, Shape::Binary, Shape::Binder];
        let shape = *shapes
            .choose_weighted(rng, |shape| self.shape_count(*shape, n))
            .expect("counts[n] > 0 guarantees at least one shape is possible");

        match shape {
            Shape::Leaf => {
                let label = pick(rng, &self.grammar.leaves);
                stack_children(&[], label)
            }
            Shape::Unary => {
                let op = pick(rng, &self.grammar.unary);
                let child = self.gen_sized(rng, n - 1);
                stack_children(&[child], op)
            }
            Shape::Binary => {
                let op = pick(rng, &self.grammar.binary);
                let left_size = self.pick_split(rng, n - 1);
                let left = self.gen_sized(rng, left_size);
                let right = self.gen_sized(rng, n - 1 - left_size);
                stack_children(&[left, right], op)
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
            Shape::Leaf => {
                if n == 1 {
                    labels
                } else {
                    0.0
                }
            }
            Shape::Unary => labels * self.counts[n - 1],
            Shape::Binary => labels * self.split_total(n - 1),
            // Count bodies, then choose among each body's free variables.
            Shape::Binder => labels * self.counts[n - 2],
        }
    }

    /// Number of ordered child pairs with total size `rest`.
    fn split_total(&self, rest: usize) -> f64 {
        (1..rest)
            .map(|k| self.counts[k] * self.counts[rest - k])
            .sum()
    }

    /// Pick a left-child size, weighted by the number of resulting term pairs.
    fn pick_split<R: Rng>(&self, rng: &mut R, rest: usize) -> usize {
        let sizes = (1..rest).collect::<Vec<_>>();
        *sizes
            .choose_weighted(rng, |k| self.counts[*k] * self.counts[rest - *k])
            .expect("the shape was chosen with non-zero weight")
    }
}

/// Clone a uniformly chosen pool entry.
fn pick<R: Rng, T: Clone>(rng: &mut R, pool: &[T]) -> T {
    pool.choose(rng)
        .expect("shape was chosen with non-zero weight, so its pool is non-empty")
        .clone()
}

/// Count terms of each exact size through `limit`.
///
/// ```text
/// counts[1] = |leaves|
/// counts[n] = |unary|  * counts[n-1]
///           + |binary| * sum_{k=1}^{n-2} counts[k] * counts[n-1-k]
///           + |binder| * counts[n-2]
/// ```
///
/// Binder counts represent bodies; their free-variable choices are made during
/// sampling. Counts use `f64` because sampling needs ratios, not exact bigints.
#[expect(clippy::cast_precision_loss)]
fn count_terms<L>(grammar: &Grammar<L>, limit: usize) -> Vec<f64> {
    let n_leaf = grammar.leaves.len() as f64;
    let n_unary = grammar.unary.len() as f64;
    let n_binary = grammar.binary.len() as f64;
    let n_binder = grammar.binder.len() as f64;

    let mut counts = vec![0.0; limit + 1];
    if limit >= 1 {
        counts[1] = n_leaf;
    }
    for n in 2..=limit {
        let unary = n_unary * counts[n - 1];
        let binary = n_binary
            * (1..n - 1)
                .map(|k| counts[k] * counts[n - 1 - k])
                .sum::<f64>();
        let binder = if n >= 3 {
            n_binder * counts[n - 2]
        } else {
            0.0
        };
        counts[n] = unary + binary + binder;
    }
    counts
}
