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
    ///
    /// If `binder` is non-empty, every entry must occur exactly once in
    /// `ops[0]`. Exact binder counting is exponential in this pool's length.
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
    /// Sum of the free-variable counts of all terms at each size.
    fv_totals: Vec<f64>,
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
        let (counts, comps, fv_totals) = count_terms(&grammar, target);

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
            fv_totals,
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
                let body = self.gen_body_weighted_by_free_vars(rng, n - 2);
                let free = L::free_var_indices(&self.grammar, &body);
                let var_idx = *free
                    .choose(rng)
                    .expect("weighted body generation guarantees a free variable");
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

    /// Generate a body with probability proportional to its free-variable count.
    fn gen_body_weighted_by_free_vars<R: Rng>(&self, rng: &mut R, n: usize) -> RecExpr<L> {
        let var_count = self.grammar.vars.len();
        debug_assert!(self.fv_totals[n] > 0.0);
        loop {
            let body = self.gen_sized(rng, n);
            let free_count = L::free_var_indices(&self.grammar, &body).len();
            if rng.gen_range(0..var_count) < free_count {
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
            Shape::Binder => labels * self.fv_totals[n - 2],
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
/// comps[k][m] = sum_{j=1}^{m} counts[j] * comps[k-1][m-j]
/// counts[n]   = sum_k |ops[k]| * comps[k][n-1]
///             + |binder| * fv_totals[n-2]
/// ```
///
/// Sizes are strictly increasing in `n`, so `comps[k][m]` for `m < n` is already
/// final by the time `counts[n]` reads it.
///
/// Counts are stratified by the exact free-variable subset. This is required
/// because ordinary operators take the union of their children's free
/// variables, while a binder removes one.
/// Counts use `f64` because sampling needs ratios, not exact bigints.
#[expect(clippy::cast_precision_loss)]
fn count_terms<L: Samplable>(
    grammar: &Grammar<L>,
    limit: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let max_arity = grammar.max_arity();
    // Free-variable state is irrelevant when no binder can inspect it.
    let var_count = if grammar.binder.is_empty() {
        0
    } else {
        grammar.vars.len()
    };
    assert!(
        var_count < usize::BITS as usize,
        "too many binder variables for exact subset counting"
    );
    let subset_count = 1usize << var_count;
    let n_binder = grammar.binder.len() as f64;

    let mut by_fv = vec![vec![0.0; subset_count]; limit + 1];
    let mut tuple_by_fv = vec![vec![vec![0.0; subset_count]; limit + 1]; max_arity + 1];
    tuple_by_fv[0][0][0] = 1.0;

    for n in 1..=limit {
        for k in 1..=max_arity {
            for j in 1..n {
                for left_mask in 0..subset_count {
                    let left_count = by_fv[j][left_mask];
                    if left_count == 0.0 {
                        continue;
                    }
                    for right_mask in 0..subset_count {
                        let right_count = tuple_by_fv[k - 1][n - 1 - j][right_mask];
                        tuple_by_fv[k][n - 1][left_mask | right_mask] += left_count * right_count;
                    }
                }
            }
        }

        for (k, tuples) in tuple_by_fv.iter().enumerate().take(max_arity + 1) {
            let labels = grammar.pool(k).len() as f64;
            for mask in 0..subset_count {
                by_fv[n][mask] += labels * tuples[n - 1][mask];
            }
        }

        if n >= 3 {
            for body_mask in 0..subset_count {
                let body_count = by_fv[n - 2][body_mask];
                for var_idx in 0..var_count {
                    let bit = 1usize << var_idx;
                    if body_mask & bit != 0 {
                        by_fv[n][body_mask & !bit] += n_binder * body_count;
                    }
                }
            }
        }

        if n == 1 {
            // Ordinary leaves were all placed in mask 0 above. Move variable
            // labels to their actual singleton masks.
            for (var_idx, var) in grammar.vars.iter().take(var_count).enumerate() {
                let occurrences = grammar.pool(0).iter().filter(|leaf| *leaf == var).count();
                assert_eq!(
                    occurrences, 1,
                    "each binder variable must occur exactly once in the leaf pool"
                );
                by_fv[1][0] -= 1.0;
                by_fv[1][1usize << var_idx] += 1.0;
            }
        }
    }

    // Complete the documented tuple table at the boundary.
    for k in 1..=max_arity {
        for j in 1..=limit {
            for left_mask in 0..subset_count {
                let left_count = by_fv[j][left_mask];
                if left_count == 0.0 {
                    continue;
                }
                for right_mask in 0..subset_count {
                    let right_count = tuple_by_fv[k - 1][limit - j][right_mask];
                    tuple_by_fv[k][limit][left_mask | right_mask] += left_count * right_count;
                }
            }
        }
    }

    let counts = by_fv
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let fv_totals = by_fv
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(mask, count)| f64::from(mask.count_ones()) * count)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let comps = tuple_by_fv
        .into_iter()
        .map(|by_size| {
            by_size
                .into_iter()
                .map(|row| row.into_iter().sum::<f64>())
                .collect()
        })
        .collect();

    (counts, comps, fv_totals)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
mod tests {
    use egg::{Id, RecExpr, define_language};
    use hashbrown::{HashMap, HashSet};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::langs::math::Math;
    use crate::langs::prop::Prop;
    use crate::utils::id0;

    #[test]
    fn one_tuple_counts_equal_term_counts_through_limit() {
        for limit in [1, 2, 3, 7, 12] {
            for (counts, comps) in [
                {
                    let (counts, comps, _) = count_terms(&Prop::grammar(None), limit);
                    (counts, comps)
                },
                {
                    let (counts, comps, _) = count_terms(&Math::grammar(None), limit);
                    (counts, comps)
                },
            ] {
                for m in 0..=limit {
                    assert_eq!(comps[1][m], counts[m], "limit={limit}, m={m}");
                }
            }
        }
    }

    #[test]
    fn binder_counts_include_each_valid_variable_choice() {
        let expected = [
            5.0,
            20.0,
            209.0,
            1_852.0,
            20_102.0,
            217_224.0,
            2_493_545.0,
            29_093_596.0,
            348_297_218.0,
            4_231_264_728.0,
            52_161_542_746.0,
            650_147_332_760.0,
            8_183_703_739_724.0,
            103_860_232_893_712.0,
            1_327_641_834_673_817.0,
            17_077_796_005_488_284.0,
        ];
        let (counts, _, _) = count_terms(&Math::grammar(None), expected.len());
        assert_eq!(&counts[1..], expected);
    }

    define_language! {
        #[derive(Deserialize, Serialize)]
        enum Tiny {
            "x" = X,
            "y" = Y,
            "c" = C,
            "+" = Add([Id; 2]),
            "b" = Bind([Id; 2]),
        }
    }

    impl Samplable for Tiny {
        fn grammar(_leaf_symbols: Option<Vec<Self>>) -> Grammar<Self> {
            Grammar::new(
                vec![
                    vec![Tiny::X, Tiny::Y, Tiny::C],
                    Vec::new(),
                    vec![Tiny::Add([id0(), id0()])],
                ],
                vec![Tiny::X, Tiny::Y],
                vec![Tiny::Bind([id0(), id0()])],
            )
        }

        fn free_var_indices(_grammar: &Grammar<Self>, expr: &RecExpr<Self>) -> Vec<usize> {
            fn free(expr: &RecExpr<Tiny>, id: Id) -> usize {
                match expr[id] {
                    Tiny::X => 1,
                    Tiny::Y => 2,
                    Tiny::C => 0,
                    Tiny::Add(children) => free(expr, children[0]) | free(expr, children[1]),
                    Tiny::Bind(children) => {
                        let bound = match expr[children[1]] {
                            Tiny::X => 1,
                            Tiny::Y => 2,
                            _ => unreachable!("binder variable is always x or y"),
                        };
                        free(expr, children[0]) & !bound
                    }
                }
            }

            let mask = free(expr, expr.root());
            (0..2).filter(|idx| mask & (1 << idx) != 0).collect()
        }
    }

    #[test]
    fn binder_free_grammar_ignores_unused_variable_metadata() {
        let grammar = Grammar::new(
            vec![
                vec![Tiny::X, Tiny::Y, Tiny::C],
                Vec::new(),
                vec![Tiny::Add([id0(), id0()])],
            ],
            vec![Tiny::X, Tiny::Y],
            Vec::new(),
        );
        let (counts, _, fv_totals) = count_terms(&grammar, 3);
        assert_eq!(counts, [0.0, 3.0, 0.0, 9.0]);
        assert_eq!(fv_totals, [0.0; 4]);
    }

    fn enumerate_tiny(n: usize) -> Vec<RecExpr<Tiny>> {
        if n == 1 {
            return [Tiny::X, Tiny::Y, Tiny::C]
                .into_iter()
                .map(|leaf| stack_children(&[], leaf))
                .collect();
        }

        let grammar = Tiny::grammar(None);
        let mut result = Vec::new();
        for left_size in 1..n - 1 {
            let right_size = n - 1 - left_size;
            for left in enumerate_tiny(left_size) {
                for right in enumerate_tiny(right_size) {
                    result.push(stack_children(
                        &[left.clone(), right],
                        Tiny::Add([id0(), id0()]),
                    ));
                }
            }
        }
        if n >= 3 {
            for body in enumerate_tiny(n - 2) {
                for var_idx in Tiny::free_var_indices(&grammar, &body) {
                    let var = stack_children(&[], grammar.vars[var_idx].clone());
                    result.push(stack_children(
                        &[body.clone(), var],
                        Tiny::Bind([id0(), id0()]),
                    ));
                }
            }
        }
        result
    }

    #[test]
    fn binder_sampling_is_uniform_over_terms() {
        const TARGET: usize = 5;
        const DRAWS: usize = 200_000;

        let expected = enumerate_tiny(TARGET)
            .into_iter()
            .map(|expr| expr.to_string())
            .collect::<HashSet<_>>();
        let sampler = SizeUniformSampler::<Tiny>::new(TARGET, None);
        assert_eq!(sampler.space_size(), expected.len() as f64);

        let mut observed = HashMap::<String, usize>::new();
        let mut rng = ChaCha8Rng::seed_from_u64(0x5eed);
        for expr in sampler.sample_many(&mut rng, DRAWS) {
            *observed.entry(expr.to_string()).or_default() += 1;
        }
        assert_eq!(
            observed.keys().collect::<HashSet<_>>(),
            expected.iter().collect::<HashSet<_>>()
        );

        let uniform = 1.0 / expected.len() as f64;
        let tv = expected
            .iter()
            .map(|term| {
                let empirical = observed.get(term).copied().unwrap_or(0) as f64 / DRAWS as f64;
                (empirical - uniform).abs()
            })
            .sum::<f64>()
            / 2.0;
        assert!(tv < 0.015, "total-variation distance {tv} is too large");
    }
}
