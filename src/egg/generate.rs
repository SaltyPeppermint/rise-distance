#![expect(clippy::similar_names)]
use std::collections::HashSet;

use egg::StopReason;
use rand::Rng;

use crate::Label;
use crate::tree::{TreeShaped, TypedTree};

/// Description of a language for the Boltzmann sampler.
///
/// Operators are partitioned by arity and binding behavior:
///   - `leaves`: terminal symbols (the full pool drawn at leaf positions).
///   - `variables`: subset of `leaves` that may appear in a binder's bound slot.
///   - `unary_ops`: arity-1 operators.
///   - `normal_binary_ops`: arity-2 operators where both children are expressions.
///   - `binder_ops`: arity-2 operators where child[0] is an expression and
///     child[1] is forced to be a variable leaf.
pub struct LanguageSpec<L: Label + Copy> {
    pub leaves: Vec<L>,
    pub variables: Vec<L>,
    pub unary_ops: Vec<L>,
    pub normal_binary_ops: Vec<L>,
    pub binder_ops: Vec<L>,
}

/// Boltzmann sampler for random terms of a target size.
///
/// The grammar distinguishes three kinds of binary operators:
///   - Normal binaries: both children drawn from T
///   - Binder binaries: child[0] drawn from T, child[1] forced to a variable leaf
///
/// The corrected generating function is:
///   `T(x) = n_l*x + n_u*x*T + n_b_normal*x*T^2 + n_b_binder*x*T*(n_v*x)`
///
/// where `n_l = |leaves|`, `n_u = |unary_ops|`, `n_b_normal = |normal_binary_ops|`,
/// `n_b_binder` = |`binder_ops`|, `n_v` = |`variables`|.
///
/// Rearranging as a quadratic in T:
///   `n_b_normal*x*T^2 + (n_u*x + n_b_binder*n_v*x^2 - 1)*T + n_l*x = 0`
///
/// The singularity `rho` is found numerically (`discriminant = 0`).
/// We binary-search for `x < rho` giving the target expected size, then use rejection sampling.
///
/// Additionally, generated trees are filtered so that the bound variable in each binder
/// node actually appears free in child[0].
pub struct BoltzmannSampler<L: Label + Copy> {
    p_leaf: f64,
    p_unary: f64,
    p_binder: f64,
    spec: LanguageSpec<L>,
    target: usize,
    tolerance: usize,
    max_depth: usize,
}

#[expect(clippy::cast_precision_loss)]
impl<L: Label + Copy> BoltzmannSampler<L> {
    /// Create a sampler targeting terms of the given expected size.
    #[must_use]
    pub fn new(target: usize, tolerance: usize, spec: LanguageSpec<L>) -> Self {
        let n_l = spec.leaves.len() as f64;
        let n_u = spec.unary_ops.len() as f64;
        let n_bn = spec.normal_binary_ops.len() as f64;
        let n_bb = spec.binder_ops.len() as f64;
        let n_v = spec.variables.len() as f64;

        let rho = find_singularity(n_l, n_u, n_bn, n_bb, n_v);
        let x = find_tuning_param(target as f64, rho, n_l, n_u, n_bn, n_bb, n_v);
        let t = eval_t(x, n_l, n_u, n_bn, n_bb, n_v);

        let p_leaf = n_l * x / t;
        let p_unary = n_u * x;
        let p_binder = n_bb * n_v * x * x;

        let max_depth = (target + tolerance) * 4;

        BoltzmannSampler {
            p_leaf,
            p_unary: p_leaf + p_unary,
            p_binder: p_leaf + p_unary + p_binder,
            spec,
            target,
            tolerance,
            max_depth,
        }
    }

    fn gen_node(&self, rng: &mut impl Rng, depth: usize) -> TypedTree<L> {
        let r = rng.r#gen::<f64>();
        if depth >= self.max_depth || r < self.p_leaf {
            let label = self.spec.leaves[rng.gen_range(0..self.spec.leaves.len())];
            TypedTree::leaf_untyped(label)
        } else if r < self.p_unary {
            let op = self.spec.unary_ops[rng.gen_range(0..self.spec.unary_ops.len())];
            let child = self.gen_node(rng, depth + 1);
            TypedTree::new_untyped(op, vec![child])
        } else if r < self.p_binder {
            let op = self.spec.binder_ops[rng.gen_range(0..self.spec.binder_ops.len())];
            let expr = self.gen_node(rng, depth + 1);
            let var_label = self.spec.variables[rng.gen_range(0..self.spec.variables.len())];
            let var = TypedTree::leaf_untyped(var_label);
            TypedTree::new_untyped(op, vec![expr, var])
        } else {
            let op =
                self.spec.normal_binary_ops[rng.gen_range(0..self.spec.normal_binary_ops.len())];
            let left = self.gen_node(rng, depth + 1);
            let right = self.gen_node(rng, depth + 1);
            TypedTree::new_untyped(op, vec![left, right])
        }
    }

    /// Generate a random term whose size is in `[target - tolerance, target + tolerance]`
    /// and where every binder's bound variable appears free in its expression child.
    /// Returns None if no valid tree is found within `100_000` attempts.
    pub fn sample<R: Rng, F: Fn(&TypedTree<L>) -> Option<StopReason>>(
        &self,
        rng: &mut R,
        filter_hook: &F,
    ) -> Option<(TypedTree<L>, StopReason, usize)> {
        let lo = self.target.saturating_sub(self.tolerance);
        let hi = self.target + self.tolerance;
        (0..100_000).find_map(|n| {
            let candidate = self.gen_node(rng, 0);
            if (lo..=hi).contains(&candidate.size_without_types())
                && binders_valid(&candidate, &self.spec)
                && let Some(reason) = filter_hook(&candidate)
            {
                return Some((candidate, reason, n));
            }
            None
        })
    }

    /// Generate `count` random terms within the size window.
    pub fn sample_many<R: Rng, F: Fn(&TypedTree<L>) -> Option<StopReason>>(
        &self,
        rng: &mut R,
        count: usize,
        filter_hook: &F,
    ) -> Vec<(TypedTree<L>, StopReason)> {
        let (trees, total_attempts, failed) =
            (0..count).map(|_| self.sample(rng, filter_hook)).fold(
                (Vec::with_capacity(count), 0, 0),
                |(mut trees, attempts, failed), result| match result {
                    Some((tree, reason, a)) => {
                        trees.push((tree, reason));
                        (trees, attempts + a, failed)
                    }
                    None => (trees, attempts, failed + 1),
                },
            );
        println!(
            "Took a total of {total_attempts} attempts for {} terms. {failed} failed!",
            trees.len()
        );
        trees
    }
}

/// Returns true if every binder node in the tree has its bound variable
/// appearing free somewhere in its expression child (child[0]).
pub fn binders_valid<L: Label + Copy>(tree: &TypedTree<L>, spec: &LanguageSpec<L>) -> bool {
    if spec.binder_ops.contains(tree.label()) {
        let children = tree.children();
        let bound = children[1].label();
        if !spec.variables.contains(bound) {
            return false;
        }
        let expr = &children[0];
        free_vars(expr, spec).contains(bound) && binders_valid(expr, spec)
    } else {
        tree.children().iter().all(|c| binders_valid(c, spec))
    }
}

/// Collect all variable leaves that appear free in the tree.
/// Variables bound by an enclosing binder are excluded from the free
/// set of that binder's expression child.
pub fn free_vars<L: Label + Copy>(tree: &TypedTree<L>, spec: &LanguageSpec<L>) -> HashSet<L> {
    if spec.binder_ops.contains(tree.label()) {
        let children = tree.children();
        let mut vars = free_vars(&children[0], spec);
        let bound = children[1].label();
        if spec.variables.contains(bound) {
            vars.remove(bound);
        }
        vars
    } else if spec.variables.contains(tree.label()) {
        let mut set = HashSet::new();
        set.insert(*tree.label());
        set
    } else {
        tree.children()
            .iter()
            .flat_map(|c| free_vars(c, spec))
            .collect()
    }
}

/// Evaluate `T(x)` for the corrected grammar. Takes the smaller root.
fn eval_t(x: f64, n_l: f64, n_u: f64, n_bn: f64, n_bb: f64, n_v: f64) -> f64 {
    let b = n_u * x + n_bb * n_v * x * x - 1.0;
    let disc = b * b - 4.0 * n_bn * n_l * x * x;
    (-b - disc.sqrt()) / (2.0 * n_bn * x)
}

fn discriminant(x: f64, n_l: f64, n_u: f64, n_bn: f64, n_bb: f64, n_v: f64) -> f64 {
    let b = n_u * x + n_bb * n_v * x * x - 1.0;
    b * b - 4.0 * n_bn * n_l * x * x
}

fn find_singularity(n_l: f64, n_u: f64, n_bn: f64, n_bb: f64, n_v: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = 1.0 / (n_u + 1.0);
    while discriminant(hi, n_l, n_u, n_bn, n_bb, n_v) > 0.0 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        if discriminant(mid, n_l, n_u, n_bn, n_bb, n_v) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

fn expected_size(x: f64, n_l: f64, n_u: f64, n_bn: f64, n_bb: f64, n_v: f64) -> f64 {
    let eps = x * 1e-8;
    let t1 = eval_t(x + eps, n_l, n_u, n_bn, n_bb, n_v);
    let t0 = eval_t(x - eps, n_l, n_u, n_bn, n_bb, n_v);
    let t = eval_t(x, n_l, n_u, n_bn, n_bb, n_v);
    let t_prime = (t1 - t0) / (2.0 * eps);
    x * t_prime / t
}

fn find_tuning_param(
    target: f64,
    rho: f64,
    n_l: f64,
    n_u: f64,
    n_bn: f64,
    n_bb: f64,
    n_v: f64,
) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = rho * (1.0 - 1e-10);
    for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        if expected_size(mid, n_l, n_u, n_bn, n_bb, n_v) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_singularity_positive_and_discriminant_near_zero() {
        let (n_l, n_u, n_bn, n_bb, n_v) = (5.0, 4.0, 5.0, 2.0, 2.0);
        let rho = find_singularity(n_l, n_u, n_bn, n_bb, n_v);
        assert!(rho > 0.0, "singularity should be positive");
        let disc = discriminant(rho, n_l, n_u, n_bn, n_bb, n_v);
        assert!(
            disc.abs() < 1e-10,
            "discriminant at rho should be ~0, got {disc}"
        );
    }

    #[test]
    fn find_tuning_param_expected_size_matches_target() {
        let (n_l, n_u, n_bn, n_bb, n_v) = (5.0, 4.0, 5.0, 2.0, 2.0);
        let rho = find_singularity(n_l, n_u, n_bn, n_bb, n_v);
        for target in [5.0, 10.0, 20.0, 50.0] {
            let x = find_tuning_param(target, rho, n_l, n_u, n_bn, n_bb, n_v);
            let actual = expected_size(x, n_l, n_u, n_bn, n_bb, n_v);
            assert!(
                (actual - target).abs() < 0.5,
                "target={target}, actual expected size={actual}"
            );
        }
    }

    #[test]
    fn expected_size_increases_with_x() {
        let n_l = 5.0f64;
        let n_u = 4.0f64;
        let n_bn = 5.0f64;
        let n_bb = 2.0f64;
        let n_v = 2.0f64;
        let rho = find_singularity(n_l, n_u, n_bn, n_bb, n_v);

        let e1 = expected_size(rho * 0.3, n_l, n_u, n_bn, n_bb, n_v);
        let e2 = expected_size(rho * 0.7, n_l, n_u, n_bn, n_bb, n_v);
        let e3 = expected_size(rho * 0.99, n_l, n_u, n_bn, n_bb, n_v);
        assert!(e1 < e2, "e1={e1} should be < e2={e2}");
        assert!(e2 < e3, "e2={e2} should be < e3={e3}");
    }
}
