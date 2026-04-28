#![expect(clippy::similar_names)]
use std::collections::HashSet;

use egg::StopReason;
use rand::Rng;

use crate::Label;
use crate::tree::{TreeShaped, TypedTree};

/// A binder operator: at sampling time, child `bound_slot` is forced to be a
/// variable leaf, and all other children are drawn from the term distribution.
/// `scope_slots` lists the child indices in which the bound variable is in
/// scope.
#[derive(Clone, Debug)]
pub struct BinderOp<L: 'static> {
    pub op: L,
    pub arity: usize,
    pub bound_slot: usize,
    pub scope_slots: &'static [usize],
}

/// Description of a language for the Fixpoint sampler.
///
/// `NORMAL_OPS[k - 1]` is the list of non-binder operators of arity `k` (so
/// `NORMAL_OPS[0]` is unary, `NORMAL_OPS[1]` binary, etc.). Leaves are returned
/// separately via [`leaves`](LanguageSpec::leaves) since they typically include
/// non-const values (interned symbols, parsed constants).
///
/// `BINDERS` lists binder operators of any arity.
///
/// `variables` is a subset of `leaves()` — leaves that may appear as a
/// binder's bound variable.
pub trait LanguageSpec {
    type Label: Label + Copy + 'static;

    /// Non-binder operators by arity, starting at arity 1.
    const NORMAL_OPS: &'static [&'static [Self::Label]];
    const BINDERS: &'static [BinderOp<Self::Label>];

    fn leaves(&self) -> &'static [Self::Label];
    fn variables(&self) -> &'static [Self::Label];

    fn is_valid_tree(&self, tree: &TypedTree<Self::Label>) -> bool;
    fn free_vars(&self, tree: &TypedTree<Self::Label>) -> HashSet<Self::Label>;
}

fn max_arity<S: LanguageSpec + ?Sized>(_spec: &S) -> usize {
    let normal_max = S::NORMAL_OPS.len();
    let binder_max = S::BINDERS.iter().map(|b| b.arity).max().unwrap_or(0);
    normal_max.max(binder_max)
}

/// Fixpoint sampler for random terms of a target size.
///
/// The grammar is described by a [`LanguageSpec`] in n-ary form. The generating
/// function aggregates over all arities `k`:
///
/// ```text
/// T(x) = sum_k |ops[k]| * x * T^k
///      + sum_b x * T^(b.arity - 1) * (n_v * x)
/// ```
///
/// where the second sum is over binders (each binder has one slot forced to a
/// variable leaf, contributing the `n_v * x` factor).
///
/// `T(x)` is solved numerically by Newton iteration. The singularity `rho` is
/// found as the largest `x` where the iteration still converges to a finite
/// real root. We binary-search `x < rho` for the desired expected size, then
/// use rejection sampling, filtering candidates with `spec.is_valid_tree`.
pub struct FixPointSampler<S: LanguageSpec> {
    /// Cumulative thresholds + which class to sample for each draw.
    choices: Vec<(f64, OpChoice)>,
    spec: S,
    target: usize,
    tolerance: usize,
    max_depth: usize,
}

#[derive(Clone, Copy)]
enum OpChoice {
    Leaf,
    Normal { arity: usize },
    Binder { idx: usize },
}

#[expect(clippy::cast_precision_loss)]
impl<S: LanguageSpec> FixPointSampler<S> {
    /// Create a sampler targeting terms of the given expected size.
    ///
    /// # Panics
    ///
    /// Panics if any operator arity in the spec exceeds `i32::MAX`.
    #[must_use]
    pub fn new(target: usize, tolerance: usize, spec: S) -> Self {
        let max_arity = max_arity(&spec);
        let counts = OpCounts::from_spec(&spec);

        let rho = find_singularity(&counts, max_arity);
        let x = find_tuning_param(target as f64, rho, &counts, max_arity);
        let t = eval_t(x, &counts, max_arity);

        let mut choices = Vec::new();
        let mut acc = 0.0;

        let p_leaf = counts.n_l * x / t;
        acc += p_leaf;
        choices.push((acc, OpChoice::Leaf));

        for arity in 1..=max_arity {
            let n_k = counts.normal_at(arity);
            if n_k == 0.0 {
                continue;
            }
            let p = n_k * x * t.powi(i32::try_from(arity).unwrap() - 1);
            acc += p;
            choices.push((acc, OpChoice::Normal { arity }));
        }

        for (idx, b) in S::BINDERS.iter().enumerate() {
            let p = x * t.powi(i32::try_from(b.arity).unwrap() - 1) * counts.n_v * x;
            acc += p;
            choices.push((acc, OpChoice::Binder { idx }));
        }

        if let Some(last) = choices.last_mut() {
            last.0 = 1.0;
        }

        let max_depth = (target + tolerance) * 4;

        FixPointSampler {
            choices,
            spec,
            target,
            tolerance,
            max_depth,
        }
    }

    fn gen_node(&self, rng: &mut impl Rng, depth: usize) -> TypedTree<S::Label> {
        if depth >= self.max_depth {
            return self.gen_leaf(rng);
        }
        let r = rng.r#gen::<f64>();
        let choice = self
            .choices
            .iter()
            .find(|(thr, _)| r < *thr)
            .map_or(OpChoice::Leaf, |(_, c)| *c);

        match choice {
            OpChoice::Leaf => self.gen_leaf(rng),
            OpChoice::Normal { arity } => {
                let pool = S::NORMAL_OPS[arity - 1];
                let op = pool[rng.gen_range(0..pool.len())];
                let children = (0..arity).map(|_| self.gen_node(rng, depth + 1)).collect();
                TypedTree::new_untyped(op, children)
            }
            OpChoice::Binder { idx } => {
                let b = &S::BINDERS[idx];
                let mut children = Vec::with_capacity(b.arity);
                for slot in 0..b.arity {
                    if slot == b.bound_slot {
                        let vars = self.spec.variables();
                        let var = vars[rng.gen_range(0..vars.len())];
                        children.push(TypedTree::leaf_untyped(var));
                    } else {
                        children.push(self.gen_node(rng, depth + 1));
                    }
                }
                TypedTree::new_untyped(b.op, children)
            }
        }
    }

    fn gen_leaf(&self, rng: &mut impl Rng) -> TypedTree<S::Label> {
        let leaves = self.spec.leaves();
        let label = leaves[rng.gen_range(0..leaves.len())];
        TypedTree::leaf_untyped(label)
    }

    /// Generate a random term whose size is in `[target - tolerance, target + tolerance]`
    /// and whose structure passes `spec.is_valid_tree`. Returns `None` if no
    /// valid tree is found within `100_000` attempts.
    pub fn sample<R: Rng, F: Fn(&TypedTree<S::Label>) -> Option<StopReason>>(
        &self,
        rng: &mut R,
        filter_hook: &F,
    ) -> Option<(TypedTree<S::Label>, StopReason, usize)> {
        let lo = self.target.saturating_sub(self.tolerance);
        let hi = self.target + self.tolerance;
        (0..100_000).find_map(|n| {
            let candidate = self.gen_node(rng, 0);
            if (lo..=hi).contains(&candidate.size_without_types())
                && self.spec.is_valid_tree(&candidate)
                && let Some(reason) = filter_hook(&candidate)
            {
                return Some((candidate, reason, n));
            }
            None
        })
    }

    /// Generate `count` random terms within the size window.
    pub fn sample_many<R: Rng, F: Fn(&TypedTree<S::Label>) -> Option<StopReason>>(
        &self,
        rng: &mut R,
        count: usize,
        filter_hook: &F,
    ) -> Vec<(TypedTree<S::Label>, StopReason)> {
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

struct OpCounts {
    n_l: f64,
    n_v: f64,
    normal_per_arity: Vec<f64>,
    binder_per_arity: Vec<f64>,
}

impl OpCounts {
    fn from_spec<S: LanguageSpec>(spec: &S) -> Self {
        #![allow(clippy::cast_precision_loss)]
        let max_arity = max_arity(spec);
        let mut normal_per_arity = vec![0.0; max_arity + 1];
        for (i, ops) in S::NORMAL_OPS.iter().enumerate() {
            let k = i + 1;
            if k <= max_arity {
                normal_per_arity[k] = ops.len() as f64;
            }
        }
        let mut binder_per_arity = vec![0.0; max_arity + 1];
        for b in S::BINDERS {
            binder_per_arity[b.arity] += 1.0;
        }
        OpCounts {
            n_l: spec.leaves().len() as f64,
            n_v: spec.variables().len() as f64,
            normal_per_arity,
            binder_per_arity,
        }
    }

    fn normal_at(&self, k: usize) -> f64 {
        self.normal_per_arity.get(k).copied().unwrap_or(0.0)
    }
}

fn gf_residual(t: f64, x: f64, counts: &OpCounts, max_arity: usize) -> f64 {
    let mut sum = counts.n_l * x;
    for k in 1..=max_arity {
        let n_k = counts.normal_at(k);
        if n_k != 0.0 {
            sum += n_k * x * t.powi(i32::try_from(k).unwrap());
        }
        let n_bk = counts.binder_per_arity.get(k).copied().unwrap_or(0.0);
        if n_bk != 0.0 {
            sum += n_bk * x * t.powi(i32::try_from(k).unwrap() - 1) * counts.n_v * x;
        }
    }
    t - sum
}

fn gf_residual_dt(t: f64, x: f64, counts: &OpCounts, max_arity: usize) -> f64 {
    #![allow(clippy::cast_precision_loss)]
    let mut sum = 0.0;
    for k in 1..=max_arity {
        let n_k = counts.normal_at(k);
        if n_k != 0.0 {
            sum += n_k * x * (k as f64) * t.powi(i32::try_from(k).unwrap() - 1);
        }
        let n_bk = counts.binder_per_arity.get(k).copied().unwrap_or(0.0);
        if n_bk != 0.0 && k >= 2 {
            sum += n_bk
                * x
                * counts.n_v
                * x
                * ((k - 1) as f64)
                * t.powi(i32::try_from(k).unwrap() - 2);
        }
    }
    1.0 - sum
}

fn eval_t(x: f64, counts: &OpCounts, max_arity: usize) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut t = counts.n_l * x;
    for _ in 0..200 {
        let f = gf_residual(t, x, counts, max_arity);
        let df = gf_residual_dt(t, x, counts, max_arity);
        if df.abs() < 1e-30 || !df.is_finite() {
            return f64::NAN;
        }
        let step = f / df;
        let next = t - step;
        if !next.is_finite() {
            return f64::NAN;
        }
        let next = next.max(0.0);
        if (next - t).abs() < 1e-14 * (1.0 + t.abs()) {
            return next;
        }
        t = next;
    }
    f64::NAN
}

fn find_singularity(counts: &OpCounts, max_arity: usize) -> f64 {
    let n_u = counts.normal_at(1);
    let mut hi = if n_u > 0.0 { 1.0 / (n_u + 1.0) } else { 1.0 };
    while !eval_t(hi, counts, max_arity).is_finite() {
        hi *= 0.5;
        if hi < 1e-30 {
            return 0.0;
        }
    }
    let mut last_good = hi;
    while eval_t(hi, counts, max_arity).is_finite() {
        last_good = hi;
        hi *= 2.0;
        if hi > 1e6 {
            break;
        }
    }
    let mut lo = last_good;
    for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        if eval_t(mid, counts, max_arity).is_finite() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

fn expected_size(x: f64, counts: &OpCounts, max_arity: usize) -> f64 {
    let eps = x * 1e-8;
    let t1 = eval_t(x + eps, counts, max_arity);
    let t0 = eval_t(x - eps, counts, max_arity);
    let t = eval_t(x, counts, max_arity);
    let t_prime = (t1 - t0) / (2.0 * eps);
    x * t_prime / t
}

fn find_tuning_param(target: f64, rho: f64, counts: &OpCounts, max_arity: usize) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = rho * (1.0 - 1e-10);
    for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        if expected_size(mid, counts, max_arity) < target {
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

    fn math_counts() -> (OpCounts, usize) {
        let counts = OpCounts {
            n_l: 5.0,
            n_v: 2.0,
            normal_per_arity: vec![5.0, 4.0, 5.0],
            binder_per_arity: vec![0.0, 0.0, 2.0],
        };
        (counts, 2)
    }

    #[test]
    fn find_singularity_positive_and_t_finite_below() {
        let (counts, max_arity) = math_counts();
        let rho = find_singularity(&counts, max_arity);
        assert!(rho > 0.0, "singularity should be positive, got {rho}");
        let t_below = eval_t(rho * 0.9, &counts, max_arity);
        assert!(t_below.is_finite(), "T should be finite below rho");
    }

    #[test]
    fn find_tuning_param_expected_size_matches_target() {
        let (counts, max_arity) = math_counts();
        let rho = find_singularity(&counts, max_arity);
        for target in [5.0, 10.0, 20.0, 50.0] {
            let x = find_tuning_param(target, rho, &counts, max_arity);
            let actual = expected_size(x, &counts, max_arity);
            assert!(
                (actual - target).abs() < 0.5,
                "target={target}, actual expected size={actual}"
            );
        }
    }

    #[test]
    fn expected_size_increases_with_x() {
        let (counts, max_arity) = math_counts();
        let rho = find_singularity(&counts, max_arity);
        let e1 = expected_size(rho * 0.3, &counts, max_arity);
        let e2 = expected_size(rho * 0.7, &counts, max_arity);
        let e3 = expected_size(rho * 0.99, &counts, max_arity);
        assert!(e1 < e2, "e1={e1} should be < e2={e2}");
        assert!(e2 < e3, "e2={e2} should be < e3={e3}");
    }
}
