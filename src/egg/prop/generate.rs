use std::sync::LazyLock;

use egg::RecExpr;
use rand::Rng;

use crate::egg::{id0, stack_children};
use crate::sampler::BoltzmannSampler;

use super::Prop;

/// Boltzmann sampler for random propositional terms of a target size.
///
/// Grammar (all binary ops symmetric — both children drawn from T):
///   `T(x) = n_l*x + n_u*x*T + n_b*x*T^2`
///
/// where `n_l = |leaf_symbols|`, `n_u = |unary_ops|` (Not),
/// `n_b = |binary_ops|` (And, Or, Implies).
///
/// Rearranging as a quadratic in T:
///   `n_b*x*T^2 + (n_u*x - 1)*T + n_l*x = 0`
///
/// Singularity `rho` has a closed form (set discriminant to zero):
///   `rho = 1 / (n_u + 2 * sqrt(n_b * n_l))`
///
/// We binary-search for `x < rho` giving the target expected size, then use rejection sampling.
pub struct PropSampler {
    /// Probability of generating a leaf node at each step.
    p_leaf: f64,
    /// Cumulative probability threshold for unary nodes (after leaf).
    p_unary: f64,
    // `p_binary = 1 - p_unary` (implicit, remainder)
    /// Pool of leaf labels.
    symbols: Vec<Prop>,
    /// Unary operators.
    unary_ops: Vec<Prop>,
    /// Binary operators.
    binary_ops: Vec<Prop>,
    /// Target expression size for rejection sampling.
    target: usize,
    /// Accepted size range is `[target - tolerance, target + tolerance]`.
    tolerance: usize,
    /// Maximum recursion depth before forcing a leaf (prevents stack overflow).
    max_depth: usize,
}

static UNARY_OPS: LazyLock<[Prop; 1]> = LazyLock::new(|| [Prop::Not(id0())]);

static BINARY_OPS: LazyLock<[Prop; 3]> = LazyLock::new(|| {
    [
        Prop::And([id0(), id0()]),
        Prop::Or([id0(), id0()]),
        Prop::Implies([id0(), id0()]),
    ]
});

#[expect(clippy::cast_precision_loss)]
impl BoltzmannSampler for PropSampler {
    type Lang = Prop;

    /// Create a sampler targeting terms of the given expected size.
    ///
    /// `leaf_symbols` is the pool of leaf labels.
    /// If `None`, defaults to `[a, b, c]`.
    fn new(target: usize, tolerance: usize, leaf_symbols: Option<Vec<Prop>>) -> Self {
        let symbols = leaf_symbols.unwrap_or_else(default_symbols);
        let unary_ops = UNARY_OPS.to_vec();
        let binary_ops = BINARY_OPS.to_vec();

        let n_l = symbols.len() as f64;
        let n_u = unary_ops.len() as f64;
        let n_b = binary_ops.len() as f64;

        let rho = 1.0 / (n_u + 2.0 * (n_b * n_l).sqrt());

        let x = find_tuning_param(target as f64, rho, n_l, n_u, n_b);
        let t = eval_t(x, n_l, n_u, n_b);

        // Boltzmann probabilities => each term in GF divided by T:
        //   P(leaf)   = n_l * x / T
        //   P(unary)  = n_u * x
        //   P(binary) = n_b * x * T   (remainder)
        let p_leaf = n_l * x / t;
        let p_unary = n_u * x;

        let max_depth = (target + tolerance) * 4;

        PropSampler {
            p_leaf,
            p_unary: p_leaf + p_unary,
            symbols,
            unary_ops,
            binary_ops,
            target,
            tolerance,
            max_depth,
        }
    }

    fn target(&self) -> usize {
        self.target
    }

    fn tolerance(&self) -> usize {
        self.tolerance
    }

    fn gen_node<R: Rng>(&self, rng: &mut R, depth: usize) -> RecExpr<Prop> {
        let r = rng.r#gen::<f64>();
        if depth >= self.max_depth || r < self.p_leaf {
            let label = self.symbols[rng.gen_range(0..self.symbols.len())].clone();
            stack_children(&[], label)
        } else if r < self.p_unary {
            let op = self.unary_ops[rng.gen_range(0..self.unary_ops.len())].clone();
            let child = self.gen_node(rng, depth + 1);
            stack_children(&[child], op)
        } else {
            let op = self.binary_ops[rng.gen_range(0..self.binary_ops.len())].clone();
            let left = self.gen_node(rng, depth + 1);
            let right = self.gen_node(rng, depth + 1);
            stack_children(&[left, right], op)
        }
    }
}

/// Evaluate `T(x)` for the grammar:
///   `n_b*x*T^2 + (n_u*x - 1)*T + n_l*x = 0`
/// Taking the smaller root (the one that goes to 0 as `x -> 0`).
fn eval_t(x: f64, n_l: f64, n_u: f64, n_b: f64) -> f64 {
    let b = n_u * x - 1.0;
    let disc = b * b - 4.0 * n_b * n_l * x * x;
    (-b - disc.sqrt()) / (2.0 * n_b * x)
}

/// Expected size at parameter `x`: `E[size] = x * T'(x) / T(x)` (numerical derivative).
fn expected_size(x: f64, n_l: f64, n_u: f64, n_b: f64) -> f64 {
    let eps = x * 1e-8;
    let t1 = eval_t(x + eps, n_l, n_u, n_b);
    let t0 = eval_t(x - eps, n_l, n_u, n_b);
    let t = eval_t(x, n_l, n_u, n_b);
    let t_prime = (t1 - t0) / (2.0 * eps);
    x * t_prime / t
}

/// Binary search for `x` in `(0, rho)` giving the desired expected size.
fn find_tuning_param(target: f64, rho: f64, n_l: f64, n_u: f64, n_b: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = rho * (1.0 - 1e-10);
    for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        if expected_size(mid, n_l, n_u, n_b) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

fn default_symbols() -> Vec<Prop> {
    (0..20)
        .map(|i| Prop::Symbol(format!("x{i}").into()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::{AstSize, CostFunction};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn sampler_produces_exprs_near_target() {
        let sampler = PropSampler::new(15, 5, None);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let exprs = sampler.sample_many(&mut rng, 50, &|_| Some(()));
        assert_eq!(exprs.len(), 50);
        for expr in &exprs {
            let size = AstSize.cost_rec(&expr.0);
            assert!(
                (10..=20).contains(&size),
                "size {size} out of range [10, 20]"
            );
        }
    }

    #[test]
    fn expected_size_increases_with_x() {
        let n_l = 3.0f64;
        let n_u = 1.0f64;
        let n_b = 3.0f64;
        let rho = 1.0 / (n_u + 2.0 * (n_b * n_l).sqrt());

        let e1 = expected_size(rho * 0.3, n_l, n_u, n_b);
        let e2 = expected_size(rho * 0.7, n_l, n_u, n_b);
        let e3 = expected_size(rho * 0.99, n_l, n_u, n_b);
        assert!(e1 < e2, "e1={e1} should be < e2={e2}");
        assert!(e2 < e3, "e2={e2} should be < e3={e3}");
    }

    #[test]
    fn small_target_size() {
        let sampler = PropSampler::new(5, 2, None);
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let exprs = sampler.sample_many(&mut rng, 30, &|_| Some(()));
        for expr in &exprs {
            let size = AstSize.cost_rec(&expr.0);
            assert!((3..=7).contains(&size), "size {size} out of range [3, 7]");
        }
    }

    #[test]
    fn find_tuning_param_expected_size_matches_target() {
        let (n_l, n_u, n_b) = (3.0_f64, 1.0_f64, 3.0_f64);
        let rho = 1.0 / (n_u + 2.0 * (n_b * n_l).sqrt());
        for target in [5.0, 10.0, 20.0, 50.0] {
            let x = find_tuning_param(target, rho, n_l, n_u, n_b);
            let actual = expected_size(x, n_l, n_u, n_b);
            assert!(
                (actual - target).abs() < 0.5,
                "target={target}, actual expected size={actual}"
            );
        }
    }

    #[test]
    fn sample_many_count_zero() {
        let sampler = PropSampler::new(10, 3, None);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let exprs = sampler.sample_many(&mut rng, 0, &|_| Some(()));
        assert!(exprs.is_empty());
    }
}
