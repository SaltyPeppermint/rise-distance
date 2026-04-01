use ordered_float::NotNan;
use rand::Rng;

use crate::tree::{Tree, TreeShaped};

use super::label::MathLabel;

/// Boltzmann sampler for random math terms of a target size.
///
/// The grammar has the shape:
///   T(x) = `n_leaf` * x + `n_unary` * x * T(x) + `n_binary` * x * T(x)^2
///
/// We solve for the singularity rho and tune x < rho to control expected size.
/// Terms outside `[target - tolerance, target + tolerance]` are rejected.
pub struct BoltzmannSampler {
    /// Probability of generating a leaf node at each step.
    p_leaf: f64,
    /// Probability of generating a unary node (cumulative with `p_leaf` gives threshold).
    p_unary: f64,
    // p_binary = 1 - p_leaf - p_unary (implicit)
    /// Pool of variable/constant names for leaves.
    symbols: Vec<MathLabel>,
    /// Unary operators.
    unary_ops: Vec<MathLabel>,
    /// Binary operators.
    binary_ops: Vec<MathLabel>,
    /// Target tree size for rejection sampling.
    target: usize,
    /// Accepted size range is `[target - tolerance, target + tolerance]`.
    tolerance: usize,
    /// Maximum recursion depth before forcing a leaf (prevents stack overflow).
    max_depth: usize,
}

const UNARY_OPS: [MathLabel; 4] = [
    MathLabel::Ln,
    MathLabel::Sqrt,
    MathLabel::Sin,
    MathLabel::Cos,
];

const BINARY_OPS: [MathLabel; 7] = [
    MathLabel::Add,
    MathLabel::Sub,
    MathLabel::Mul,
    MathLabel::Div,
    MathLabel::Pow,
    MathLabel::Diff,
    MathLabel::Integral,
];

#[expect(clippy::cast_precision_loss)]
impl BoltzmannSampler {
    /// Create a sampler targeting terms of the given expected size.
    ///
    /// `symbols` is the pool of leaf labels to draw from (variables and constants).
    /// If empty, defaults to `[x, y, 0, 1, 2]`.
    pub fn new(target: usize, tolerance: usize, leaf_symbols: Option<Vec<MathLabel>>) -> Self {
        let leaf_symbols = leaf_symbols.unwrap_or_else(default_symbols);
        let unary_ops = UNARY_OPS.to_vec();
        let binary_ops = BINARY_OPS.to_vec();

        let n_l = leaf_symbols.len() as f64;
        let n_u = unary_ops.len() as f64;
        let n_b = binary_ops.len() as f64;

        // The generating function is T(x) = n_l*x + n_u*x*T(x) + n_b*x*T(x)^2.
        // Rearranging: n_b*x*T^2 + (n_u*x - 1)*T + n_l*x = 0
        // T(x) = (1 - n_u*x - sqrt((1 - n_u*x)^2 - 4*n_b*n_l*x^2)) / (2*n_b*x)
        //
        // The singularity rho is where the discriminant vanishes:
        //   (1 - n_u*rho)^2 = 4*n_b*n_l*rho^2
        //   1 - n_u*rho = 2*sqrt(n_b*n_l)*rho      (taking positive root)
        //   rho = 1 / (n_u + 2*sqrt(n_b*n_l))
        let rho = 1.0 / (n_u + 2.0 * (n_b * n_l).sqrt());

        // Expected size E[size] = x * T'(x) / T(x) + 1, but more directly:
        // E[size] diverges as x -> rho. We binary search for x that gives the target.
        let x = find_tuning_param(target as f64, rho, n_l, n_u, n_b);

        let t = eval_t(x, n_l, n_u, n_b);

        // Each node is drawn proportionally to its contribution to T:
        //   T = n_l*x + n_u*x*T + n_b*x*T^2
        // Dividing each term by T gives probabilities that sum to 1:
        //   P(leaf)   = n_l * x / T
        //   P(unary)  = n_u * x
        //   P(binary) = n_b * x * T
        let p_leaf = n_l * x / t;
        let p_unary = n_u * x;

        // Cap recursion depth at ~4x target to prevent stack overflow on unlucky samples.
        let max_depth = (target + tolerance) * 4;

        BoltzmannSampler {
            p_leaf,
            p_unary,
            symbols: leaf_symbols,
            unary_ops,
            binary_ops,
            target,
            tolerance,
            max_depth,
        }
    }

    fn gen_node(&self, rng: &mut impl Rng, depth: usize) -> Tree<MathLabel> {
        let r = rng.r#gen::<f64>();
        if depth >= self.max_depth || r < self.p_leaf {
            let label = self.symbols[rng.gen_range(0..self.symbols.len())];
            Tree::leaf_untyped(label)
        } else if r < self.p_leaf + self.p_unary {
            let op = self.unary_ops[rng.gen_range(0..self.unary_ops.len())];
            let child = self.gen_node(rng, depth + 1);
            Tree::new_untyped(op, vec![child])
        } else {
            let op = self.binary_ops[rng.gen_range(0..self.binary_ops.len())];
            let left = self.gen_node(rng, depth + 1);
            let right = self.gen_node(rng, depth + 1);
            Tree::new_untyped(op, vec![left, right])
        }
    }

    /// Generate a random term whose size is in `[target - tolerance, target + tolerance]`.
    /// Uses rejection sampling. Returns None if no valid tree is found within `10_000` attempts.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<Tree<MathLabel>> {
        let lo = self.target.saturating_sub(self.tolerance);
        let hi = self.target + self.tolerance;
        (0..10_000)
            .map(|_| self.gen_node(rng, 0))
            .find(|tree| (lo..=hi).contains(&tree.size_without_types()))
    }

    /// Generate `count` random terms within the size window.
    pub fn sample_many<R: Rng>(&self, rng: &mut R, count: usize) -> Vec<Tree<MathLabel>> {
        (0..count).filter_map(|_| self.sample(rng)).collect()
    }
}

/// Evaluate T(x) = (1 - `n_u`*x - sqrt((1-n_u*x)^2 - 4*`n_b`*`n_l`*x^2)) / (2*`n_b`*x)
fn eval_t(x: f64, n_l: f64, n_u: f64, n_b: f64) -> f64 {
    let a = 1.0 - n_u * x;
    let disc = a * a - 4.0 * n_b * n_l * x * x;
    (a - disc.sqrt()) / (2.0 * n_b * x)
}

/// Expected size of a Boltzmann-sampled tree at parameter x.
/// E[size] = 1 + `n_u`*x*E[size] + `n_b`*x*(E[`size_left`] + E[`size_right`])... but more directly,
/// by differentiating the generating function:
///   E[size] = x * T'(x) / T(x)
/// where T'(x) = dT/dx. We compute this numerically.
fn expected_size(x: f64, n_l: f64, n_u: f64, n_b: f64) -> f64 {
    let eps = x * 1e-8;
    let t1 = eval_t(x + eps, n_l, n_u, n_b);
    let t0 = eval_t(x - eps, n_l, n_u, n_b);
    let t = eval_t(x, n_l, n_u, n_b);
    let t_prime = (t1 - t0) / (2.0 * eps);
    x * t_prime / t
}

/// Binary search for the tuning parameter x in (0, rho) that gives the desired expected size.
fn find_tuning_param(target: f64, rho: f64, n_l: f64, n_u: f64, n_b: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = rho * (1.0 - 1e-10); // stay away from singularity
    for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        let es = expected_size(mid, n_l, n_u, n_b);
        if es < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

fn default_symbols() -> Vec<MathLabel> {
    vec![
        MathLabel::Symbol("x".into()),
        MathLabel::Symbol("y".into()),
        MathLabel::Constant(NotNan::new(0.0).unwrap()),
        MathLabel::Constant(NotNan::new(1.0).unwrap()),
        MathLabel::Constant(NotNan::new(2.0).unwrap()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn sampler_produces_trees_near_target() {
        let sampler = BoltzmannSampler::new(15, 5, None);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let trees = sampler.sample_many(&mut rng, 50);
        assert_eq!(trees.len(), 50);
        for tree in &trees {
            let size = tree.size_without_types();
            assert!(
                (10..=20).contains(&size),
                "size {size} out of range [10, 20]"
            );
        }
    }

    #[test]
    fn expected_size_increases_with_x() {
        let n_l = 5.0f64;
        let n_u = 4.0f64;
        let n_b = 7.0f64;
        let rho = 1.0 / (n_u + 2.0 * (n_b * n_l).sqrt());

        let e1 = expected_size(rho * 0.3, n_l, n_u, n_b);
        let e2 = expected_size(rho * 0.7, n_l, n_u, n_b);
        let e3 = expected_size(rho * 0.99, n_l, n_u, n_b);
        assert!(e1 < e2, "e1={e1} should be < e2={e2}");
        assert!(e2 < e3, "e2={e2} should be < e3={e3}");
    }

    #[test]
    fn small_target_size() {
        let sampler = BoltzmannSampler::new(5, 2, None);
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let trees = sampler.sample_many(&mut rng, 30);
        for tree in &trees {
            let size = tree.size_without_types();
            assert!((3..=7).contains(&size), "size {size} out of range [3, 7]");
        }
    }
}
