#![expect(clippy::similar_names)]
use std::collections::HashSet;

use egg::Symbol;
use ordered_float::NotNan;
use rand::Rng;

use crate::tree::{Tree, TreeShaped};

use super::label::MathLabel;

/// Boltzmann sampler for random math terms of a target size.
///
/// The grammar distinguishes three kinds of binary operators:
///   - Normal binaries (Add, Sub, Mul, Div, Pow): both children drawn from T
///   - Binder binaries (Diff, Integral): child[0] drawn from T, child[1] forced to a variable leaf
///
/// The corrected generating function is:
///   `T(x) = n_l*x + n_u*x*T + n_b_normal*x*T^2 + n_b_binder*x*T*(n_v*x)`
///
/// where `n_l = |leaf_symbols|`, `n_u = |unary_ops|`, `n_b_normal = |normal_binary_ops|`,
/// `n_b_binder` = |`binder_ops`|, `n_v` = |`var_symbols`|.
///
/// Rearranging as a quadratic in T:
///   `n_b_normal*x*T^2 + (n_u*x + n_b_binder*n_v*x^2 - 1)*T + n_l*x = 0`
///
/// The singularity `rho` is found numerically (`discriminant = 0`).
/// We binary-search for `x < rho` giving the target expected size, then use rejection sampling.
///
/// Additionally, generated trees are filtered so that the bound variable in each Diff/Integral
/// node actually appears free in child[0].
pub struct BoltzmannSampler {
    /// Probability of generating a leaf node at each step.
    p_leaf: f64,
    /// Cumulative probability threshold for unary nodes (after leaf).
    p_unary: f64,
    /// Cumulative probability threshold for binder nodes (after unary).
    p_binder: f64,
    // `p_normal_binary = 1 - p_binder` (implicit, remainder)
    /// Full pool of leaf labels (variables + constants).
    symbols: Vec<MathLabel>,
    /// Variable-only subset of the leaf pool (used as binder targets).
    var_symbols: Vec<MathLabel>,
    /// Unary operators.
    unary_ops: Vec<MathLabel>,
    /// Normal binary operators (both children are expressions).
    normal_binary_ops: Vec<MathLabel>,
    /// Binder operators (child[0] = expr, child[1] = variable).
    binder_ops: Vec<MathLabel>,
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

const NORMAL_BINARY_OPS: [MathLabel; 5] = [
    MathLabel::Add,
    MathLabel::Sub,
    MathLabel::Mul,
    MathLabel::Div,
    MathLabel::Pow,
];

const BINDER_OPS: [MathLabel; 2] = [MathLabel::Diff, MathLabel::Integral];

#[expect(clippy::cast_precision_loss)]
impl BoltzmannSampler {
    /// Create a sampler targeting terms of the given expected size.
    ///
    /// `leaf_symbols` is the full pool of leaf labels (variables and constants).
    /// Variable symbols are those of the form `MathLabel::Symbol(_)`.
    /// If `None`, defaults to `[x, y, 0, 1, 2]`.
    pub fn new(target: usize, tolerance: usize, leaf_symbols: Option<Vec<MathLabel>>) -> Self {
        let leaf_symbols = leaf_symbols.unwrap_or_else(default_symbols);
        let var_symbols = leaf_symbols
            .iter()
            .filter(|l| matches!(l, MathLabel::Symbol(_)))
            .copied()
            .collect::<Vec<_>>();
        let unary_ops = UNARY_OPS.to_vec();
        let normal_binary_ops = NORMAL_BINARY_OPS.to_vec();
        let binder_ops = BINDER_OPS.to_vec();

        let n_l = leaf_symbols.len() as f64;
        let n_u = unary_ops.len() as f64;
        let n_bn = normal_binary_ops.len() as f64;
        let n_bb = binder_ops.len() as f64;
        let n_v = var_symbols.len() as f64;

        // Corrected GF (quadratic in T):
        //   n_bn*x*T^2 + (n_u*x + n_bb*n_v*x^2 - 1)*T + n_l*x = 0
        //
        // Singularity rho: discriminant D(x) = (n_u*x + n_bb*n_v*x^2 - 1)^2 - 4*n_bn*n_l*x^2 = 0
        // No closed form => find numerically.
        let rho = find_singularity(n_l, n_u, n_bn, n_bb, n_v);

        let x = find_tuning_param(target as f64, rho, n_l, n_u, n_bn, n_bb, n_v);
        let t = eval_t(x, n_l, n_u, n_bn, n_bb, n_v);

        // Boltzmann probabilities => each term in GF divided by T:
        //   P(leaf)          = n_l * x / T
        //   P(unary)         = n_u * x
        //   P(binder)        = n_bb * n_v * x^2          (binder node + forced var leaf)
        //   P(normal_binary) = n_bn * x * T
        let p_leaf = n_l * x / t;
        let p_unary = n_u * x;
        let p_binder = n_bb * n_v * x * x;

        let max_depth = (target + tolerance) * 4;

        BoltzmannSampler {
            p_leaf,
            p_unary: p_leaf + p_unary,
            p_binder: p_leaf + p_unary + p_binder,
            symbols: leaf_symbols,
            var_symbols,
            unary_ops,
            normal_binary_ops,
            binder_ops,
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
        } else if r < self.p_unary {
            let op = self.unary_ops[rng.gen_range(0..self.unary_ops.len())];
            let child = self.gen_node(rng, depth + 1);
            Tree::new_untyped(op, vec![child])
        } else if r < self.p_binder {
            let op = self.binder_ops[rng.gen_range(0..self.binder_ops.len())];
            let expr = self.gen_node(rng, depth + 1);
            let var_label = self.var_symbols[rng.gen_range(0..self.var_symbols.len())];
            let var = Tree::leaf_untyped(var_label);
            Tree::new_untyped(op, vec![expr, var])
        } else {
            let op = self.normal_binary_ops[rng.gen_range(0..self.normal_binary_ops.len())];
            let left = self.gen_node(rng, depth + 1);
            let right = self.gen_node(rng, depth + 1);
            Tree::new_untyped(op, vec![left, right])
        }
    }

    /// Generate a random term whose size is in `[target - tolerance, target + tolerance]`
    /// and where every Diff/Integral node's bound variable appears free in its expression child.
    /// Returns None if no valid tree is found within `10_000` attempts.
    pub fn sample<R: Rng, F: Fn(&Tree<MathLabel>) -> bool>(
        &self,
        rng: &mut R,
        filter_hook: &F,
    ) -> Option<(Tree<MathLabel>, usize)> {
        let lo = self.target.saturating_sub(self.tolerance);
        let hi = self.target + self.tolerance;
        (0..100_000)
            .map(|a| (self.gen_node(rng, 0), a))
            .find(|(candidate, _)| {
                (lo..=hi).contains(&candidate.size_without_types())
                    && binders_valid(candidate)
                    && filter_hook(candidate)
            })
    }

    /// Generate `count` random terms within the size window.
    pub fn sample_many<R: Rng, F: Fn(&Tree<MathLabel>) -> bool>(
        &self,
        rng: &mut R,
        count: usize,
        filter_hook: &F,
    ) -> Vec<Tree<MathLabel>> {
        let (trees, total_attempts, failed) =
            (0..count).map(|_| self.sample(rng, filter_hook)).fold(
                (Vec::with_capacity(count), 0, 0),
                |(mut trees, attempts, failed), result| match result {
                    Some((tree, a)) => {
                        trees.push(tree);
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

/// Returns true if every Diff/Integral node in the tree has its bound variable
/// appearing free somewhere in its expression child (child[0]).
fn binders_valid(tree: &Tree<MathLabel>) -> bool {
    if let MathLabel::Diff | MathLabel::Integral = tree.label() {
        let children = tree.children();
        let expr = &children[0];
        let MathLabel::Symbol(var) = children[1].label() else {
            return false;
        };
        free_vars(expr).contains(var) && binders_valid(expr)
    } else {
        tree.children().iter().all(binders_valid)
    }
}

/// Collect all variable symbols that appear free in the tree.
/// Variables bound by an enclosing Diff/Integral are excluded from the free
/// set of that binder's expression child.
fn free_vars(tree: &Tree<MathLabel>) -> HashSet<Symbol> {
    match tree.label() {
        MathLabel::Symbol(s) => {
            let mut set = HashSet::new();
            set.insert(*s);
            set
        }
        MathLabel::Diff | MathLabel::Integral => {
            // child[0] contributes free vars minus the bound var in child[1];
            // we exclude the bound variable from child[0]'s free vars.
            let children = tree.children();
            let mut vars = free_vars(&children[0]);
            if let MathLabel::Symbol(bound) = children[1].label() {
                vars.remove(bound);
            }
            vars
        }
        _ => tree.children().iter().flat_map(free_vars).collect(),
    }
}

/// Evaluate `T(x)` for the corrected grammar:
///   `n_bn*x*T^2 + (n_u*x + n_bb*n_v*x^2 - 1)*T + n_l*x = 0`
/// Taking the smaller root (the one that goes to 0 as `x -> 0`).
fn eval_t(x: f64, n_l: f64, n_u: f64, n_bn: f64, n_bb: f64, n_v: f64) -> f64 {
    let b = n_u * x + n_bb * n_v * x * x - 1.0;
    let disc = b * b - 4.0 * n_bn * n_l * x * x;
    (-b - disc.sqrt()) / (2.0 * n_bn * x)
}

/// Discriminant of the quadratic in `T`. Zero at the singularity `rho`.
fn discriminant(x: f64, n_l: f64, n_u: f64, n_bn: f64, n_bb: f64, n_v: f64) -> f64 {
    let b = n_u * x + n_bb * n_v * x * x - 1.0;
    b * b - 4.0 * n_bn * n_l * x * x
}

/// Find the singularity `rho` numerically: the smallest positive `x` where `disc(x) = 0`.
/// Binary search in `(0, 1/n_u) => disc` is positive near `0` and negative past `rho`.
fn find_singularity(n_l: f64, n_u: f64, n_bn: f64, n_bb: f64, n_v: f64) -> f64 {
    // Upper bound: at x = 1/n_u the original (n_bb=0) singularity would be exceeded.
    // Use 1/(n_u + 1) as a safe starting upper bracket where disc might still be positive,
    // then scan to find a sign change.
    let mut lo = 0.0_f64;
    let mut hi = 1.0 / (n_u + 1.0);
    // Extend hi until disc goes negative.
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

/// Expected size of a Boltzmann-sampled tree at parameter `x`, computed via numerical differentiation:
///   `E[size] = x * T'(x) / T(x)`
fn expected_size(x: f64, n_l: f64, n_u: f64, n_bn: f64, n_bb: f64, n_v: f64) -> f64 {
    let eps = x * 1e-8;
    let t1 = eval_t(x + eps, n_l, n_u, n_bn, n_bb, n_v);
    let t0 = eval_t(x - eps, n_l, n_u, n_bn, n_bb, n_v);
    let t = eval_t(x, n_l, n_u, n_bn, n_bb, n_v);
    let t_prime = (t1 - t0) / (2.0 * eps);
    x * t_prime / t
}

/// Binary search for `x` in `(0, rho)` giving the desired expected size.
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

        let trees = sampler.sample_many(&mut rng, 50, &|_| true);
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

    #[test]
    fn small_target_size() {
        let sampler = BoltzmannSampler::new(5, 2, None);
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let trees = sampler.sample_many(&mut rng, 30, &|_| true);
        for tree in &trees {
            let size = tree.size_without_types();
            assert!((3..=7).contains(&size), "size {size} out of range [3, 7]");
        }
    }

    #[test]
    fn binders_have_valid_bound_variables() {
        let sampler = BoltzmannSampler::new(15, 5, None);
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        let trees = sampler.sample_many(&mut rng, 100, &|_| true);
        for tree in &trees {
            assert!(
                binders_valid(tree),
                "tree has binder with non-free bound variable: {tree:?}"
            );
        }
    }

    #[test]
    fn binder_child1_is_always_a_variable() {
        let sampler = BoltzmannSampler::new(15, 5, None);
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        let trees = sampler.sample_many(&mut rng, 100, &|_| true);
        for tree in &trees {
            assert_no_constant_binder(tree);
        }
    }

    fn assert_no_constant_binder(tree: &Tree<MathLabel>) {
        if matches!(tree.label(), MathLabel::Diff | MathLabel::Integral) {
            let children = tree.children();
            assert!(
                matches!(children[1].label(), MathLabel::Symbol(_)),
                "binder child[1] is not a Symbol: {:?}",
                children[1].label()
            );
        }
        for child in tree.children() {
            assert_no_constant_binder(child);
        }
    }

    // --- helpers for hand-crafted trees ---

    fn sym(name: &str) -> Tree<MathLabel> {
        Tree::leaf_untyped(MathLabel::Symbol(name.into()))
    }

    fn diff(expr: Tree<MathLabel>, var: Tree<MathLabel>) -> Tree<MathLabel> {
        Tree::new_untyped(MathLabel::Diff, vec![expr, var])
    }

    fn add(l: Tree<MathLabel>, r: Tree<MathLabel>) -> Tree<MathLabel> {
        Tree::new_untyped(MathLabel::Add, vec![l, r])
    }

    // --- free_vars ---

    #[test]
    fn free_vars_single_symbol() {
        let tree = sym("x");
        let fv = free_vars(&tree);
        assert_eq!(fv, ["x".into()].into());
    }

    #[test]
    fn free_vars_binder_removes_bound_var() {
        // diff(x, x)  =>  x is bound, so free vars = {}
        let tree = diff(sym("x"), sym("x"));
        assert!(free_vars(&tree).is_empty());
    }

    #[test]
    fn free_vars_binder_keeps_other_vars() {
        // diff(add(x, y), x)  =>  x bound, y free
        let tree = diff(add(sym("x"), sym("y")), sym("x"));
        let fv = free_vars(&tree);
        assert!(!fv.contains(&"x".into()), "x should be bound");
        assert!(fv.contains(&"y".into()), "y should be free");
    }

    #[test]
    fn free_vars_nested_binders() {
        // diff(diff(add(x,y), x), y)
        //   inner diff: free in add(x,y) minus x  => {y}
        //   outer diff: free in inner minus y     => {}
        let inner = diff(add(sym("x"), sym("y")), sym("x"));
        let tree = diff(inner, sym("y"));
        assert!(free_vars(&tree).is_empty());
    }

    #[test]
    fn free_vars_no_binders_collects_all_symbols() {
        // add(x, y) => {x, y}
        let tree = add(sym("x"), sym("y"));
        let fv = free_vars(&tree);
        assert_eq!(fv, ["x".into(), "y".into()].into());
    }

    // --- binders_valid ---

    #[test]
    fn binders_valid_simple_valid() {
        // diff(x, x): x appears free in expr => valid
        assert!(binders_valid(&diff(sym("x"), sym("x"))));
    }

    #[test]
    fn binders_valid_simple_invalid() {
        // diff(x, y): y does NOT appear free in expr `x` => invalid
        assert!(!binders_valid(&diff(sym("x"), sym("y"))));
    }

    #[test]
    fn binders_valid_no_binders() {
        // Pure arithmetic tree is always valid
        assert!(binders_valid(&add(sym("x"), sym("y"))));
    }

    #[test]
    fn binders_valid_nested_valid() {
        // diff(diff(add(x,y), x), y): inner valid (x free in add(x,y)),
        // outer valid (y is free in diff(add(x,y), x) after x is removed => {y})
        let inner = diff(add(sym("x"), sym("y")), sym("x"));
        let tree = diff(inner, sym("y"));
        assert!(binders_valid(&tree));
    }

    #[test]
    fn binders_valid_nested_outer_invalid() {
        // diff(diff(x, x), y): inner valid (x free in x),
        // but outer bound var y is NOT free in diff(x,x) (free vars = {})
        let inner = diff(sym("x"), sym("x"));
        let tree = diff(inner, sym("y"));
        assert!(!binders_valid(&tree));
    }

    // --- numerical routines ---

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
    fn sample_many_count_zero() {
        let sampler = BoltzmannSampler::new(10, 3, None);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let trees = sampler.sample_many(&mut rng, 0, &|_| true);
        assert!(trees.is_empty());
    }

    #[test]
    fn default_symbols_contains_expected_leaves() {
        let syms = default_symbols();
        assert!(syms.contains(&MathLabel::Symbol("x".into())));
        assert!(syms.contains(&MathLabel::Symbol("y".into())));
        assert!(syms.contains(&MathLabel::Constant(NotNan::new(0.0).unwrap())));
        assert!(syms.contains(&MathLabel::Constant(NotNan::new(1.0).unwrap())));
        assert!(syms.contains(&MathLabel::Constant(NotNan::new(2.0).unwrap())));
        assert_eq!(syms.len(), 5);
    }
}
