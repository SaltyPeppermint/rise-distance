use ordered_float::NotNan;

use crate::egg::LanguageSpec;
use crate::egg::math::MathLabel;

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

/// Build a `LanguageSpec` for `MathLabel`.
///
/// `leaves` is the full pool of leaf labels (variables and constants).
/// Variable symbols are those of the form `MathLabel::Symbol(_)`.
/// If `None`, defaults to `[x, y, 0, 1, 2]`.
#[must_use]
pub fn math_spec() -> LanguageSpec<MathLabel> {
    let leaves = default_symbols();
    let variables = leaves
        .iter()
        .filter(|l| matches!(l, MathLabel::Symbol(_)))
        .copied()
        .collect();
    LanguageSpec {
        leaves,
        variables,
        unary_ops: UNARY_OPS.to_vec(),
        normal_binary_ops: NORMAL_BINARY_OPS.to_vec(),
        binder_ops: BINDER_OPS.to_vec(),
    }
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
    use egg::StopReason;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::egg::BoltzmannSampler;
    use crate::egg::generate::{binders_valid, free_vars};
    use crate::tree::{TreeShaped, TypedTree};

    #[test]
    fn sampler_produces_trees_near_target() {
        let sampler = BoltzmannSampler::new(15, 5, math_spec());
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let trees = sampler.sample_many(&mut rng, 50, &|_| Some(StopReason::Other(String::new())));
        assert_eq!(trees.len(), 50);
        for tree in &trees {
            let size = tree.0.size_without_types();
            assert!(
                (10..=20).contains(&size),
                "size {size} out of range [10, 20]"
            );
        }
    }

    #[test]
    fn small_target_size() {
        let sampler = BoltzmannSampler::new(5, 2, math_spec());
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let trees = sampler.sample_many(&mut rng, 30, &|_| Some(StopReason::Other(String::new())));
        for tree in &trees {
            let size = tree.0.size_without_types();
            assert!((3..=7).contains(&size), "size {size} out of range [3, 7]");
        }
    }

    #[test]
    fn binders_have_valid_bound_variables() {
        let sampler = BoltzmannSampler::new(15, 5, math_spec());
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        let spec = math_spec();
        let trees = sampler.sample_many(&mut rng, 100, &|_| Some(StopReason::Other(String::new())));
        for tree in &trees {
            assert!(
                binders_valid(&tree.0, &spec),
                "tree has binder with non-free bound variable: {tree:?}"
            );
        }
    }

    #[test]
    fn binder_child1_is_always_a_variable() {
        let sampler = BoltzmannSampler::new(15, 5, math_spec());
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        let trees = sampler.sample_many(&mut rng, 100, &|_| Some(StopReason::Other(String::new())));
        for tree in &trees {
            assert_no_constant_binder(&tree.0);
        }
    }

    fn assert_no_constant_binder(tree: &TypedTree<MathLabel>) {
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

    fn sym(name: &str) -> TypedTree<MathLabel> {
        TypedTree::leaf_untyped(MathLabel::Symbol(name.into()))
    }

    fn diff(expr: TypedTree<MathLabel>, var: TypedTree<MathLabel>) -> TypedTree<MathLabel> {
        TypedTree::new_untyped(MathLabel::Diff, vec![expr, var])
    }

    fn add(l: TypedTree<MathLabel>, r: TypedTree<MathLabel>) -> TypedTree<MathLabel> {
        TypedTree::new_untyped(MathLabel::Add, vec![l, r])
    }

    // --- free_vars ---

    #[test]
    fn free_vars_single_symbol() {
        let spec = math_spec();
        let tree = sym("x");
        let fv = free_vars(&tree, &spec);
        assert_eq!(fv, [MathLabel::Symbol("x".into())].into());
    }

    #[test]
    fn free_vars_binder_removes_bound_var() {
        let spec = math_spec();
        let tree = diff(sym("x"), sym("x"));
        assert!(free_vars(&tree, &spec).is_empty());
    }

    #[test]
    fn free_vars_binder_keeps_other_vars() {
        let spec = math_spec();
        let tree = diff(add(sym("x"), sym("y")), sym("x"));
        let fv = free_vars(&tree, &spec);
        assert!(
            !fv.contains(&MathLabel::Symbol("x".into())),
            "x should be bound"
        );
        assert!(
            fv.contains(&MathLabel::Symbol("y".into())),
            "y should be free"
        );
    }

    #[test]
    fn free_vars_nested_binders() {
        let spec = math_spec();
        let inner = diff(add(sym("x"), sym("y")), sym("x"));
        let tree = diff(inner, sym("y"));
        assert!(free_vars(&tree, &spec).is_empty());
    }

    #[test]
    fn free_vars_no_binders_collects_all_symbols() {
        let spec = math_spec();
        let tree = add(sym("x"), sym("y"));
        let fv = free_vars(&tree, &spec);
        assert_eq!(
            fv,
            [MathLabel::Symbol("x".into()), MathLabel::Symbol("y".into())].into()
        );
    }

    // --- binders_valid ---

    #[test]
    fn binders_valid_simple_valid() {
        let spec = math_spec();
        assert!(binders_valid(&diff(sym("x"), sym("x")), &spec));
    }

    #[test]
    fn binders_valid_simple_invalid() {
        let spec = math_spec();
        assert!(!binders_valid(&diff(sym("x"), sym("y")), &spec));
    }

    #[test]
    fn binders_valid_no_binders() {
        let spec = math_spec();
        assert!(binders_valid(&add(sym("x"), sym("y")), &spec));
    }

    #[test]
    fn binders_valid_nested_valid() {
        let spec = math_spec();
        let inner = diff(add(sym("x"), sym("y")), sym("x"));
        let tree = diff(inner, sym("y"));
        assert!(binders_valid(&tree, &spec));
    }

    #[test]
    fn binders_valid_nested_outer_invalid() {
        let spec = math_spec();
        let inner = diff(sym("x"), sym("x"));
        let tree = diff(inner, sym("y"));
        assert!(!binders_valid(&tree, &spec));
    }

    #[test]
    fn sample_many_count_zero() {
        let sampler = BoltzmannSampler::new(10, 3, math_spec());
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let trees = sampler.sample_many(&mut rng, 0, &|_| Some(StopReason::Other(String::new())));
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
