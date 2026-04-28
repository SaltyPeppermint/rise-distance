use std::collections::HashSet;
use std::sync::LazyLock;

use ordered_float::NotNan;

use crate::egg::generate::{BinderOp, LanguageSpec};
use crate::egg::math::MathLabel;
use crate::tree::{TreeShaped, TypedTree};

const UNARY_OPS: &[MathLabel] = &[
    MathLabel::Ln,
    MathLabel::Sqrt,
    MathLabel::Sin,
    MathLabel::Cos,
];

const NORMAL_BINARY_OPS: &[MathLabel] = &[
    MathLabel::Add,
    MathLabel::Sub,
    MathLabel::Mul,
    MathLabel::Div,
    MathLabel::Pow,
];

const BINDERS: &[BinderOp<MathLabel>] = &[
    BinderOp {
        op: MathLabel::Diff,
        arity: 2,
        bound_slot: 1,
        scope_slots: &[0],
    },
    BinderOp {
        op: MathLabel::Integral,
        arity: 2,
        bound_slot: 1,
        scope_slots: &[0],
    },
];

static LEAVES: LazyLock<Vec<MathLabel>> = LazyLock::new(|| {
    vec![
        MathLabel::Symbol("x".into()),
        MathLabel::Symbol("y".into()),
        MathLabel::Constant(NotNan::new(0.0).unwrap()),
        MathLabel::Constant(NotNan::new(1.0).unwrap()),
        MathLabel::Constant(NotNan::new(2.0).unwrap()),
    ]
});

static VARIABLES: LazyLock<Vec<MathLabel>> = LazyLock::new(|| {
    LEAVES
        .iter()
        .filter(|l| matches!(l, MathLabel::Symbol(_)))
        .copied()
        .collect()
});

/// `LanguageSpec` for the math language.
///
/// Leaves are `[x, y, 0, 1, 2]`; variables are the `Symbol(_)` subset.
/// `Diff` and `Integral` are arity-2 binders with the bound variable in
/// child[1] and scope on child[0].
#[derive(Default)]
pub struct MathSpec;

impl LanguageSpec for MathSpec {
    type Label = MathLabel;

    const NORMAL_OPS: &'static [&'static [MathLabel]] = &[UNARY_OPS, NORMAL_BINARY_OPS];
    const BINDERS: &'static [BinderOp<MathLabel>] = BINDERS;

    fn leaves(&self) -> &'static [MathLabel] {
        &LEAVES
    }

    fn variables(&self) -> &'static [MathLabel] {
        &VARIABLES
    }

    /// Math validity: every binder's bound variable must appear free in at
    /// least one of its `scope_slots` children.
    fn is_valid_tree(&self, tree: &TypedTree<MathLabel>) -> bool {
        if let Some(b) = Self::BINDERS.iter().find(|b| &b.op == tree.label()) {
            let children = tree.children();
            if children.len() != b.arity {
                return false;
            }
            let bound = children[b.bound_slot].label();
            if !self.variables().contains(bound) {
                return false;
            }
            let scoped_has_bound = b
                .scope_slots
                .iter()
                .any(|&i| self.free_vars(&children[i]).contains(bound));
            if !scoped_has_bound {
                return false;
            }
            children.iter().all(|c| self.is_valid_tree(c))
        } else {
            tree.children().iter().all(|c| self.is_valid_tree(c))
        }
    }

    fn free_vars(&self, tree: &TypedTree<MathLabel>) -> HashSet<MathLabel> {
        if let Some(b) = Self::BINDERS.iter().find(|b| &b.op == tree.label()) {
            let children = tree.children();
            let bound = if children.len() == b.arity {
                let lbl = children[b.bound_slot].label();
                self.variables().contains(lbl).then_some(*lbl)
            } else {
                None
            };
            let mut out = HashSet::new();
            for (i, c) in children.iter().enumerate() {
                if i == b.bound_slot {
                    continue;
                }
                let mut fv = self.free_vars(c);
                if b.scope_slots.contains(&i)
                    && let Some(v) = bound
                {
                    fv.remove(&v);
                }
                out.extend(fv);
            }
            out
        } else if self.variables().contains(tree.label()) {
            let mut set = HashSet::new();
            set.insert(*tree.label());
            set
        } else {
            tree.children()
                .iter()
                .flat_map(|c| self.free_vars(c))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use egg::StopReason;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::egg::FixPointSampler;
    use crate::egg::generate::LanguageSpec;
    use crate::tree::{TreeShaped, TypedTree};

    #[test]
    fn sampler_produces_trees_near_target() {
        let sampler = FixPointSampler::new(15, 5, MathSpec);
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
        let sampler = FixPointSampler::new(5, 2, MathSpec);
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let trees = sampler.sample_many(&mut rng, 30, &|_| Some(StopReason::Other(String::new())));
        for tree in &trees {
            let size = tree.0.size_without_types();
            assert!((3..=7).contains(&size), "size {size} out of range [3, 7]");
        }
    }

    #[test]
    fn binders_have_valid_bound_variables() {
        let sampler = FixPointSampler::new(15, 5, MathSpec);
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        let spec = MathSpec;
        let trees = sampler.sample_many(&mut rng, 100, &|_| Some(StopReason::Other(String::new())));
        for tree in &trees {
            assert!(
                spec.is_valid_tree(&tree.0),
                "tree has binder with non-free bound variable: {tree:?}"
            );
        }
    }

    #[test]
    fn binder_child1_is_always_a_variable() {
        let sampler = FixPointSampler::new(15, 5, MathSpec);
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

    fn sym(name: &str) -> TypedTree<MathLabel> {
        TypedTree::leaf_untyped(MathLabel::Symbol(name.into()))
    }

    fn diff(expr: TypedTree<MathLabel>, var: TypedTree<MathLabel>) -> TypedTree<MathLabel> {
        TypedTree::new_untyped(MathLabel::Diff, vec![expr, var])
    }

    fn add(l: TypedTree<MathLabel>, r: TypedTree<MathLabel>) -> TypedTree<MathLabel> {
        TypedTree::new_untyped(MathLabel::Add, vec![l, r])
    }

    #[test]
    fn free_vars_single_symbol() {
        let spec = MathSpec;
        let tree = sym("x");
        let fv = spec.free_vars(&tree);
        assert_eq!(fv, [MathLabel::Symbol("x".into())].into());
    }

    #[test]
    fn free_vars_binder_removes_bound_var() {
        let spec = MathSpec;
        let tree = diff(sym("x"), sym("x"));
        assert!(spec.free_vars(&tree).is_empty());
    }

    #[test]
    fn free_vars_binder_keeps_other_vars() {
        let spec = MathSpec;
        let tree = diff(add(sym("x"), sym("y")), sym("x"));
        let fv = spec.free_vars(&tree);
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
        let spec = MathSpec;
        let inner = diff(add(sym("x"), sym("y")), sym("x"));
        let tree = diff(inner, sym("y"));
        assert!(spec.free_vars(&tree).is_empty());
    }

    #[test]
    fn free_vars_no_binders_collects_all_symbols() {
        let spec = MathSpec;
        let tree = add(sym("x"), sym("y"));
        let fv = spec.free_vars(&tree);
        assert_eq!(
            fv,
            [MathLabel::Symbol("x".into()), MathLabel::Symbol("y".into())].into()
        );
    }

    #[test]
    fn binders_valid_simple_valid() {
        let spec = MathSpec;
        assert!(spec.is_valid_tree(&diff(sym("x"), sym("x"))));
    }

    #[test]
    fn binders_valid_simple_invalid() {
        let spec = MathSpec;
        assert!(!spec.is_valid_tree(&diff(sym("x"), sym("y"))));
    }

    #[test]
    fn binders_valid_no_binders() {
        let spec = MathSpec;
        assert!(spec.is_valid_tree(&add(sym("x"), sym("y"))));
    }

    #[test]
    fn binders_valid_nested_valid() {
        let spec = MathSpec;
        let inner = diff(add(sym("x"), sym("y")), sym("x"));
        let tree = diff(inner, sym("y"));
        assert!(spec.is_valid_tree(&tree));
    }

    #[test]
    fn binders_valid_nested_outer_invalid() {
        let spec = MathSpec;
        let inner = diff(sym("x"), sym("x"));
        let tree = diff(inner, sym("y"));
        assert!(!spec.is_valid_tree(&tree));
    }

    #[test]
    fn sample_many_count_zero() {
        let sampler = FixPointSampler::new(10, 3, MathSpec);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let trees = sampler.sample_many(&mut rng, 0, &|_| Some(StopReason::Other(String::new())));
        assert!(trees.is_empty());
    }

    #[test]
    fn default_leaves_contains_expected() {
        let syms = MathSpec.leaves();
        assert!(syms.contains(&MathLabel::Symbol("x".into())));
        assert!(syms.contains(&MathLabel::Symbol("y".into())));
        assert!(syms.contains(&MathLabel::Constant(NotNan::new(0.0).unwrap())));
        assert!(syms.contains(&MathLabel::Constant(NotNan::new(1.0).unwrap())));
        assert!(syms.contains(&MathLabel::Constant(NotNan::new(2.0).unwrap())));
        assert_eq!(syms.len(), 5);
    }
}
