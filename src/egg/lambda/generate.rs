use std::collections::HashSet;
use std::sync::LazyLock;

use crate::egg::generate::{BinderOp, LanguageSpec};
use crate::egg::lambda::LambdaLabel;
use crate::tree::{TreeShaped, TypedTree};

const BINARY_OPS: &[LambdaLabel] = &[LambdaLabel::Add, LambdaLabel::Eq, LambdaLabel::App];

const TERNARY_OPS: &[LambdaLabel] = &[LambdaLabel::If];

const BINDERS: &[BinderOp<LambdaLabel>] = &[
    // Var: arity-1, child[0] must be a variable. No scope.
    BinderOp {
        op: LambdaLabel::Var,
        arity: 1,
        bound_slot: 0,
        scope_slots: &[],
    },
    // Lam: bound var at child[0], body at child[1].
    BinderOp {
        op: LambdaLabel::Lam,
        arity: 2,
        bound_slot: 0,
        scope_slots: &[1],
    },
    // Fix: same shape as Lam.
    BinderOp {
        op: LambdaLabel::Fix,
        arity: 2,
        bound_slot: 0,
        scope_slots: &[1],
    },
    // Let: bound var at child[0], body at child[2]. child[1] (the value `e`)
    // is *not* in scope.
    BinderOp {
        op: LambdaLabel::Let,
        arity: 3,
        bound_slot: 0,
        scope_slots: &[2],
    },
];

static VARIABLES: LazyLock<Vec<LambdaLabel>> = LazyLock::new(|| {
    vec![
        LambdaLabel::Symbol("x".into()),
        LambdaLabel::Symbol("y".into()),
        LambdaLabel::Symbol("z".into()),
    ]
});

static LEAVES: LazyLock<Vec<LambdaLabel>> = LazyLock::new(|| {
    let mut v = vec![
        LambdaLabel::Bool(true),
        LambdaLabel::Bool(false),
        LambdaLabel::Num(0),
        LambdaLabel::Num(1),
        LambdaLabel::Num(2),
    ];
    v.extend(VARIABLES.iter().copied());
    v
});

/// `LanguageSpec` for the lambda calculus.
///
/// Operator classification:
///   - **Leaves**: `Bool(true)`, `Bool(false)`, a few `Num`s, and some `Symbol`s.
///     Variables are the `Symbol(_)` subset.
///   - **Unary**: none directly — `Var` is modeled as a degenerate "binder" so
///     that the sampler forces its child to be a variable leaf. (`Var` has
///     `scope_slots = []`; lambda's validator does not require the bound
///     variable to appear free, so this is fine.)
///   - **Binary**: `Add`, `Eq`, `App`.
///   - **Binary binders**: `Lam` and `Fix` — bound slot is child[0], scope is
///     child[1]. Unlike math, lambda allows `(lam x 0)` where the bound variable
///     is unused, so the validator does not require it to appear free.
///   - **Ternary**: `If`.
///   - **Ternary binders**: `Let` — bound slot is child[0], scope is child[2].
///     child[1] (the value) is *not* in the scope of the bound variable.
#[derive(Default)]
pub struct LambdaSpec;

impl LanguageSpec for LambdaSpec {
    type Label = LambdaLabel;

    const NORMAL_OPS: &'static [&'static [LambdaLabel]] = &[&[], BINARY_OPS, TERNARY_OPS];
    const BINDERS: &'static [BinderOp<LambdaLabel>] = BINDERS;

    fn leaves(&self) -> &'static [LambdaLabel] {
        &LEAVES
    }

    fn variables(&self) -> &'static [LambdaLabel] {
        &VARIABLES
    }

    /// Lambda's structural validity check: every binder's `bound_slot` child
    /// must actually be a variable leaf. Unlike math, we do *not* require the
    /// bound variable to appear free in the scope — `(lam x 0)` is fine.
    fn is_valid_tree(&self, tree: &TypedTree<LambdaLabel>) -> bool {
        if let Some(b) = Self::BINDERS.iter().find(|b| &b.op == tree.label()) {
            let children = tree.children();
            if children.len() != b.arity {
                return false;
            }
            if !self.variables().contains(children[b.bound_slot].label()) {
                return false;
            }
        }
        tree.children().iter().all(|c| self.is_valid_tree(c))
    }

    /// Lambda's `free_vars`: variable references go through `(var x)` rather
    /// than appearing as bare `Symbol` leaves. A bare `Symbol` in a non-`Var`
    /// position is not a free occurrence; `(var x)` *is*. Binder bound slots
    /// are also bare `Symbol`s and don't count as free occurrences.
    fn free_vars(&self, tree: &TypedTree<LambdaLabel>) -> HashSet<LambdaLabel> {
        if matches!(tree.label(), LambdaLabel::Var) {
            let children = tree.children();
            if children.len() == 1
                && let LambdaLabel::Symbol(_) = children[0].label()
            {
                let mut set = HashSet::new();
                set.insert(*children[0].label());
                return set;
            }
            return HashSet::new();
        }
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
            return out;
        }
        tree.children()
            .iter()
            .flat_map(|c| self.free_vars(c))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use egg::StopReason;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::egg::FixPointSampler;

    fn sym(name: &str) -> TypedTree<LambdaLabel> {
        TypedTree::leaf_untyped(LambdaLabel::Symbol(name.into()))
    }

    fn var(s: &str) -> TypedTree<LambdaLabel> {
        TypedTree::new_untyped(LambdaLabel::Var, vec![sym(s)])
    }

    fn lam(v: &str, body: TypedTree<LambdaLabel>) -> TypedTree<LambdaLabel> {
        TypedTree::new_untyped(LambdaLabel::Lam, vec![sym(v), body])
    }

    fn let_(
        v: &str,
        e: TypedTree<LambdaLabel>,
        body: TypedTree<LambdaLabel>,
    ) -> TypedTree<LambdaLabel> {
        TypedTree::new_untyped(LambdaLabel::Let, vec![sym(v), e, body])
    }

    fn app(f: TypedTree<LambdaLabel>, x: TypedTree<LambdaLabel>) -> TypedTree<LambdaLabel> {
        TypedTree::new_untyped(LambdaLabel::App, vec![f, x])
    }

    #[test]
    fn sampler_produces_trees_near_target() {
        let sampler = FixPointSampler::new(15, 5, LambdaSpec);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let trees = sampler.sample_many(&mut rng, 30, &|_| Some(StopReason::Other(String::new())));
        assert_eq!(trees.len(), 30);
        for tree in &trees {
            let size = tree.0.size_without_types();
            assert!(
                (10..=20).contains(&size),
                "size {size} out of range [10, 20]"
            );
        }
    }

    #[test]
    fn binder_bound_slot_is_always_a_variable() {
        let sampler = FixPointSampler::new(15, 5, LambdaSpec);
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        let trees = sampler.sample_many(&mut rng, 50, &|_| Some(StopReason::Other(String::new())));
        for tree in &trees {
            assert_bound_slot_is_symbol(&tree.0);
        }
    }

    fn assert_bound_slot_is_symbol(tree: &TypedTree<LambdaLabel>) {
        match tree.label() {
            LambdaLabel::Var | LambdaLabel::Lam | LambdaLabel::Fix => {
                assert!(
                    matches!(tree.children()[0].label(), LambdaLabel::Symbol(_)),
                    "{:?} child[0] is not a Symbol: {:?}",
                    tree.label(),
                    tree.children()[0].label()
                );
            }
            LambdaLabel::Let => {
                assert!(
                    matches!(tree.children()[0].label(), LambdaLabel::Symbol(_)),
                    "Let child[0] is not a Symbol: {:?}",
                    tree.children()[0].label()
                );
            }
            _ => {}
        }
        for c in tree.children() {
            assert_bound_slot_is_symbol(c);
        }
    }

    #[test]
    fn lam_with_unused_var_is_valid() {
        // (lam x 0) — `x` doesn't appear in body, but lambda allows this.
        let spec = LambdaSpec;
        let tree = lam("x", TypedTree::leaf_untyped(LambdaLabel::Num(0)));
        assert!(spec.is_valid_tree(&tree));
    }

    #[test]
    fn lam_with_used_var_is_valid() {
        let spec = LambdaSpec;
        let tree = lam("x", var("x"));
        assert!(spec.is_valid_tree(&tree));
    }

    #[test]
    fn let_with_const_body_is_valid() {
        let spec = LambdaSpec;
        let tree = let_(
            "x",
            TypedTree::leaf_untyped(LambdaLabel::Num(1)),
            TypedTree::leaf_untyped(LambdaLabel::Num(2)),
        );
        assert!(spec.is_valid_tree(&tree));
    }

    #[test]
    fn var_with_non_symbol_is_invalid() {
        // (var 5) — child of Var must be a symbol.
        let spec = LambdaSpec;
        let tree = TypedTree::new_untyped(
            LambdaLabel::Var,
            vec![TypedTree::leaf_untyped(LambdaLabel::Num(5))],
        );
        assert!(!spec.is_valid_tree(&tree));
    }

    #[test]
    fn free_vars_lam_removes_bound() {
        // (lam x (+ (var x) (var y)))  =>  free vars in body = {y}
        let spec = LambdaSpec;
        let body = TypedTree::new_untyped(LambdaLabel::Add, vec![var("x"), var("y")]);
        let tree = lam("x", body);
        let fv = spec.free_vars(&tree);
        // free_vars returns the *labels* — for lambda's spec, vars are Symbols,
        // and `var(x)`'s `Symbol` child counts as the variable occurrence.
        assert!(
            !fv.contains(&LambdaLabel::Symbol("x".into())),
            "x should be bound"
        );
        assert!(
            fv.contains(&LambdaLabel::Symbol("y".into())),
            "y should be free"
        );
    }

    #[test]
    fn free_vars_let_value_is_outside_scope() {
        // (let x (var x) (var x))  =>  in `e` (child[1]), `x` is NOT bound;
        // in body (child[2]), `x` IS bound. So free vars = {x}.
        let spec = LambdaSpec;
        let tree = let_("x", var("x"), var("x"));
        let fv = spec.free_vars(&tree);
        assert!(
            fv.contains(&LambdaLabel::Symbol("x".into())),
            "x should be free (it's used in `e` which is outside scope)"
        );
    }

    #[test]
    fn free_vars_app_collects_all() {
        // (app (var x) (var y))
        let spec = LambdaSpec;
        let tree = app(var("x"), var("y"));
        let fv = spec.free_vars(&tree);
        assert!(fv.contains(&LambdaLabel::Symbol("x".into())));
        assert!(fv.contains(&LambdaLabel::Symbol("y".into())));
    }
}
