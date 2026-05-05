use std::fmt::{self, Display};
use std::hash::Hash;

use egg::{Analysis, EGraph, Language, RecExpr};
// use hashbrown::DefaultHashBuilder;
use serde::Serialize;

use crate::TypedTree;
use crate::graph::Graph;
use crate::ids::{AnyId, DataChildId, DataId, EClassId, FunId, NatId, TypeChildId};
use crate::nodes::Label;

use super::TreeShaped;

/// A node in a labeled, ordered tree.
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct OriginTree<L: Label> {
    label: L,
    ty: Option<Box<OriginTree<L>>>,
    children: Vec<OriginTree<L>>,
    origin: AnyId,
}

impl<L: Label> OriginTree<L> {
    /// Create a leaf node with no children.
    pub fn leaf_untyped(label: L, origin: AnyId) -> Self {
        // let cached_hash = compute_hash(&label, None, &[], origin);
        OriginTree {
            label,
            ty: None,
            children: Vec::new(),
            origin,
            // cached_hash,
        }
    }

    /// Create a leaf node with no children.
    pub fn leaf_typed(label: L, ty: Option<OriginTree<L>>, origin: AnyId) -> Self {
        // let cached_hash = compute_hash(&label, ty.as_ref(), &[], origin);
        OriginTree {
            label,
            ty: ty.map(Box::new),
            children: Vec::new(),
            origin,
            // cached_hash,
        }
    }

    /// Create a node with the given children.
    pub fn new_untyped(label: L, children: Vec<OriginTree<L>>, origin: AnyId) -> Self {
        // let cached_hash = compute_hash(&label, None, &children, origin);
        OriginTree {
            label,
            ty: None,
            children,
            origin,
            // cached_hash,
        }
    }

    /// Create a node with the given children.
    pub fn new_typed(
        label: L,
        children: Vec<OriginTree<L>>,
        ty: Option<OriginTree<L>>,
        origin: AnyId,
    ) -> Self {
        // let cached_hash = compute_hash(&label, ty.as_ref(), &children, origin);
        OriginTree {
            label,
            ty: ty.map(Box::new),
            children,
            origin,
            // cached_hash,
        }
    }

    /// Build a type tree from an e-class's type annotation.
    #[must_use]
    pub fn from_eclass(graph: &Graph<L>, id: EClassId) -> Option<Self> {
        let ty_id = graph.class(id).ty()?;
        Some(Self::from_type(graph, *ty_id))
    }

    fn from_type(graph: &Graph<L>, id: TypeChildId) -> Self {
        match id {
            TypeChildId::Nat(nat_id) => Self::from_nat(graph, nat_id),
            TypeChildId::Type(fun_ty_id) => Self::from_fun(graph, fun_ty_id),
            TypeChildId::Data(data_ty_id) => Self::from_data(graph, data_ty_id),
        }
    }

    fn from_fun(graph: &Graph<L>, id: FunId) -> Self {
        let node = graph.fun_ty(id).label().to_owned();
        let children = graph
            .fun_ty(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_type(graph, c_id))
            .collect();
        OriginTree::new_untyped(node, children, id.into())
    }

    #[must_use]
    pub fn from_data(graph: &Graph<L>, id: DataId) -> Self {
        let node = graph.data_ty(id).label().to_owned();
        let children = graph
            .data_ty(id)
            .children()
            .iter()
            .map(|&c_id| match c_id {
                DataChildId::Nat(nat_id) => Self::from_nat(graph, nat_id),
                DataChildId::DataType(data_ty_id) => Self::from_data(graph, data_ty_id),
            })
            .collect();
        OriginTree::new_untyped(node, children, id.into())
    }

    #[must_use]
    pub fn from_nat(graph: &Graph<L>, id: NatId) -> Self {
        let node = graph.nat(id).label().to_owned();
        let children = graph
            .nat(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_nat(graph, c_id))
            .collect();
        OriginTree::new_untyped(node, children, id.into())
    }

    pub fn origin(&self) -> AnyId {
        self.origin
    }

    #[expect(clippy::missing_panics_doc)]
    pub fn from_recexpr<LL: Language + Into<L>, N: Analysis<LL>>(
        eg: &EGraph<LL, N>,
        rec_expr: &RecExpr<LL>,
    ) -> Self {
        let root = rec_expr.root();
        let label = rec_expr[root].clone().into();
        let origin = eg.find(eg.lookup_expr(rec_expr).unwrap()).into();
        let children = rec_expr[root]
            .children()
            .iter()
            .map(|c_id| {
                let child_rec_expr =
                    RecExpr::from(rec_expr.as_ref()[0..=usize::from(*c_id)].to_owned());
                Self::from_recexpr(eg, &child_rec_expr)
            })
            .collect();
        Self::new_typed(label, children, None, origin)
    }
}

impl<L: Label> TreeShaped<L> for OriginTree<L> {
    /// Returns true if this node has no children.
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn children(&self) -> &[Self] {
        &self.children
    }

    fn label(&self) -> &L {
        &self.label
    }

    fn ty(&self) -> Option<&Self> {
        self.ty.as_deref()
    }
}

impl<L: Label + Display> Display for OriginTree<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ty.is_some() {
            write!(f, "({} ", L::type_of())?;
        }
        if self.is_leaf() {
            write!(f, "{}", self.label)?;
        } else {
            write!(f, "({}", self.label)?;
            for child in &self.children {
                write!(f, " {child}")?;
            }
            write!(f, ")")?;
        }
        if let Some(ty) = &self.ty {
            write!(f, " {ty})")?;
        }
        Ok(())
    }
}

impl<L: Label> From<OriginTree<L>> for TypedTree<L> {
    fn from(value: OriginTree<L>) -> Self {
        TypedTree {
            label: value.label,
            ty: value.ty.map(|t| Box::new((*t).into())),
            children: value.children.into_iter().map(|x| x.into()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::{EGraph, RecExpr};

    use crate::egg::math::{ConstantFold, Math, RULES};
    use crate::ids::EClassId;

    fn origin_eclass(tree: &OriginTree<Math>) -> EClassId {
        match tree.origin {
            AnyId::EClass(id) => id,
            other => panic!("expected EClass origin, got {other:?}"),
        }
    }

    fn assert_origin_matches(
        tree: &OriginTree<Math>,
        eg: &EGraph<Math, ConstantFold>,
        expected_recexpr: &str,
    ) {
        let expr: RecExpr<Math> = expected_recexpr.parse().unwrap();
        let expected = EClassId::new(usize::from(eg.find(eg.lookup_expr(&expr).unwrap())));
        assert_eq!(
            origin_eclass(tree),
            expected,
            "origin mismatch for {expected_recexpr}"
        );
    }

    #[test]
    fn binary_add_distinct_children() {
        // Regression test for the off-by-one in the slice bound:
        // with [0..c_id] both children resolved to the same node.
        let mut eg = EGraph::<Math, ConstantFold>::default();
        let expr: RecExpr<Math> = "(+ x y)".parse().unwrap();
        eg.add_expr(&expr);
        eg.rebuild();

        let tree = OriginTree::from_recexpr(&eg, &expr);
        assert!(matches!(tree.label, Math::Add(_)));
        assert_eq!(tree.children.len(), 2);
        assert_eq!(tree.children[0].label, Math::Symbol("x".into()));
        assert_eq!(tree.children[1].label, Math::Symbol("y".into()));
        assert_origin_matches(&tree, &eg, "(+ x y)");
        assert_origin_matches(&tree.children[0], &eg, "x");
        assert_origin_matches(&tree.children[1], &eg, "y");
    }

    #[test]
    fn deeply_nested() {
        let mut eg = EGraph::<Math, ConstantFold>::default();
        let expr: RecExpr<Math> = "(d x (+ 1 (* 2 x)))".parse().unwrap();
        eg.add_expr(&expr);
        eg.rebuild();

        let tree = OriginTree::from_recexpr(&eg, &expr);
        assert!(matches!(tree.label, Math::Diff(_)));
        assert_eq!(tree.children.len(), 2);
        assert_eq!(tree.children[0].label, Math::Symbol("x".into()));

        let add = &tree.children[1];
        assert!(matches!(add.label, Math::Add(_)));
        assert_eq!(add.children.len(), 2);
        assert_eq!(add.children[0].label, Math::Constant("1".parse().unwrap()));

        let mul = &add.children[1];
        assert!(matches!(mul.label, Math::Mul(_)));
        assert_eq!(mul.children.len(), 2);
        assert_eq!(mul.children[0].label, Math::Constant("2".parse().unwrap()));
        assert_eq!(mul.children[1].label, Math::Symbol("x".into()));

        assert_origin_matches(&tree, &eg, "(d x (+ 1 (* 2 x)))");
        assert_origin_matches(add, &eg, "(+ 1 (* 2 x))");
        assert_origin_matches(mul, &eg, "(* 2 x)");
        assert_origin_matches(&mul.children[1], &eg, "x");
        // Both `x` occurrences should map to the same e-class
        assert_eq!(
            origin_eclass(&tree.children[0]),
            origin_eclass(&mul.children[1])
        );
    }

    #[test]
    fn shared_subexpr_origins_match_after_union() {
        // After running rules that make (+ x 0) ≡ x, both subtrees should
        // resolve to the same canonical e-class via `eg.find`.
        let mut eg = EGraph::<Math, ConstantFold>::default();
        let expr: RecExpr<Math> = "(+ (+ x 0) y)".parse().unwrap();
        eg.add_expr(&expr);
        let lhs = eg.add_expr(&"(+ x 0)".parse().unwrap());
        let rhs = eg.add_expr(&"x".parse().unwrap());
        eg.union(lhs, rhs);
        eg.rebuild();

        let tree = OriginTree::from_recexpr(&eg, &expr);
        let inner_add = &tree.children[0];
        let x_canon = EClassId::new(usize::from(
            eg.find(eg.lookup_expr(&"x".parse().unwrap()).unwrap()),
        ));
        assert_eq!(origin_eclass(inner_add), x_canon);
    }

    #[test]
    fn origins_canonical_after_eqsat() {
        // Run eqsat for 3 iterations so the `zero-add` rule unions (+ x 0) with x.
        // Both should map to the same canonical e-class once we lift via from_recexpr.
        let expr: RecExpr<Math> = "(+ (+ x 0) y)".parse().unwrap();
        let runner = egg::Runner::<Math, ConstantFold>::default()
            .with_iter_limit(3)
            .with_scheduler(egg::SimpleScheduler)
            .with_expr(&expr)
            .run(&*RULES);
        let eg = &runner.egraph;

        // Sanity-check: eqsat actually unioned (+ x 0) with x.
        assert_eq!(
            eg.find(eg.lookup_expr(&"(+ x 0)".parse().unwrap()).unwrap()),
            eg.find(eg.lookup_expr(&"x".parse().unwrap()).unwrap()),
        );

        let tree = OriginTree::from_recexpr(eg, &expr);
        let inner_add = &tree.children[0];
        let leaf_x = &inner_add.children[0];

        // The (+ x 0) subtree's origin should be the same canonical class as x's.
        assert_eq!(origin_eclass(inner_add), origin_eclass(leaf_x));
        assert_origin_matches(inner_add, eg, "x");
        assert_origin_matches(leaf_x, eg, "x");

        // The outer (+ ... y) is not unioned with anything, so it stays distinct.
        assert_ne!(origin_eclass(&tree), origin_eclass(inner_add));
    }
}
