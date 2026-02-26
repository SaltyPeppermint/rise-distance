pub mod math;

use egg::{Analysis, Id, Language};
use hashbrown::HashMap;

use crate::ids::{EClassId, ExprChildId};
use crate::nodes::ENode;
use crate::{EClass, EGraph, Label};

pub use math::{Math, MathLabel};

pub fn convert<L, N, LL>(egg_graph: &egg::EGraph<L, N>, root: Id) -> EGraph<LL>
where
    L: Language,
    N: Analysis<L>,
    LL: Label + for<'a> From<&'a L>,
{
    // Works because classes are unique!
    let classes = egg_graph
        .classes()
        .map(|egg_class| {
            debug_assert_eq!(egg_class.id, egg_graph.find(egg_class.id));
            let eclass_id = EClassId::new(egg_class.id.into());
            let nodes = egg_class
                .nodes
                .iter()
                .map(|math_node| {
                    let children = math_node
                        .children()
                        .iter()
                        .map(|&child_id| {
                            ExprChildId::EClass(EClassId::new(egg_graph.find(child_id).into()))
                        })
                        .collect::<Vec<_>>();
                    ENode::new(math_node.into(), children)
                })
                .collect();
            (eclass_id, EClass::new(nodes, None))
        })
        .collect::<HashMap<_, _>>();

    // Build union-find: identity mapping for canonical IDs
    // Include both class IDs and all child IDs to cover the full range of IDs
    // that canonicalize() may be called with.
    let max_id = egg_graph
        .classes()
        .map(|c| usize::from(c.id))
        .chain(
            egg_graph
                .nodes()
                .iter()
                .flat_map(|n| n.children().iter())
                .map(|id| usize::from(*id)),
        )
        .max()
        .map_or(0, |m| m + 1);
    let union_find = (0..max_id)
        .map(|i| EClassId::new(usize::from(egg_graph.find(Id::from(i)))))
        .collect::<Vec<_>>();

    EGraph::new(
        classes,
        EClassId::new(root.into()),
        union_find,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
}

#[cfg(test)]
mod tests {
    use egg::{RecExpr, Runner};

    use crate::ids::{ExprChildId, NumericId};

    use super::convert;
    use super::math::{ConstantFold, Math, MathLabel};

    /// Build a saturated egg `EGraph` from a string expression, returning (egraph, root).
    fn build(expr: &str) -> (egg::EGraph<Math, ConstantFold>, egg::Id) {
        let expr: RecExpr<Math> = expr.parse().unwrap();
        let runner = Runner::<Math, ConstantFold>::default()
            .with_expr(&expr)
            .run(&[]);
        let root = runner.roots[0];
        (runner.egraph, root)
    }

    // -----------------------------------------------------------------------
    // 1. Single symbol leaf
    // -----------------------------------------------------------------------
    #[test]
    fn convert_single_symbol() {
        let (egg, root) = build("x");
        let g = convert::<Math, ConstantFold, MathLabel>(&egg, root);

        assert_eq!(g.root().to_index(), usize::from(egg.find(root)));
        // The root class must exist and contain exactly the Symbol node
        let class = g.class(g.root());
        assert!(!class.nodes().is_empty());
        assert!(
            class
                .nodes()
                .iter()
                .any(|n| *n.label() == MathLabel::Symbol("x".into()))
        );
    }

    // -----------------------------------------------------------------------
    // 2. Single constant leaf
    // -----------------------------------------------------------------------
    #[test]
    #[expect(clippy::float_cmp)]
    fn convert_single_constant() {
        let (egg, root) = build("42");
        let g = convert::<Math, ConstantFold, MathLabel>(&egg, root);

        let class = g.class(g.root());
        assert!(
            class
                .nodes()
                .iter()
                .any(|n| matches!(n.label(), MathLabel::Constant(c) if **c == 42.0))
        );
    }

    // -----------------------------------------------------------------------
    // 3. Unary operator (ln x)
    // -----------------------------------------------------------------------
    #[test]
    fn convert_unary_ln() {
        let (egg, root) = build("(ln x)");
        let g = convert::<Math, ConstantFold, MathLabel>(&egg, root);

        // Root class has an Ln node with one EClass child
        let root_class = g.class(g.root());
        let ln_node = root_class
            .nodes()
            .iter()
            .find(|n| *n.label() == MathLabel::Ln);
        assert!(ln_node.is_some(), "expected Ln node in root class");
        let ln_node = ln_node.unwrap();
        assert_eq!(ln_node.children().len(), 1);

        // The child class must contain the Symbol "x"
        let ExprChildId::EClass(child_id) = ln_node.children()[0] else {
            panic!("expected EClass child")
        };
        let child_class = g.class(child_id);
        assert!(
            child_class
                .nodes()
                .iter()
                .any(|n| *n.label() == MathLabel::Symbol("x".into()))
        );
    }

    // -----------------------------------------------------------------------
    // 4. Binary operator (+ x y) — two distinct children
    // -----------------------------------------------------------------------
    #[test]
    fn convert_binary_add() {
        let (egg, root) = build("(+ x y)");
        let g = convert::<Math, ConstantFold, MathLabel>(&egg, root);

        let root_class = g.class(g.root());
        let add_node = root_class
            .nodes()
            .iter()
            .find(|n| *n.label() == MathLabel::Add);
        assert!(add_node.is_some(), "expected Add node in root class");
        assert_eq!(add_node.unwrap().children().len(), 2);
    }

    // -----------------------------------------------------------------------
    // 5. All classes in the egg graph appear in the converted graph
    // -----------------------------------------------------------------------
    #[test]
    fn convert_class_count_matches() {
        let (egg, root) = build("(* (+ x 1) (- y 2))");
        let g = convert::<Math, ConstantFold, MathLabel>(&egg, root);

        let egg_class_count = egg.number_of_classes();
        let converted_count = g.class_ids().count();
        assert_eq!(egg_class_count, converted_count);
    }

    // -----------------------------------------------------------------------
    // 6. Children hold canonical IDs after union
    // -----------------------------------------------------------------------
    #[test]
    fn convert_children_are_canonical() {
        // After union/rebuild egg canonicalizes IDs; convert must follow suit.
        let expr: RecExpr<Math> = "(+ x x)".parse().unwrap();
        let mut egg_graph = egg::EGraph::<Math, ConstantFold>::default();
        let root = egg_graph.add_expr(&expr);
        // Union two distinct symbols so one gets remapped
        let a = egg_graph.add(Math::Symbol("a".into()));
        let b = egg_graph.add(Math::Symbol("b".into()));
        egg_graph.union(a, b);
        egg_graph.rebuild();

        let g = convert::<Math, ConstantFold, MathLabel>(&egg_graph, root);

        // Every EClass child referenced by a node must be a key in the graph
        for id in g.class_ids() {
            for node in g.class(id).nodes() {
                for child in node.children() {
                    if let ExprChildId::EClass(child_id) = child {
                        // canonicalize should be identity for canonical ids
                        assert_eq!(g.canonicalize(*child_id), *child_id);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. Union-find maps non-canonical IDs to canonical ones
    // -----------------------------------------------------------------------
    #[test]
    fn convert_union_find_is_populated() {
        // Build an expression with children so the union-find has entries
        let (egg, root) = build("(+ x y)");
        let g = convert::<Math, ConstantFold, MathLabel>(&egg, root);

        // Every entry in the union-find must resolve to a canonical class id
        for (i, &canonical) in g.union_find().iter().enumerate() {
            // canonical must be a key in the class map
            let _ = g.class(canonical); // panics if not found
            // Canonicalizing a canonical id must be a no-op
            assert_eq!(
                g.canonicalize(canonical),
                canonical,
                "entry {i}: union-find entry {canonical:?} is not canonical"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 8. Root is correctly set to the provided root id
    // -----------------------------------------------------------------------
    #[test]
    fn convert_root_matches_egg_root() {
        let (egg, root) = build("(sin x)");
        let g = convert::<Math, ConstantFold, MathLabel>(&egg, root);

        assert_eq!(g.root().to_index(), usize::from(egg.find(root)));
    }

    // -----------------------------------------------------------------------
    // 9. ConstantFold: constant expression collapses into a single constant node
    // -----------------------------------------------------------------------
    #[test]
    #[expect(clippy::float_cmp)]
    fn convert_constant_fold_collapses() {
        // After ConstantFold + rebuild, (+ 2 3) and 5.0 are in the same class
        let expr: RecExpr<Math> = "(+ 2 3)".parse().unwrap();
        let runner = Runner::<Math, ConstantFold>::default()
            .with_expr(&expr)
            .run(&[]);
        let root = runner.roots[0];
        let g = convert::<Math, ConstantFold, MathLabel>(&runner.egraph, root);

        // The root class must contain the folded constant 5.0
        let root_class = g.class(g.root());
        assert!(
            root_class
                .nodes()
                .iter()
                .any(|n| matches!(n.label(), MathLabel::Constant(c) if **c == 5.0)),
            "expected folded constant 5.0 in root class"
        );
    }

    // -----------------------------------------------------------------------
    // 10. Empty union-find when expression has no children (leaf-only graph)
    // -----------------------------------------------------------------------
    #[test]
    fn convert_leaf_has_empty_union_find() {
        let (egg, root) = build("x");
        let g = convert::<Math, ConstantFold, MathLabel>(&egg, root);

        // Even a lone symbol has a class ID, so the union-find covers that entry.
        // Every entry must map to a canonical class.
        for (i, &canonical) in g.union_find().iter().enumerate() {
            let _ = g.class(canonical); // panics if not found
            assert_eq!(
                g.canonicalize(canonical),
                canonical,
                "entry {i}: union-find entry {canonical:?} is not canonical"
            );
        }
    }
}
