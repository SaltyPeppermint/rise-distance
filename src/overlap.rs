use crate::ids::{EClassId, ExprChildId};
use crate::nodes::Label;
use crate::{EGraph, PartialChild, PartialTree, TreeNode, tree_node_to_partial};

/// Match a reference tree against an e-class, producing a partial tree
/// that maximizes structural overlap with the reference.
///
/// At each e-class, finds e-nodes whose label matches the `ref_tree`'s label.
/// If multiple match, tries all and picks the one with the largest
/// `resolved_count`. Returns `None` if no e-node matches (caller creates a Hole).
pub(crate) fn match_ref_tree<L: Label>(
    graph: &EGraph<L>,
    eclass_id: EClassId,
    ref_tree: &TreeNode<L>,
) -> Option<PartialTree<L>> {
    let canonical_id = graph.canonicalize(eclass_id);
    let eclass = graph.class(canonical_id);
    let ty = Some(TreeNode::<L>::from_eclass(graph, canonical_id));

    let mut best = None;

    for node in eclass
        .nodes()
        .iter()
        .filter(|node| node.label() == ref_tree.label())
    {
        let children = node.children();
        let ref_children = ref_tree.children();

        let mut partial_children = Vec::with_capacity(children.len());
        let mut ref_idx = 0;

        for &child_id in children {
            match child_id {
                ExprChildId::Nat(nat_id) => {
                    let nat_tree = TreeNode::<L>::from_nat(graph, nat_id);
                    partial_children.push(PartialChild::Resolved(tree_node_to_partial(&nat_tree)));
                    ref_idx += 1;
                }
                ExprChildId::Data(data_id) => {
                    let data_tree = TreeNode::<L>::from_data(graph, data_id);
                    partial_children.push(PartialChild::Resolved(tree_node_to_partial(&data_tree)));
                    ref_idx += 1;
                }
                ExprChildId::EClass(child_eclass_id) => {
                    if ref_idx < ref_children.len() {
                        match match_ref_tree(graph, child_eclass_id, &ref_children[ref_idx]) {
                            Some(pt) => {
                                partial_children.push(PartialChild::Resolved(pt));
                            }
                            None => {
                                partial_children
                                    .push(PartialChild::Hole(graph.canonicalize(child_eclass_id)));
                            }
                        }
                    } else {
                        partial_children
                            .push(PartialChild::Hole(graph.canonicalize(child_eclass_id)));
                    }
                    ref_idx += 1;
                }
            }
        }

        let pt = PartialTree::new(node.label().clone(), ty.clone(), partial_children);
        let overlap = pt.resolved_count();

        if best
            .as_ref()
            .is_none_or(|(_, best_overlap)| overlap > *best_overlap)
        {
            best = Some((pt, overlap));
        }
    }

    best.map(|(pt, _)| pt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EGraph;
    use crate::TreeNode;
    use crate::graph::EClass;
    use crate::nodes::ENode;
    use crate::overlap::match_ref_tree;
    use crate::test_utils::*;
    use hashbrown::HashMap;

    #[test]
    fn match_ref_tree_exact() {
        // Class 0: f(class1)
        // Class 1: leaf "a"
        // ref_tree: (f a) — should match exactly with no holes
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        let partial = match_ref_tree(&graph, EClassId::new(0), &ref_tree).unwrap();
        assert_eq!(partial.resolved_count(), 2); // f + a
        assert!(partial.holes().is_empty());
        assert_eq!(partial.fixed_size(false), 2);
    }

    #[test]
    fn match_ref_tree_partial_hole() {
        // Class 0: f(class1)
        // Class 1: leaf "a", leaf "b"
        // ref_tree: (f c) — "f" matches at root, "c" does NOT match class1
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let ref_tree: TreeNode<String> = "(f c)".parse().unwrap();

        let partial = match_ref_tree(&graph, EClassId::new(0), &ref_tree).unwrap();
        assert_eq!(partial.resolved_count(), 1); // only f
        assert_eq!(partial.holes().len(), 1);
        assert_eq!(partial.holes()[0], EClassId::new(1));
    }

    #[test]
    fn match_ref_tree_no_match_at_root() {
        // Class 0: leaf "a"
        // ref_tree: "b" — no match
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let ref_tree: TreeNode<String> = "b".parse().unwrap();

        assert!(match_ref_tree(&graph, EClassId::new(0), &ref_tree).is_none());
    }

    #[test]
    fn match_ref_tree_best_overlap() {
        // Class 0: two nodes both labeled "f", pointing to class1 and class2 resp.
        // Class 1: leaf "a"
        // Class 2: leaf "b"
        // ref_tree: (f a) — both f-nodes match at root, but only the one
        //   pointing to class1 can match child "a"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![
                        ENode::new("f".to_owned(), vec![eid(1)]),
                        ENode::new("f".to_owned(), vec![eid(2)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        let partial = match_ref_tree(&graph, EClassId::new(0), &ref_tree).unwrap();
        // Should pick the f->class1 node which fully matches (f a)
        assert_eq!(partial.resolved_count(), 2);
        assert!(partial.holes().is_empty());
    }
}
