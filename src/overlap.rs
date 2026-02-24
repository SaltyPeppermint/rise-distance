use hashbrown::{HashMap, HashSet};

use crate::graph::EClass;
use crate::ids::{EClassId, ExprChildId};
use crate::nodes::{ENode, Label};
use crate::{EGraph, PartialChild, PartialTree, TreeNode, tree_node_to_partial};

/// Match a reference tree against an e-class, producing a partial tree
/// that maximizes structural overlap with the reference.
///
/// At each e-class, finds e-nodes whose label matches the `ref_tree`'s label.
/// If multiple match, tries all and picks the one with the largest
/// `resolved_count`. Returns `None` if no e-node matches (caller creates a Hole).
pub fn match_ref_tree<L: Label>(
    graph: &EGraph<L>,
    eclass_id: EClassId,
    ref_tree: &TreeNode<L>,
) -> Option<PartialTree<L>> {
    let canonical_id = graph.canonicalize(eclass_id);
    let eclass = graph.class(canonical_id);
    let ty = Some(TreeNode::from_eclass(graph, canonical_id));

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
                    let nat_tree = TreeNode::from_nat(graph, nat_id);
                    partial_children.push(PartialChild::Resolved(tree_node_to_partial(&nat_tree)));
                    ref_idx += 1;
                }
                ExprChildId::Data(data_id) => {
                    let data_tree = TreeNode::from_data(graph, data_id);
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

/// Walk the e-graph guided by `ref_tree`, recording which single e-node was
/// chosen (best overlap) at each matched e-class.
///
/// Returns a map from canonical `EClassId` -> the chosen `ENode`.
/// Classes where no e-node label matched are absent from the map (they are holes).
fn collect_matched_nodes<'a, L: Label>(
    graph: &'a EGraph<L>,
    eclass_id: EClassId,
    ref_tree: &TreeNode<L>,
) -> HashMap<EClassId, &'a ENode<L>> {
    let canonical_id = graph.canonicalize(eclass_id);
    let eclass = graph.class(canonical_id);

    let mut best = None;

    for node in eclass
        .nodes()
        .iter()
        .filter(|node| node.label() == ref_tree.label())
    {
        let children = node.children();
        let ref_children = ref_tree.children();

        let mut child_matches = HashMap::new();
        let mut resolved = 1usize; // count this node itself
        let mut ref_idx = 0;

        for &child_id in children {
            if let ExprChildId::EClass(child_eclass_id) = child_id {
                if ref_idx < ref_children.len() {
                    let sub = collect_matched_nodes(graph, child_eclass_id, &ref_children[ref_idx]);
                    // If the child class itself is in sub, that means it matched
                    let child_canonical = graph.canonicalize(child_eclass_id);
                    if sub.contains_key(&child_canonical) {
                        resolved += sub.len();
                    }
                    child_matches.extend(sub);
                }
                ref_idx += 1;
            } else {
                resolved += 1;
                ref_idx += 1;
            }
        }

        if best
            .as_ref()
            .is_none_or(|(_, best_resolved, _)| resolved > *best_resolved)
        {
            best = Some((node, resolved, child_matches));
        }
    }

    if let Some((chosen_node, _, mut child_matches)) = best {
        child_matches.insert(canonical_id, chosen_node);
        child_matches
    } else {
        HashMap::new()
    }
}

/// Collect all e-class IDs transitively reachable from a set of root classes,
/// following all e-node children in the original graph.
/// Essentially a transitive hull
fn collect_reachable<L: Label>(
    graph: &EGraph<L>,
    roots: impl IntoIterator<Item = EClassId>,
) -> HashSet<EClassId> {
    let mut visited = HashSet::new();
    let mut stack = roots.into_iter().collect::<Vec<_>>();

    while let Some(id) = stack.pop() {
        let canonical = graph.canonicalize(id);
        if !visited.insert(canonical) {
            continue;
        }
        for node in graph.class(canonical).nodes() {
            for &child_id in node.children() {
                if let ExprChildId::EClass(child) = child_id {
                    let child_canonical = graph.canonicalize(child);
                    if !visited.contains(&child_canonical) {
                        stack.push(child_canonical);
                    }
                }
            }
        }
    }

    visited
}

/// Prune an e-graph based on overlap with a reference tree.
///
/// For each e-class along the matched path, keeps only the single best-matching
/// e-node (removing all siblings). For e-classes reachable from holes (where no
/// label matched), all nodes are preserved along with their transitive children.
///
/// Returns the pruned e-graph and a `Vec` of `EClassId`s whose e-classes had
/// nodes removed.
pub fn prune_by_ref_tree<L: Label>(
    graph: &EGraph<L>,
    root: EClassId,
    ref_tree: &TreeNode<L>,
) -> (EGraph<L>, HashSet<EClassId>) {
    let matched = collect_matched_nodes(graph, root, ref_tree);

    // Find hole e-classes: matched e-nodes whose EClass children were NOT matched
    let mut hole_roots = Vec::new();
    for (_, chosen_node) in &matched {
        for &child_id in chosen_node.children() {
            if let ExprChildId::EClass(child) = child_id {
                let child_canonical = graph.canonicalize(child);
                if !matched.contains_key(&child_canonical) {
                    hole_roots.push(child_canonical);
                }
            }
        }
    }

    // Everything reachable from holes must NOT be pruned
    let protected = collect_reachable(graph, hole_roots);

    let mut pruned_ids = HashSet::new();

    let new_classes = graph
        .class_ids()
        .map(|class_id| {
            let canonical = graph.canonicalize(class_id);
            let eclass = graph.class(canonical);

            if let Some(&chosen_node) = matched.get(&canonical) {
                if protected.contains(&canonical) {
                    // Protected by hole reachability -> keep all nodes
                    (canonical, eclass.clone())
                } else {
                    // Pruned -> keep only the chosen node
                    if eclass.nodes().len() > 1 {
                        pruned_ids.insert(canonical);
                    }
                    (
                        canonical,
                        EClass::new(vec![chosen_node.to_owned()], eclass.ty()),
                    )
                }
            } else {
                // Not on the matched path -> copy as-is
                (canonical, eclass.clone())
            }
        })
        .collect();

    // TODO: fun_ty_nodes and data_ty_nodes could also be pruned to remove unreferenced entries
    let pruned_graph = EGraph::new(
        new_classes,
        graph.root(),
        graph.union_find().to_vec(),
        graph.fun_ty_nodes().clone(),
        graph.nat_nodes().clone(),
        graph.data_ty_nodes().clone(),
    );

    (pruned_graph, pruned_ids)
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
        // ref_tree: (f a) -> should match exactly with no holes
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
        // ref_tree: (f c) -> "f" matches at root, "c" does NOT match class1
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
        // ref_tree: "b" -> no match
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
        // ref_tree: (f a) -> both f-nodes match at root, but only the one
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

    #[test]
    fn prune_exact_match_removes_siblings() {
        // Class 0: f(class1), g(class1)  -> two nodes in one class
        // Class 1: leaf "a"
        // ref_tree: (f a) -> matches f, should prune g away
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![
                        ENode::new("f".to_owned(), vec![eid(1)]),
                        ENode::new("g".to_owned(), vec![eid(1)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();
        let (pruned, pruned_ids) = prune_by_ref_tree(&graph, EClassId::new(0), &ref_tree);

        // Class 0 was pruned (had 2 nodes, now 1)
        assert_eq!(pruned_ids, [EClassId::new(0)].into());
        assert_eq!(pruned.class(EClassId::new(0)).nodes().len(), 1);
        assert_eq!(pruned.class(EClassId::new(0)).nodes()[0].label(), "f");
        // Class 1 had only one node, unchanged
        assert_eq!(pruned.class(EClassId::new(1)).nodes().len(), 1);
    }

    #[test]
    fn prune_no_match_at_root_returns_identical() {
        // Class 0: leaf "a"
        // ref_tree: "b" -> no match at root, graph unchanged
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
        let (pruned, pruned_ids) = prune_by_ref_tree(&graph, EClassId::new(0), &ref_tree);

        assert!(pruned_ids.is_empty());
        assert_eq!(pruned.class(EClassId::new(0)).nodes().len(), 1);
    }

    #[test]
    fn prune_hole_protects_transitive_children() {
        // Class 0: f(class1)
        // Class 1: leaf "a", leaf "b"  -> two nodes
        // Class 2: g(class1)           -> also points to class1
        // ref_tree: (f c) -> f matches at root, but "c" doesn't match class1
        //   so class1 is a hole. Class1 should NOT be pruned despite having 2 nodes.
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
        let (pruned, pruned_ids) = prune_by_ref_tree(&graph, EClassId::new(0), &ref_tree);

        // Class 0 has only 1 node so not "pruned" (nothing removed)
        // Class 1 is a hole -> protected, keeps both nodes
        assert!(pruned_ids.is_empty());
        assert_eq!(pruned.class(EClassId::new(1)).nodes().len(), 2);
    }

    #[test]
    fn prune_hole_protects_deep_transitive_children() {
        // Class 0: f(class1)           -> matched
        // Class 1: leaf "a", g(class2) -> "a" doesn't match ref, g doesn't match ref
        //   so class1 is a hole
        // Class 2: leaf "x", leaf "y"  -> reachable from hole class1 via g
        // ref_tree: (f c) -> f matches, c doesn't match class1
        //   class1 is a hole, class2 is transitively reachable from the hole
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(
                    vec![
                        ENode::leaf("a".to_owned()),
                        ENode::new("g".to_owned(), vec![eid(2)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("x".to_owned()), ENode::leaf("y".to_owned())],
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
        let (pruned, pruned_ids) = prune_by_ref_tree(&graph, EClassId::new(0), &ref_tree);

        assert!(pruned_ids.is_empty());
        // Class 1 (hole) keeps both nodes
        assert_eq!(pruned.class(EClassId::new(1)).nodes().len(), 2);
        // Class 2 (transitively reachable from hole) keeps both nodes
        assert_eq!(pruned.class(EClassId::new(2)).nodes().len(), 2);
    }

    #[test]
    fn prune_best_overlap_picks_correct_node() {
        // Class 0: f(class1), f(class2) -> two f-nodes pointing to different classes
        // Class 1: leaf "a"
        // Class 2: leaf "b"
        // ref_tree: (f a) -> should pick f->class1 and prune f->class2
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
        let (pruned, pruned_ids) = prune_by_ref_tree(&graph, EClassId::new(0), &ref_tree);

        assert_eq!(pruned_ids, [EClassId::new(0)].into());
        let root_nodes = pruned.class(EClassId::new(0)).nodes();
        assert_eq!(root_nodes.len(), 1);
        assert_eq!(root_nodes[0].label(), "f");
        // The kept f-node should point to class1 (which has "a")
        assert_eq!(root_nodes[0].children(), &[eid(1)]);
    }

    #[test]
    fn prune_matched_class_reachable_from_hole_is_protected() {
        // Class 0: f(class1, class2)
        // Class 1: leaf "a", leaf "b"  -> matched by ref_tree (picks "a")
        // Class 2: g(class1)           -> not matched (hole), and points back to class1
        // ref_tree: (f a c) -> f matches, "a" matches in class1, "c" doesn't match class2
        //   class2 is a hole, and class1 is reachable from class2 via g
        //   so class1 should be PROTECTED (not pruned) even though it was matched
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::new("g".to_owned(), vec![eid(1)])], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let ref_tree: TreeNode<String> = "(f a c)".parse().unwrap();
        let (pruned, pruned_ids) = prune_by_ref_tree(&graph, EClassId::new(0), &ref_tree);

        // Class 1 was matched but is reachable from hole class2 -> protected
        assert!(pruned_ids.is_empty());
        assert_eq!(pruned.class(EClassId::new(1)).nodes().len(), 2);
    }
}
