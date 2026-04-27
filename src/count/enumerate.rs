use dashmap::DashMap;
use egg::EGraph;
use egg::Id;
use egg::Language;
use indicatif::{ParallelProgressIterator, ProgressBar};
use rayon::prelude::*;

use crate::egg::TypeAnalysis;
use crate::tree::OriginExpr;

use super::Counter;
use super::TermCount;
// use crate::EGraph;
// use crate::OriginTree;
// use crate::ids::{EClassId, ExprChildId};
// use crate::nodes::LabelLanguage;

impl<C: Counter> TermCount<C> {
    /// Enumerate all terms from the root e-class with sizes in `1..=max_size`.
    #[must_use]
    pub fn enumerate_root<L: Language, N: TypeAnalysis<L>>(
        &self,
        graph: &EGraph<L, N>,
        max_size: usize,
        progress: Option<ProgressBar>,
    ) -> Vec<OriginTree<L>> {
        self.enumerate(graph, graph.root(), max_size, progress)
    }

    /// Enumerate all terms from an e-class with sizes in `1..=max_size`.
    #[must_use]
    #[expect(clippy::missing_panics_doc)]
    pub fn enumerate<L: Language, N: TypeAnalysis<L>>(
        &self,
        graph: &EGraph<L, N>,
        id: Id,
        max_size: usize,
        progress: Option<ProgressBar>,
    ) -> Vec<OriginExpr<L>> {
        let canon_id = graph.canonicalize(id);
        let sum = self
            .data
            .get(&canon_id)
            .unwrap()
            .values()
            .sum::<C>()
            .try_into()
            .unwrap();

        // Build (size, node_index) pairs so rayon can distribute work
        // across both sizes and nodes, not just sizes.
        let Some(histogram) = self.get(&canon_id) else {
            return Vec::new();
        };
        let eclass = graph.class(canon_id);
        let nodes = eclass.nodes();
        let type_overhead = self.type_overhead(eclass);
        let ty = OriginTree::from_eclass(graph, canon_id);

        let cache = DashMap::new();

        let work = (0..=max_size)
            .filter(|size| histogram.contains_key(size))
            .filter_map(|size| size.checked_sub(1 + type_overhead))
            .flat_map(|budget| (0..nodes.len()).map(move |ni| (budget, ni)))
            .collect::<Vec<_>>();

        let iter = work.into_par_iter().flat_map(|(child_budget, ni)| {
            let node = &nodes[ni];
            let children = node.children();

            self.enumerate_children(graph, children, child_budget, &cache)
                .par_bridge()
                .map(|child_combo| {
                    OriginTree::new_typed(
                        node.label().clone(),
                        child_combo,
                        ty.clone(),
                        canon_id.into(),
                    )
                })
        });

        if let Some(pb) = progress {
            pb.set_length(sum);
            iter.progress_with(pb).collect()
        } else {
            iter.collect()
        }
    }

    /// Enumerate all terms of exactly `size` from an e-class, using a shared cache.
    fn enumerate_class_cached<L: Language, N: TypeAnalysis<L>>(
        &self,
        graph: &EGraph<L>,
        id: Id,
        size: usize,
        cache: &DashMap<(Id, usize), Vec<OriginExpr<L>>>,
    ) -> Vec<OriginExpr<L>> {
        let canon_id = graph.canonicalize(id);
        let key = (canon_id, size);

        // Cache hit
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }

        let result = self.enumerate_class_inner(graph, canon_id, size, cache);
        cache.insert(key, result.clone());
        result
    }

    /// Inner logic for enumerating all terms of exactly `size` from a canonical e-class.
    fn enumerate_class_inner<L: Language, N: TypeAnalysis<L>>(
        &self,
        graph: &EGraph<L>,
        canon_id: Id,
        size: usize,
        cache: &DashMap<(Id, usize), Vec<OriginExpr<L>>>,
    ) -> Vec<OriginExpr<L>> {
        // Check if this class has any terms at this size
        let Some(histogram) = self.get(&canon_id) else {
            return Vec::new();
        };
        if !histogram.contains_key(&size) {
            return Vec::new();
        }

        let eclass = graph.class(canon_id);
        let type_overhead = self.type_overhead(eclass);

        // Bail if type size overhead is too big
        let Some(child_budget) = size.checked_sub(1 + type_overhead) else {
            return Vec::new();
        };

        let ty = OriginTree::from_eclass(graph, canon_id);

        let mut results = Vec::new();

        for node in eclass.nodes() {
            let children = node.children();

            for child_combo in self.enumerate_children(graph, children, child_budget, cache) {
                results.push(OriginTree::new_typed(
                    node.label().clone(),
                    child_combo,
                    ty.clone(),
                    canon_id.into(),
                ));
            }
        }

        results
    }

    /// Enumerate all ways to fill `children` with exactly `budget` total size,
    /// returning the cartesian product of child terms for each valid size tuple.
    fn enumerate_children<L: Language, N: TypeAnalysis<L>>(
        &self,
        graph: &EGraph<L>,
        children: &[Id],
        budget: usize,
        cache: &DashMap<(Id, usize), Vec<OriginExpr<L>>>,
    ) -> impl Iterator<Item = Vec<OriginExpr<L>>> {
        // Accumulate via left-fold: start with the empty tuple at budget=`budget`,
        // then for each child, expand every (remaining_budget, partial_combo) by
        // enumerating that child at each feasible size.

        let mut acc = vec![(budget, Vec::new())];

        for &child_id in children {
            let next_acc = acc
                .into_iter()
                .flat_map(|(remaining, partial)| {
                    self.expand_child(graph, child_id, remaining, partial, cache)
                })
                .collect();

            acc = next_acc;
        }

        // Only keep combos that used the entire budget
        acc.into_iter()
            .filter(|(remaining, _)| *remaining == 0)
            .map(|(_, combo)| combo)
    }

    /// Expand a single partial combo by one child, returning all valid extensions.
    fn expand_child<L: Language, N: TypeAnalysis<L>>(
        &self,
        graph: &EGraph<L>,
        child_id: Id,
        remaining: usize,
        partial: Vec<OriginExpr<L>>,
        cache: &DashMap<(Id, usize), Vec<OriginExpr<L>>>,
    ) -> Vec<(usize, Vec<OriginExpr<L>>)> {
        let canonical_child = graph.canonicalize(child_id);
        let Some(child_histogram) = self.get(&canonical_child) else {
            return Vec::new();
        };

        let mut results = Vec::new();
        for (&child_size, _) in child_histogram {
            if child_size > remaining {
                continue;
            }
            let child_trees =
                self.enumerate_class_cached(graph, canonical_child, child_size, cache);
            for tree in child_trees {
                let mut combo = partial.clone();
                combo.push(tree.clone());
                results.push((remaining - child_size, combo));
            }
        }
        results
        // match child_id {
        //     ExprChildId::Nat(nat_id) => {
        //         let child_size = self.type_sizes.get_nat_size(nat_id);
        //         if child_size <= remaining {
        //             let tree = OriginTree::from_nat(graph, nat_id);
        //             let mut combo = partial;
        //             combo.push(tree);
        //             vec![(remaining - child_size, combo)]
        //         } else {
        //             Vec::new()
        //         }
        //     }
        //     ExprChildId::Data(data_id) => {
        //         let child_size = self.type_sizes.get_data_size(data_id);
        //         if child_size <= remaining {
        //             let tree = OriginTree::from_data(graph, data_id);
        //             let mut combo = partial;
        //             combo.push(tree);
        //             vec![(remaining - child_size, combo)]
        //         } else {
        //             Vec::new()
        //         }
        //     }
        //     ExprChildId::EClass(eclass_id) => {

        //     }
        // }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::graph::{Class, EGraph};
//     use crate::nodes::ENode;
//     use crate::test_utils::*;
//     use crate::tree::TreeShaped;

//     use hashbrown::HashMap;
//     use num::BigUint;

//     #[test]
//     fn enumerate_single_leaf() {
//         let graph = EGraph::new(
//             cfv(vec![Class::new(
//                 vec![ENode::leaf("a".to_owned())],
//                 dummy_ty(),
//             )]),
//             EClassId::new(0),
//             Vec::new(),
//             HashMap::new(),
//             dummy_nat_nodes(),
//             HashMap::new(),
//         );

//         let tc = TermCount::<BigUint>::new(10, false, &graph);
//         let terms = tc.enumerate_root(&graph, 10, None);
//         assert_eq!(terms.len(), 1);
//         assert_eq!(terms[0].node(), "a");
//     }

//     #[test]
//     fn enumerate_two_leaves() {
//         let graph = EGraph::new(
//             cfv(vec![Class::new(
//                 vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
//                 dummy_ty(),
//             )]),
//             EClassId::new(0),
//             Vec::new(),
//             HashMap::new(),
//             dummy_nat_nodes(),
//             HashMap::new(),
//         );

//         let tc = TermCount::<BigUint>::new(10, false, &graph);
//         let terms = tc.enumerate_root(&graph, 10, None);
//         assert_eq!(terms.len(), 2);
//         let labels: Vec<_> = terms.iter().map(|t| t.node().as_str()).collect();
//         assert!(labels.contains(&"a"));
//         assert!(labels.contains(&"b"));
//     }

//     #[test]
//     fn enumerate_parent_child() {
//         let graph = EGraph::new(
//             cfv(vec![
//                 Class::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
//                 Class::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
//             ]),
//             EClassId::new(0),
//             Vec::new(),
//             HashMap::new(),
//             dummy_nat_nodes(),
//             HashMap::new(),
//         );

//         let tc = TermCount::<BigUint>::new(10, false, &graph);
//         let terms = tc.enumerate_root(&graph, 10, None);
//         assert_eq!(terms.len(), 1);
//         assert_eq!(terms[0].node(), "f");
//         assert_eq!(terms[0].children()[0].node(), "a");
//     }

//     #[test]
//     fn enumerate_combinatorial() {
//         // Class 0: f(class1, class2)
//         // Class 1: "a1", "a2"
//         // Class 2: "b1", "b2", "b3"
//         let graph = EGraph::new(
//             cfv(vec![
//                 Class::new(
//                     vec![ENode::new("f".to_owned(), vec![eid(1), eid(2)])],
//                     dummy_ty(),
//                 ),
//                 Class::new(
//                     vec![ENode::leaf("a1".to_owned()), ENode::leaf("a2".to_owned())],
//                     dummy_ty(),
//                 ),
//                 Class::new(
//                     vec![
//                         ENode::leaf("b1".to_owned()),
//                         ENode::leaf("b2".to_owned()),
//                         ENode::leaf("b3".to_owned()),
//                     ],
//                     dummy_ty(),
//                 ),
//             ]),
//             EClassId::new(0),
//             Vec::new(),
//             HashMap::new(),
//             dummy_nat_nodes(),
//             HashMap::new(),
//         );

//         let tc = TermCount::<BigUint>::new(10, false, &graph);
//         let terms = tc.enumerate_root(&graph, 10, None);
//         // 2 * 3 = 6 combinations
//         assert_eq!(terms.len(), 6);
//     }

//     #[test]
//     fn enumerate_respects_max_size() {
//         let graph = EGraph::new(
//             cfv(vec![
//                 Class::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
//                 Class::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
//             ]),
//             EClassId::new(0),
//             Vec::new(),
//             HashMap::new(),
//             dummy_nat_nodes(),
//             HashMap::new(),
//         );

//         let tc = TermCount::<BigUint>::new(10, false, &graph);
//         // max_size=1 should not include f(a) which is size 2
//         let terms = tc.enumerate_root(&graph, 1, None);
//         assert_eq!(terms.len(), 0);
//     }

//     #[test]
//     fn enumerate_count_matches_term_count() {
//         // Class 0: f(class1, class1) -> same child twice
//         // Class 1: "a", "b", g(class2)
//         // Class 2: "c"
//         let graph = EGraph::new(
//             cfv(vec![
//                 Class::new(
//                     vec![ENode::new("f".to_owned(), vec![eid(1), eid(1)])],
//                     dummy_ty(),
//                 ),
//                 Class::new(
//                     vec![
//                         ENode::leaf("a".to_owned()),
//                         ENode::leaf("b".to_owned()),
//                         ENode::new("g".to_owned(), vec![eid(2)]),
//                     ],
//                     dummy_ty(),
//                 ),
//                 Class::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
//             ]),
//             EClassId::new(0),
//             Vec::new(),
//             HashMap::new(),
//             dummy_nat_nodes(),
//             HashMap::new(),
//         );

//         let max_size = 10;
//         let tc = TermCount::<BigUint>::new(max_size, false, &graph);

//         let terms = tc.enumerate_root(&graph, max_size, None);
//         let expected_total: BigUint = tc
//             .data
//             .get(&graph.root())
//             .unwrap()
//             .iter()
//             .filter(|&(s, _)| *s <= max_size)
//             .map(|(_, c)| c.clone())
//             .sum();
//         assert_eq!(BigUint::from(terms.len()), expected_total);
//     }
// }
