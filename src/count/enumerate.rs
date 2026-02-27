use indicatif::{ParallelProgressIterator, ProgressBar};
use rayon::prelude::*;

use super::Counter;
use super::TermCount;
use crate::TreeNode;
use crate::ids::{EClassId, ExprChildId};
use crate::nodes::Label;

impl<C: Counter, L: Label> TermCount<'_, C, L> {
    /// Enumerate all terms from the root e-class with sizes in `1..=max_size`.
    #[must_use]
    pub fn enumerate_root(
        &self,
        max_size: usize,
        progress: Option<ProgressBar>,
    ) -> Vec<TreeNode<L>> {
        self.enumerate(self.graph.root(), max_size, progress)
    }

    /// Enumerate all terms from an e-class with sizes in `1..=max_size`.
    #[must_use]
    #[expect(clippy::missing_panics_doc)]
    pub fn enumerate(
        &self,
        id: EClassId,
        max_size: usize,
        progress: Option<ProgressBar>,
    ) -> Vec<TreeNode<L>> {
        let canon_id = self.graph.canonicalize(id);
        let sum = self
            .of_eclass(id)
            .unwrap()
            .values()
            .sum::<C>()
            .try_into()
            .unwrap();

        // Build (size, node_index) pairs so rayon can distribute work
        // across both sizes and nodes, not just sizes.
        let canonical_id = self.graph.canonicalize(canon_id);
        let Some(histogram) = self.data.get(&canonical_id) else {
            return Vec::new();
        };
        let eclass = self.graph.class(canonical_id);
        let nodes = eclass.nodes();
        let type_overhead = self.type_overhead(eclass);
        let ty = TreeNode::from_eclass(self.graph, canonical_id);

        let sizes = (0..=max_size)
            .filter(|size| histogram.contains_key(size))
            .collect::<Vec<_>>();

        let iter = sizes
            .into_par_iter()
            .flat_map(|size| {
                (0..nodes.len()).into_par_iter().flat_map_iter({
                    let ty = ty.clone();
                    move |ni| {
                        let node = &nodes[ni];
                        let children = node.children();

                        let Some(child_budget) = size.checked_sub(1 + type_overhead) else {
                            return Vec::new().into_iter();
                        };

                        if children.is_empty() {
                            return if child_budget == 0 {
                                vec![TreeNode::new_typed(
                                    node.label().clone(),
                                    vec![],
                                    ty.clone(),
                                )]
                                .into_iter()
                            } else {
                                Vec::new().into_iter()
                            };
                        }

                        self.enumerate_children(children, child_budget)
                            .into_iter()
                            .map({
                                let label = node.label().clone();
                                let ty = ty.clone();
                                move |child_combo| {
                                    TreeNode::new_typed(
                                        label.clone(),
                                        child_combo,
                                        ty.clone(),
                                    )
                                }
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                    }
                })
            });

        if let Some(pb) = progress {
            pb.set_length(sum);
            iter.progress_with(pb).collect()
        } else {
            iter.collect()
        }
    }

    /// Enumerate all terms of exactly `size` from an e-class.
    #[must_use]
    fn enumerate_class(&self, id: EClassId, size: usize) -> Vec<TreeNode<L>> {
        let canonical_id = self.graph.canonicalize(id);

        // Check if this class has any terms at this size
        let Some(histogram) = self.data.get(&canonical_id) else {
            return Vec::new();
        };
        if !histogram.contains_key(&size) {
            return Vec::new();
        }

        let eclass = self.graph.class(canonical_id);
        let nodes = eclass.nodes();
        let type_overhead = self.type_overhead(eclass);
        let ty = TreeNode::from_eclass(self.graph, canonical_id);

        let Some(child_budget) = size.checked_sub(1 + type_overhead) else {
            return Vec::new();
        };

        let mut results = Vec::new();

        for node in nodes {
            let children = node.children();

            if children.is_empty() {
                if child_budget == 0 {
                    results.push(TreeNode::new_typed(
                        node.label().clone(),
                        vec![],
                        ty.clone(),
                    ));
                }
                continue;
            }

            // Enumerate all valid size distributions across children
            for child_combo in self.enumerate_children(children, child_budget) {
                results.push(TreeNode::new_typed(
                    node.label().clone(),
                    child_combo,
                    ty.clone(),
                ));
            }
        }

        results
    }

    /// Enumerate all ways to fill `children` with exactly `budget` total size,
    /// returning the cartesian product of child terms for each valid size tuple.
    fn enumerate_children(&self, children: &[ExprChildId], budget: usize) -> Vec<Vec<TreeNode<L>>> {
        // Accumulate via left-fold: start with the empty tuple at budget=`budget`,
        // then for each child, expand every (remaining_budget, partial_combo) by
        // enumerating that child at each feasible size.
        //
        // When the accumulator is large enough, parallelize the expansion step.
        const PAR_THRESHOLD: usize = 256;

        let mut acc = vec![(budget, Vec::new())];

        for &child_id in children {
            let next_acc = if acc.len() >= PAR_THRESHOLD {
                acc.into_par_iter()
                    .flat_map_iter(|(remaining, partial)| {
                        self.expand_child(child_id, remaining, partial)
                    })
                    .collect()
            } else {
                acc.into_iter()
                    .flat_map(|(remaining, partial)| {
                        self.expand_child(child_id, remaining, partial)
                    })
                    .collect()
            };

            acc = next_acc;
        }

        // Only keep combos that used the entire budget
        acc.into_iter()
            .filter(|(remaining, _)| *remaining == 0)
            .map(|(_, combo)| combo)
            .collect()
    }

    /// Expand a single partial combo by one child, returning all valid extensions.
    fn expand_child(
        &self,
        child_id: ExprChildId,
        remaining: usize,
        partial: Vec<TreeNode<L>>,
    ) -> Vec<(usize, Vec<TreeNode<L>>)> {
        match child_id {
            ExprChildId::Nat(nat_id) => {
                let child_size = self.type_sizes.get_nat_size(nat_id);
                if child_size <= remaining {
                    let tree = TreeNode::from_nat(self.graph, nat_id);
                    let mut combo = partial;
                    combo.push(tree);
                    vec![(remaining - child_size, combo)]
                } else {
                    Vec::new()
                }
            }
            ExprChildId::Data(data_id) => {
                let child_size = self.type_sizes.get_data_size(data_id);
                if child_size <= remaining {
                    let tree = TreeNode::from_data(self.graph, data_id);
                    let mut combo = partial;
                    combo.push(tree);
                    vec![(remaining - child_size, combo)]
                } else {
                    Vec::new()
                }
            }
            ExprChildId::EClass(eclass_id) => {
                let canonical_child = self.graph.canonicalize(eclass_id);
                let Some(child_histogram) = self.data.get(&canonical_child) else {
                    return Vec::new();
                };

                let mut results = Vec::new();
                for (&child_size, _) in child_histogram {
                    if child_size > remaining {
                        continue;
                    }
                    let child_trees = self.enumerate_class(canonical_child, child_size);
                    for tree in child_trees {
                        let mut combo = partial.clone();
                        combo.push(tree);
                        results.push((remaining - child_size, combo));
                    }
                }
                results
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EClass, EGraph};
    use crate::nodes::ENode;
    use crate::test_utils::*;
    use hashbrown::HashMap;
    use num::BigUint;

    #[test]
    fn enumerate_single_leaf() {
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

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let terms = tc.enumerate_root(10, None);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].label(), "a");
    }

    #[test]
    fn enumerate_two_leaves() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let terms = tc.enumerate_root(10, None);
        assert_eq!(terms.len(), 2);
        let labels: Vec<_> = terms.iter().map(|t| t.label().as_str()).collect();
        assert!(labels.contains(&"a"));
        assert!(labels.contains(&"b"));
    }

    #[test]
    fn enumerate_parent_child() {
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

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let terms = tc.enumerate_root(10, None);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].label(), "f");
        assert_eq!(terms[0].children()[0].label(), "a");
    }

    #[test]
    fn enumerate_combinatorial() {
        // Class 0: f(class1, class2)
        // Class 1: "a1", "a2"
        // Class 2: "b1", "b2", "b3"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a1".to_owned()), ENode::leaf("a2".to_owned())],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![
                        ENode::leaf("b1".to_owned()),
                        ENode::leaf("b2".to_owned()),
                        ENode::leaf("b3".to_owned()),
                    ],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let terms = tc.enumerate_root(10, None);
        // 2 * 3 = 6 combinations
        assert_eq!(terms.len(), 6);
    }

    #[test]
    fn enumerate_respects_max_size() {
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

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        // max_size=1 should not include f(a) which is size 2
        let terms = tc.enumerate_root(1, None);
        assert_eq!(terms.len(), 0);
    }

    #[test]
    fn enumerate_count_matches_term_count() {
        // Class 0: f(class1, class1) — same child twice
        // Class 1: "a", "b", g(class2)
        // Class 2: "c"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(1)])],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![
                        ENode::leaf("a".to_owned()),
                        ENode::leaf("b".to_owned()),
                        ENode::new("g".to_owned(), vec![eid(2)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let max_size = 10;
        let tc = TermCount::<BigUint, _>::new(max_size, false, &graph);

        let terms = tc.enumerate_root(max_size, None);
        let expected_total: BigUint = tc
            .of_root()
            .unwrap()
            .iter()
            .filter(|&(s, _)| *s <= max_size)
            .map(|(_, c)| c.clone())
            .sum();
        assert_eq!(BigUint::from(terms.len()), expected_total);
    }
}
