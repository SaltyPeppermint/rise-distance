//! Minimum distance search for E-Graphs.
//!
//! This module provides functions for finding the tree with minimum edit distance to a reference.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::structural::StructuralDistance;

use super::euler_str::EulerString;
use super::nodes::Label;
use super::structural::structural_diff;
use super::tree::TreeNode;
use super::zs::{EditCosts, PreprocessedTree, tree_distance_with_ref};

/// Core Zhang-Shasha minimum distance search over a parallel iterator of candidate trees.
///
/// Applies size-difference and Euler-string lower-bound pruning before computing
/// the full edit distance.
pub fn find_min_zs<L, CF, I>(
    candidates: I,
    reference: &TreeNode<L>,
    costs: &CF,
    with_types: bool,
) -> (Option<(TreeNode<L>, usize)>, ZSStats)
where
    L: Label,
    CF: EditCosts<L>,
    I: ParallelIterator<Item = TreeNode<L>>,
{
    let ref_flat = reference.flatten(with_types);

    let ref_size = ref_flat.size();
    let ref_euler = EulerString::new(&ref_flat);
    let ref_pp = PreprocessedTree::new(&ref_flat);
    let running_best = AtomicUsize::new(usize::MAX);

    candidates
        .map(|candidate| {
            let candidate_flat = candidate.flatten(with_types);
            let best = running_best.load(Ordering::Relaxed);

            if candidate_flat.size().abs_diff(ref_size) > best {
                return (None, ZSStats::size_pruned());
            }

            if ref_euler.lower_bound(&candidate_flat, costs) > best {
                return (None, ZSStats::euler_pruned());
            }

            let distance = tree_distance_with_ref(&candidate_flat, &ref_pp, costs);
            running_best.fetch_min(distance, Ordering::Relaxed);

            (Some((candidate, distance)), ZSStats::compared())
        })
        .reduce(
            || (None, ZSStats::default()),
            |a, b| {
                let best = [a.0, b.0].into_iter().flatten().min_by_key(|v| v.1);
                (best, a.1 + b.1)
            },
        )
}

/// Statistics from filtered extraction
#[derive(Debug, Clone, Default)]
pub struct ZSStats {
    /// Total number of trees enumerated
    pub trees_enumerated: usize,
    /// Trees pruned by simple metric
    pub size_pruned: usize,
    /// Number of trees pruned by euler string filter
    pub euler_pruned: usize,
    /// Number of trees for which full distance was computed
    pub full_comparisons: usize,
}

impl ZSStats {
    pub(crate) fn size_pruned() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 1,
            euler_pruned: 0,
            full_comparisons: 0,
        }
    }

    pub(crate) fn euler_pruned() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 0,
            euler_pruned: 1,
            full_comparisons: 0,
        }
    }

    pub(crate) fn compared() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 0,
            euler_pruned: 0,
            full_comparisons: 1,
        }
    }
}

impl std::ops::Add for ZSStats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            trees_enumerated: self.trees_enumerated + rhs.trees_enumerated,
            size_pruned: self.size_pruned + rhs.size_pruned,
            euler_pruned: self.euler_pruned + rhs.euler_pruned,
            full_comparisons: self.full_comparisons + rhs.full_comparisons,
        }
    }
}

/// Find the tree in the e-graph with minimum structural difference to the reference, with zs integration.
///
/// # Arguments
/// * `graph` - The e-graph to search
/// * `reference` - The target tree to match
/// * `costs` - Edit cost function for ZS
/// * `with_types` - Whether to include type annotations in comparison
///
/// # Returns
/// `Some((tree, distance))` if a tree was found.
#[must_use]
pub fn find_min_struct<L, CF, I>(
    candidates: I,
    reference: &TreeNode<L>,
    costs: &CF,
    with_types: bool,
) -> Option<(TreeNode<L>, StructuralDistance)>
where
    L: Label,
    CF: EditCosts<L>,
    I: ParallelIterator<Item = TreeNode<L>>,
{
    let running_best_overlap = AtomicUsize::new(0);
    let running_best_zs = AtomicUsize::new(usize::MAX);
    let ref_tree = reference.flatten(with_types);
    candidates
        .filter_map(|candidate| {
            let flat_candidate = candidate.flatten(with_types);
            let distance = structural_diff(&ref_tree, &flat_candidate, costs);
            let best_overlap =
                running_best_overlap.fetch_max(distance.overlap(), Ordering::Relaxed);
            if distance.overlap() > best_overlap {
                let best_zs = running_best_zs.fetch_min(distance.zs_sum(), Ordering::Relaxed);
                (distance.zs_sum() < best_zs).then_some((candidate, distance))
            } else {
                None
            }
        })
        .min_by_key(|(_, d)| *d)
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use rayon::iter::ParallelBridge;

    use super::*;
    use crate::EGraph;
    use crate::graph::EClass;
    use crate::ids::EClassId;
    use crate::nodes::ENode;
    use crate::zs::UnitCost;

    use crate::test_utils::*;

    #[test]
    fn min_distance_exact_match() {
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![
                node(
                    "a".to_owned(),
                    vec![
                        node(
                            "typeOf".to_owned(),
                            vec![leaf("b".to_owned()), leaf("0".to_owned())],
                        ),
                        node(
                            "typeOf".to_owned(),
                            vec![leaf("c".to_owned()), leaf("0".to_owned())],
                        ),
                    ],
                ),
                leaf("0".to_owned()),
            ],
        );
        let result = find_min_zs(
            graph
                .choice_iter(0)
                .map(|c| graph.tree_from_choices(graph.root(), &c))
                .par_bridge(),
            &reference,
            &UnitCost,
            true,
        )
        .0
        .unwrap();

        assert_eq!(result.1, 0);
    }

    #[test]
    fn min_distance_chooses_best() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("x".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );
        let result = find_min_zs(
            graph
                .choice_iter(0)
                .map(|c| graph.tree_from_choices(graph.root(), &c))
                .par_bridge(),
            &reference,
            &UnitCost,
            true,
        )
        .0
        .unwrap();

        assert_eq!(result.1, 0);
        assert_eq!(result.0.label(), "a");
    }

    #[test]
    fn min_distance_with_structure_choice() {
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![
                        ENode::new("a".to_owned(), vec![eid(1)]),
                        ENode::new("a".to_owned(), vec![eid(1), eid(2)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![
                node(
                    "a".to_owned(),
                    vec![node(
                        "typeOf".to_owned(),
                        vec![leaf("b".to_owned()), leaf("0".to_owned())],
                    )],
                ),
                leaf("0".to_owned()),
            ],
        );
        let result = find_min_zs(
            graph
                .choice_iter(0)
                .map(|c| graph.tree_from_choices(graph.root(), &c))
                .par_bridge(),
            &reference,
            &UnitCost,
            true,
        )
        .0
        .unwrap();

        assert_eq!(result.1, 0);
        assert_eq!(result.0.label(), "a");
        assert_eq!(result.0.children().len(), 1);
        assert_eq!(result.0.children()[0].label(), "b");
    }

    #[test]
    fn min_distance_extract_fast_exact_match() {
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![
                node(
                    "a".to_owned(),
                    vec![
                        node(
                            "typeOf".to_owned(),
                            vec![leaf("b".to_owned()), leaf("0".to_owned())],
                        ),
                        node(
                            "typeOf".to_owned(),
                            vec![leaf("c".to_owned()), leaf("0".to_owned())],
                        ),
                    ],
                ),
                leaf("0".to_owned()),
            ],
        );

        let result = find_min_zs(
            graph
                .choice_iter(0)
                .map(|c| graph.tree_from_choices(graph.root(), &c))
                .par_bridge(),
            &reference,
            &UnitCost,
            true,
        )
        .0
        .unwrap();
        assert_eq!(result.1, 0);
    }

    #[test]
    fn min_distance_extract_fast_chooses_best() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("x".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );

        let result = find_min_zs(
            graph
                .choice_iter(0)
                .map(|c| graph.tree_from_choices(graph.root(), &c))
                .par_bridge(),
            &reference,
            &UnitCost,
            true,
        )
        .0
        .unwrap();
        assert_eq!(result.1, 0);
        assert_eq!(result.0.label(), "a");
    }

    #[test]
    fn min_distance_extract_filtered_prunes_bad_trees() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("x".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );

        let (result, stats) = find_min_zs(
            graph
                .choice_iter(0)
                .map(|c| graph.tree_from_choices(graph.root(), &c))
                .par_bridge(),
            &reference,
            &UnitCost,
            true,
        );

        assert_eq!(result.unwrap().1, 0);
        assert_eq!(stats.trees_enumerated, 2);
        assert_eq!(
            stats.size_pruned + stats.euler_pruned + stats.full_comparisons,
            stats.trees_enumerated
        );
    }
}
