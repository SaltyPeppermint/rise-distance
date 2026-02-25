//! Term counting analysis for e-graphs.
//!
//! Counts the number of terms up to a given size that can be extracted from each e-class.

use std::borrow::Borrow;
use std::iter::Product;

use hashbrown::{HashMap, HashSet};
use num_traits::{NumAssignRef, NumRef};
use rand::distributions::uniform::SampleUniform;
use rayon::prelude::*;

use super::graph::{EClass, EGraph};
use super::ids::{EClassId, ExprChildId};
use super::nodes::Label;
use crate::utils::UniqueQueue;

mod overlap;
mod sample;

mod type_cache;

use type_cache::TypeSizeCache;

/// Counter trait for counting terms.
pub trait Counter:
    Clone
    + Send
    + Sync
    + NumRef
    + NumAssignRef
    + Default
    + std::fmt::Debug
    + SampleUniform
    + PartialEq
    + PartialOrd
    + Product // + Weight
{
}

impl<
    T: Clone
        + Send
        + Sync
        + NumRef
        + NumAssignRef
        + Default
        + std::fmt::Debug
        + SampleUniform
        + PartialEq
        + PartialOrd
        + Product, // + Weight,
> Counter for T
{
}

/// Map from e-class ID to a map of (size -> count) (histogram).
#[derive(Debug, Clone)]
pub struct TermCount<'a, C: Counter, L: Label> {
    pub(crate) data: HashMap<EClassId, HashMap<usize, C>>,
    /// Per e-class, per node index: precomputed suffix convolution tables.
    /// `suffix_cache[eclass][node_idx][i]` = convolution of children `i..n`,
    /// mapping budget -> count. Computed once at max budget (`max_size - 1`).
    pub(crate) suffix_cache: HashMap<EClassId, Vec<Vec<HashMap<usize, C>>>>,
    pub(crate) graph: &'a EGraph<L>,
    pub(crate) type_sizes: TypeSizeCache,
    pub(crate) with_types: bool,
}

impl<C: Counter, L: Label> TermCount<'_, C, L> {
    /// Run the term counting analysis on an e-graph.
    ///
    /// # Arguments
    /// * `max_size` - Maximum term size to count
    /// * `with_types` - If true, include type annotations in size calculations
    #[must_use]
    #[expect(clippy::missing_panics_doc)]
    pub fn new(max_size: usize, with_types: bool, graph: &EGraph<L>) -> TermCount<'_, C, L> {
        // Build parent map and type size cache
        let parents = Self::build_parent_map(graph);
        let type_cache = TypeSizeCache::build(graph);

        // Find leaf classes (classes with at least one leaf node)
        let mut pending = graph
            .class_ids()
            .filter(|&id| {
                graph
                    .class(id)
                    .nodes()
                    .iter()
                    .any(|n| n.children().is_empty())
            })
            .collect::<UniqueQueue<_>>();

        let mut data = HashMap::new();

        // Fixpoint iteration: process classes in rounds.
        // Each round drains the current queue, computes new data for each class
        // in parallel (reading only from the previous round's data), then applies
        // all updates sequentially before the next round.
        while !pending.is_empty() {
            // Compute new data for each class in parallel, reading from `data` (immutable)
            let results = pending
                .drain()
                .par_bridge()
                .map(|id| {
                    debug_assert_eq!(graph.canonicalize(id), id);
                    let eclass = graph.class(id);

                    let available_data = eclass.nodes().iter().filter_map(|node| {
                        let all_ready = node.children().iter().all(|child_id| match child_id {
                            ExprChildId::Nat(_) | ExprChildId::Data(_) => true,
                            ExprChildId::EClass(eclass_id) => {
                                data.contains_key(&graph.canonicalize(*eclass_id))
                            }
                        });
                        all_ready.then(|| {
                            let type_overhead = if with_types && let Some(t) = eclass.ty() {
                                1 + type_cache.get_type_size(*t)
                            } else {
                                0
                            };
                            Self::make_node_data_from_map(
                                max_size,
                                graph,
                                node.children(),
                                &data,
                                &type_cache,
                                type_overhead,
                            )
                        })
                    });

                    let merged = available_data.reduce(|mut a, b| {
                        Self::merge(&mut a, b);
                        a
                    });

                    (id, merged)
                })
                .collect::<Vec<_>>();

            // Apply results sequentially -> no concurrent mutation
            for (id, computed) in results {
                if let Some(computed_data) = computed {
                    if data.get(&id).is_none_or(|v| *v != computed_data) {
                        if let Some(parent_set) = parents.get(&id) {
                            pending.extend(parent_set.iter().copied());
                        }
                        data.insert(id, computed_data);
                    }
                } else {
                    // Not all children ready yet -> re-queue
                    assert!(!graph.class(id).nodes().is_empty());
                    pending.insert(id);
                }
            }
        }

        let suffix_cache = Self::build_suffix_cache_from_map(max_size, graph, &data, &type_cache);
        TermCount {
            data,
            suffix_cache,
            graph,
            type_sizes: type_cache,
            with_types,
        }
    }

    /// Merge two term count data maps.
    fn merge(a: &mut HashMap<usize, C>, b: HashMap<usize, C>) {
        for (size, count) in b {
            a.entry(size).and_modify(|c| *c += &count).or_insert(count);
        }
    }

    /// Compute term counts for a single e-node, reading from a plain `HashMap`.
    fn make_node_data_from_map(
        max_size: usize,
        graph: &EGraph<L>,
        children: &[ExprChildId],
        data: &HashMap<EClassId, HashMap<usize, C>>,
        type_cache: &TypeSizeCache,
        type_overhead: usize,
    ) -> HashMap<usize, C> {
        // Base size: 1 for the node itself + type overhead
        let base_size = 1 + type_overhead;

        if children.is_empty() {
            if base_size <= max_size {
                return HashMap::from([(base_size, C::one())]);
            }
            return HashMap::new();
        }

        let Some(budget) = max_size.checked_sub(base_size) else {
            return HashMap::new();
        };

        let histograms = children
            .iter()
            .map(|c| Self::get_child_data_from_map(graph, *c, data, type_cache))
            .collect::<Vec<_>>();
        let mut result = Self::convolve(&histograms, budget);

        // Offset all keys by base_size
        if base_size > 0 {
            result = result
                .into_iter()
                .map(|(size, count)| (size + base_size, count))
                .collect();
        }

        result
    }

    /// Build a map from child e-class to parent e-classes.
    fn build_parent_map(graph: &EGraph<L>) -> HashMap<EClassId, HashSet<EClassId>> {
        let mut parents = HashMap::<EClassId, HashSet<EClassId>>::new();

        for id in graph.class_ids() {
            for node in graph.class(id).nodes() {
                for child_id in node.children() {
                    if let ExprChildId::EClass(child_eclass_id) = child_id {
                        let c_id = graph.canonicalize(*child_eclass_id);
                        parents.entry(c_id).or_default().insert(id);
                    }
                }
            }
        }

        parents
    }

    /// Get the count data for a child, reading from a plain `HashMap`.
    fn get_child_data_from_map(
        graph: &EGraph<L>,
        child_id: ExprChildId,
        data: &HashMap<EClassId, HashMap<usize, C>>,
        type_cache: &TypeSizeCache,
    ) -> HashMap<usize, C> {
        match child_id {
            ExprChildId::Nat(nat_id) => {
                let size = type_cache.get_nat_size(nat_id);
                HashMap::from([(size, C::one())])
            }
            ExprChildId::Data(id) => {
                let size = type_cache.get_data_size(id);
                HashMap::from([(size, C::one())])
            }
            ExprChildId::EClass(id) => data
                .get(&graph.canonicalize(id))
                .cloned()
                .unwrap_or_default(),
        }
    }

    /// Convolve all child histograms into a single result (left-to-right).
    fn convolve<H: Borrow<HashMap<usize, C>>>(
        histograms: &[H],
        budget: usize,
    ) -> HashMap<usize, C> {
        let mut acc = HashMap::from([(0, C::one())]);
        let mut prev = HashMap::new();

        for h in histograms {
            std::mem::swap(&mut acc, &mut prev);
            for (&s_acc, c_acc) in &prev {
                for (&s_h, c_h) in h.borrow() {
                    let total = s_acc + s_h;
                    if total > budget {
                        continue;
                    }
                    let product = c_acc.to_owned() * c_h;
                    acc.entry(total)
                        .and_modify(|c| *c += &product)
                        .or_insert(product);
                }
            }
            prev.clear();
        }

        acc
    }

    #[must_use]
    pub fn of_eclass(&self, id: EClassId) -> Option<&HashMap<usize, C>> {
        self.data.get(&self.graph.canonicalize(id))
    }

    #[must_use]
    pub fn of_root(&self) -> Option<&HashMap<usize, C>> {
        self.of_eclass(self.graph.root())
    }

    pub(crate) fn type_overhead(&self, eclass: &EClass<L>) -> usize {
        if self.with_types
            && let Some(t) = eclass.ty()
        {
            1 + self.type_sizes.get_type_size(*t)
        } else {
            0
        }
    }

    /// Build suffix convolution tables for all e-classes at the maximum budget.
    fn build_suffix_cache_from_map(
        max_size: usize,
        graph: &EGraph<L>,
        data: &HashMap<EClassId, HashMap<usize, C>>,
        type_cache: &TypeSizeCache,
    ) -> HashMap<EClassId, Vec<Vec<HashMap<usize, C>>>> {
        let max_budget = max_size.saturating_sub(1);
        data.par_iter()
            .map(|(&id, _)| {
                let nodes = graph.class(id).nodes();
                let per_node = nodes
                    .iter()
                    .map(|n| {
                        let histograms = n
                            .children()
                            .iter()
                            .map(|&c_id| {
                                Self::get_child_data_from_map(graph, c_id, data, type_cache)
                            })
                            .collect::<Vec<_>>();
                        Self::suffix_convolutions(&histograms, max_budget)
                    })
                    .collect();
                (id, per_node)
            })
            .collect()
    }

    /// Convolve child histograms right-to-left, returning suffix intermediates.
    /// `suffix[i]` = convolution of children `i..n`, mapping budget -> count.
    pub(crate) fn suffix_convolutions<H: Borrow<HashMap<usize, C>>>(
        histograms: &[H],
        budget: usize,
    ) -> Vec<HashMap<usize, C>> {
        let n = histograms.len();
        let mut suffix = vec![HashMap::new(); n + 1];
        suffix[n] = HashMap::from([(0, C::one())]);

        for i in (0..n).rev() {
            let (left, right) = suffix.split_at_mut(i + 1);
            for (&s_i, c_i) in histograms[i].borrow() {
                for (&s_rest, c_rest) in &right[0] {
                    let total = s_i + s_rest;
                    if total > budget {
                        continue;
                    }
                    let product = c_i.to_owned() * c_rest;
                    left[i]
                        .entry(total)
                        .and_modify(|c: &mut C| *c += &product)
                        .or_insert(product);
                }
            }
        }

        suffix
    }

    #[must_use]
    pub fn with_types(&self) -> bool {
        self.with_types
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EClass;
    use crate::nodes::ENode;
    use crate::test_utils::*;
    use num::BigUint;

    #[test]
    fn single_leaf_no_types() {
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

        let term_count = TermCount::<BigUint, _>::new(10, false, &graph);

        let root_data = &term_count.data[&EClassId::new(0)];
        assert_eq!(root_data.len(), 1);
        assert_eq!(root_data[&1], BigUint::from(1u32));
    }

    #[test]
    fn single_leaf_with_types() {
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

        let term_count = TermCount::<BigUint, _>::new(10, true, &graph);

        let root_data = &term_count.data[&EClassId::new(0)];
        // Size = 1 (node) + 1 (typeOf) + 1 (type "0") = 3
        assert_eq!(root_data.len(), 1);
        assert_eq!(root_data[&3], BigUint::from(1u32));
    }

    #[test]
    fn two_choices_no_types() {
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
        let term_count = TermCount::<BigUint, _>::new(10, false, &graph);

        let root_data = &term_count.data[&EClassId::new(0)];
        // Two terms of size 1
        assert_eq!(root_data[&1], BigUint::from(2u32));
    }

    #[test]
    fn parent_child_no_types() {
        // Class 0: has node "f" pointing to class 1
        // Class 1: has leaf "a"
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

        let term_count = TermCount::<BigUint, _>::new(10, false, &graph);

        // Class 1: one term of size 1
        assert_eq!(term_count.data[&EClassId::new(1)][&1], BigUint::from(1u32));

        // Class 0: one term of size 2 (f + a)
        assert_eq!(term_count.data[&EClassId::new(0)][&2], BigUint::from(1u32));
    }

    #[test]
    fn parent_with_multiple_child_choices() {
        // Class 0: has node "f" pointing to class 1
        // Class 1: has two leaves "a" and "b"
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

        let term_count = TermCount::<BigUint, _>::new(10, false, &graph);

        // Class 1: two terms of size 1
        assert_eq!(term_count.data[&EClassId::new(1)][&1], BigUint::from(2u32));

        // Class 0: two terms of size 2 (f(a), f(b))
        assert_eq!(term_count.data[&EClassId::new(0)][&2], BigUint::from(2u32));
    }

    #[test]
    fn two_children() {
        // Class 0: has node "f" pointing to classes 1 and 2
        // Class 1: leaf "a"
        // Class 2: leaf "b"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2)])],
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

        let term_count = TermCount::<BigUint, _>::new(10, false, &graph);

        // Class 0: one term of size 3 (f + a + b)
        assert_eq!(term_count.data[&EClassId::new(0)][&3], BigUint::from(1u32));
    }

    #[test]
    fn combinatorial_explosion() {
        // Class 0: has node "f" pointing to classes 1 and 2
        // Class 1: two leaves "a1", "a2"
        // Class 2: three leaves "b1", "b2", "b3"
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
        let term_count = TermCount::<BigUint, _>::new(10, false, &graph);

        // Class 0: 2 * 3 = 6 terms of size 3
        assert_eq!(term_count.data[&EClassId::new(0)][&3], BigUint::from(6u32));
    }

    #[test]
    fn max_size_filters() {
        // Class 0: has node "f" pointing to class 1
        // Class 1: leaf "a"
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

        // max_size = 1, so f(a) with size 2 should be filtered out
        let term_count = TermCount::<BigUint, _>::new(1, false, &graph);

        // Class 1 should have data (size 1)
        assert!(term_count.data.contains_key(&EClassId::new(1)));
        assert_eq!(term_count.data[&EClassId::new(1)][&1], BigUint::from(1u32));

        // Class 0 should be empty (size 2 exceeds max_size)
        assert!(
            term_count
                .data
                .get(&EClassId::new(0))
                .is_none_or(|d| d.is_empty())
        );
    }
}
