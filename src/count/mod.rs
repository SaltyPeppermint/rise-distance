//! Term counting analysis for e-graphs.
//!
//! Counts the number of terms up to a given size that can be extracted from each e-class.

use std::borrow::{Borrow, Cow};
use std::iter::{Product, Sum};

use egg::{EGraph, Id};
use hashbrown::{HashMap, HashSet};
use num_traits::{NumAssignRef, NumRef};
use rand::distributions::uniform::SampleUniform;
use rayon::prelude::*;

// use crate::graph::{Class, Graph};
// use crate::ids::{EClassId, ExprChildId};
use crate::utils::UniqueQueue;
use crate::{MyAnalysis, MyLanguage};

mod enumerate;

// use crate::graph::Graph;
// use crate::ids::{DataChildId, DataId, FunId, NatId, TypeChildId};

// trait TypeSizeCache {
//     type Lang: Label;
//     type TyN: TypeAnalysis<Self::Lang>;

//     /// Pre-compute sizes for every nat, data, and fun type node in the e-graph.
//     fn build(graph: &EGraph<Self::Lang, Self::TyN>) -> Self;

//     fn get_type_size(&self, id: Self::TyN::Data) -> usize;
// }

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
    + Ord
    + for<'a> Sum<&'a Self>
    + TryInto<u64, Error: std::fmt::Debug>
    + TryFrom<u64, Error: std::fmt::Debug>
    + TryFrom<usize, Error: std::fmt::Debug>
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
        + Ord
        + for<'a> Sum<&'a Self>
        + TryInto<u64, Error: std::fmt::Debug>
        + TryFrom<u64, Error: std::fmt::Debug>
        + TryFrom<usize, Error: std::fmt::Debug>
        + Product, // + Weight,
> Counter for T
{
}

/// Map from e-class ID to a map of (size -> count) (histogram).
#[derive(Debug, Clone)]
pub struct TermCount<C>
where
    // L: Label,
    // N: TypeAnalysis<L, T>,
    C: Counter,
    // TC: TypeSizeCache<Lang = L, TyN = N>,
{
    data: HashMap<Id, HashMap<usize, C>>,
    /// Per e-class, per node index: precomputed suffix convolution tables.
    /// `suffix_cache[eclass][node_idx][i]` = convolution of children `i..n`,
    /// mapping budget -> count. Computed once at max budget (`max_size - 1`).
    suffix_cache: HashMap<Id, Vec<Vec<HashMap<usize, C>>>>,
    // pub(crate) type_sizes: TC,
}

impl<C> TermCount<C>
where
    // L: Label,
    // N: TypeAnalysis<L, T>,
    C: Counter,
    // TC: TypeSizeCache<Lang = L, TyN = N>,
{
    /// Run the term counting analysis on an e-graph.
    ///
    /// # Arguments
    /// * `max_size` - Maximum term size to count
    #[must_use]
    #[expect(clippy::missing_panics_doc)]
    pub fn new<L: MyLanguage, N: MyAnalysis<L>>(
        max_size: usize,
        graph: &EGraph<L, N>,
    ) -> TermCount<C> {
        // Build parent map and type size cache
        let parents = Self::build_parent_map(graph);
        // let type_sizes = TypeSizeCache::build(graph);

        // Find leaf classes (classes with at least one leaf node)
        let mut pending = graph
            .classes()
            .filter(|&class| graph[class.id].nodes.iter().any(|n| n.is_leaf()))
            .map(|e| e.id)
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
                    debug_assert_eq!(graph.find(id), id);
                    let eclass = &graph[id];

                    let available_data = eclass.nodes.iter().filter_map(|node| {
                        let all_ready = node
                            .children()
                            .iter()
                            .all(|child_id| data.contains_key(&graph.find(*child_id)));
                        all_ready.then(|| {
                            // TODO: Re-add types
                            // let type_overhead = if with_types && let Some(t) = eclass.data {
                            //     1 + type_sizes.get_type_size(*t)
                            // } else {
                            //     0
                            // };
                            Self::make_node_data_from_map(max_size, graph, node.children(), &data)
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
                    assert!(!graph[id].is_empty());
                    pending.insert(id);
                }
            }
        }

        let suffix_cache = Self::build_suffix_cache_from_map(max_size, graph, &data);
        TermCount { data, suffix_cache }
    }

    /// Merge two term count data maps.
    fn merge(a: &mut HashMap<usize, C>, b: HashMap<usize, C>) {
        for (size, count) in b {
            a.entry(size).and_modify(|c| *c += &count).or_insert(count);
        }
    }

    /// Compute term counts for a single e-node, reading from a plain `HashMap`.
    fn make_node_data_from_map<L: MyLanguage, N: MyAnalysis<L>>(
        max_size: usize,
        graph: &EGraph<L, N>,
        children: &[Id],
        data: &HashMap<Id, HashMap<usize, C>>,
    ) -> HashMap<usize, C> {
        // Base size: 1 for the node itself + type overhead
        let base_size = 1;

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
            .map(|c| Self::get_child_data_from_map(graph, *c, data))
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
    fn build_parent_map<L: MyLanguage, N: MyAnalysis<L>>(
        graph: &EGraph<L, N>,
    ) -> HashMap<Id, HashSet<Id>> {
        let mut parents = HashMap::<Id, HashSet<Id>>::new();

        for class in graph.classes() {
            for node in &class.nodes {
                for child_id in node.children() {
                    let c_id = graph.find(*child_id);
                    parents
                        .entry(c_id)
                        .or_default()
                        .insert(graph.find(class.id));
                }
            }
        }

        parents
    }

    /// Get the count data for a child, reading from a plain `HashMap`.
    fn get_child_data_from_map<L: MyLanguage, N: MyAnalysis<L>>(
        graph: &EGraph<L, N>,
        child_id: Id,
        data: &HashMap<Id, HashMap<usize, C>>,
    ) -> HashMap<usize, C> {
        data.get(&graph.find(child_id)).cloned().unwrap_or_default()
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

    // TODO: Reenable when types become a concern
    // pub(crate) fn type_overhead(&self, eclass: &Id) -> usize {
    //     if self.with_types
    //         && let Some(t) = eclass.ty()
    //     {
    //         1 + self.type_sizes.get_type_size(*t)
    //     } else {
    //         0
    //     }
    // }

    /// Build suffix convolution tables for all e-classes at the maximum budget.
    fn build_suffix_cache_from_map<L: MyLanguage, N: MyAnalysis<L>>(
        max_size: usize,
        graph: &EGraph<L, N>,
        data: &HashMap<Id, HashMap<usize, C>>,
    ) -> HashMap<Id, Vec<Vec<HashMap<usize, C>>>> {
        let max_budget = max_size.saturating_sub(1);
        data.par_iter()
            .map(|(&id, _)| {
                let nodes = &graph[id].nodes;
                let per_node = nodes
                    .iter()
                    .map(|n| {
                        let histograms = n
                            .children()
                            .iter()
                            .map(|&c_id| Self::get_child_data_from_map(graph, c_id, data))
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

    /// Get the histogram for a child (size -> count).
    pub(crate) fn child_histogram<L: MyLanguage, N: MyAnalysis<L>>(
        &self,
        child_id: Id,
        graph: &EGraph<L, N>,
    ) -> Cow<'_, HashMap<usize, C>> {
        // match child_id {
        //     ExprChildId::Nat(nat_id) => Cow::Owned(HashMap::from([(
        //         self.type_sizes.get_nat_size(nat_id),
        //         C::one(),
        //     )])),
        //     ExprChildId::Data(data_id) => Cow::Owned(HashMap::from([(
        //         self.type_sizes.get_data_size(data_id),
        //         C::one(),
        //     )])),
        //     ExprChildId::EClass(eclass_id) => match self.get(&graph.find(eclass_id)) {
        //         Some(h) => Cow::Borrowed(h),
        //         None => Cow::Owned(HashMap::default()),
        //     },
        // }
        // TODO: Cleanup the cow, dont think we actually need it
        match self.data.get(&graph.find(child_id)) {
            Some(h) => Cow::Borrowed(h),
            None => Cow::Owned(HashMap::default()),
        }
    }

    #[must_use]
    pub fn data(&self) -> &HashMap<Id, HashMap<usize, C>> {
        &self.data
    }

    #[must_use]
    pub fn suffix_cache(&self) -> &HashMap<Id, Vec<Vec<HashMap<usize, C>>>> {
        &self.suffix_cache
    }
}

#[cfg(test)]
mod tests {
    use egg::EGraph;
    use num::BigUint;

    use super::*;
    use crate::egg::Math;

    fn sym(name: &str) -> Math {
        Math::Symbol(name.into())
    }

    #[test]
    fn single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let term_count = TermCount::<BigUint>::new(10, &graph);

        let root_data = &term_count.data[&graph.find(root)];
        assert_eq!(root_data.len(), 1);
        assert_eq!(root_data[&1], BigUint::from(1u32));
    }

    #[test]
    fn two_choices() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let term_count = TermCount::<BigUint>::new(10, &graph);

        let root_data = &term_count.data[&graph.find(a)];
        assert_eq!(root_data[&1], BigUint::from(2u32));
    }

    #[test]
    fn parent_child() {
        // Class 0: ln(class 1)
        // Class 1: leaf "a"
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let term_count = TermCount::<BigUint>::new(10, &graph);

        // Class a: one term of size 1
        assert_eq!(term_count.data[&graph.find(a)][&1], BigUint::from(1u32));

        // Class root: one term of size 2 (ln + a)
        assert_eq!(term_count.data[&graph.find(root)][&2], BigUint::from(1u32));
    }

    #[test]
    fn parent_with_multiple_child_choices() {
        // root: ln(child)
        // child: two leaves "a" and "b"
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let term_count = TermCount::<BigUint>::new(10, &graph);

        // child: two terms of size 1
        assert_eq!(term_count.data[&graph.find(a)][&1], BigUint::from(2u32));

        // root: two terms of size 2 (ln(a), ln(b))
        assert_eq!(term_count.data[&graph.find(root)][&2], BigUint::from(2u32));
    }

    #[test]
    fn two_children() {
        // root: (+ a b)
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        let root = graph.add(Math::Add([a, b]));
        graph.rebuild();

        let term_count = TermCount::<BigUint>::new(10, &graph);

        // root: one term of size 3 (+ + a + b)
        assert_eq!(term_count.data[&graph.find(root)][&3], BigUint::from(1u32));
    }

    #[test]
    fn combinatorial_explosion() {
        // root: (+ left right)
        // left:  two leaves "a1", "a2"
        // right: three leaves "b1", "b2", "b3"
        let mut graph = EGraph::<Math, ()>::new(());
        let a1 = graph.add(sym("a1"));
        let a2 = graph.add(sym("a2"));
        graph.union(a1, a2);

        let b1 = graph.add(sym("b1"));
        let b2 = graph.add(sym("b2"));
        let b3 = graph.add(sym("b3"));
        graph.union(b1, b2);
        graph.union(b1, b3);

        let root = graph.add(Math::Add([a1, b1]));
        graph.rebuild();

        let term_count = TermCount::<BigUint>::new(10, &graph);

        // root: 2 * 3 = 6 terms of size 3
        assert_eq!(term_count.data[&graph.find(root)][&3], BigUint::from(6u32));
    }

    #[test]
    fn max_size_filters() {
        // root: ln(a)
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        // max_size = 1, so ln(a) with size 2 should be filtered out
        let term_count = TermCount::<BigUint>::new(1, &graph);

        // a should have data (size 1)
        assert!(term_count.data.contains_key(&graph.find(a)));
        assert_eq!(term_count.data[&graph.find(a)][&1], BigUint::from(1u32));

        // root should be empty (size 2 exceeds max_size)
        assert!(
            term_count
                .data
                .get(&graph.find(root))
                .is_none_or(|d| d.is_empty())
        );
    }
}
