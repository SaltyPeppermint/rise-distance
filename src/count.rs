//! Term counting analysis for e-graphs.
//!
//! Counts the number of terms up to a given size that can be extracted from each e-class.

use std::borrow::{Borrow, Cow};
use std::iter::Product;
use std::sync::{Arc, Mutex};
use std::thread;

use dashmap::DashMap;
use hashbrown::{HashMap, HashSet};
use log::debug;
use num_traits::{NumAssignRef, NumRef};
use rand::distributions::WeightedIndex;
use rand::distributions::uniform::SampleUniform;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use super::graph::{EClass, EGraph};
use super::ids::{DataChildId, DataId, EClassId, ExprChildId, FunId, NatId, TypeChildId};
use super::nodes::Label;
use crate::TreeNode;
use crate::utils::UniqueQueue;

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
    data: HashMap<EClassId, HashMap<usize, C>>,
    /// Per e-class, per node index: precomputed suffix convolution tables.
    /// `suffix_cache[eclass][node_idx][i]` = convolution of children `i..n`,
    /// mapping budget -> count. Computed once at max budget (`limit - 1`).
    suffix_cache: HashMap<EClassId, Vec<Vec<HashMap<usize, C>>>>,
    graph: &'a EGraph<L>,
    type_sizes: TypeSizeCache,
    with_types: bool,
}

impl<C: Counter, L: Label> TermCount<'_, C, L> {
    /// Run the term counting analysis on an e-graph.
    ///
    ///
    ///
    /// # Arguments
    /// * `limit` - Maximum term size to count
    /// * `with_types` - If true, include type annotations in size calculations
    /// # Panics
    /// Panics if threads fail to join (should not happen in practice).
    #[must_use]
    pub fn new(limit: usize, with_types: bool, graph: &EGraph<L>) -> TermCount<'_, C, L> {
        // Build parent map and type size cache
        let parents = Self::build_parent_map(graph);
        let type_cache = TypeSizeCache::build(graph);

        // Find leaf classes (classes with at least one leaf node)
        let leaves = graph
            .class_ids()
            .filter(|&id| {
                graph
                    .class(id)
                    .nodes()
                    .iter()
                    .any(|n| n.children().is_empty())
            })
            .collect();

        let analysis_pending = Arc::new(Mutex::new(leaves));
        let data = DashMap::new();

        // Run parallel analysis
        thread::scope(|scope| {
            for i in 0..thread::available_parallelism().map_or(1, |p| p.get()) {
                let tap = analysis_pending.clone();
                let td = &data;
                let tp = &parents;
                let tc = &type_cache;
                scope.spawn(move || {
                    debug!("Thread #{i} started!");
                    TermCount::resolve_pending_analysis(limit, with_types, graph, td, &tap, tc, tp);
                    debug!("Thread #{i} finished!");
                });
            }
        });

        let suffix_cache = Self::build_suffix_cache(limit, graph, &data, &type_cache);
        TermCount {
            data: data.into_par_iter().collect(),
            suffix_cache,
            graph,
            type_sizes: type_cache,
            with_types,
        }
    }

    /// Process pending e-classes from the work queue.
    fn resolve_pending_analysis(
        limit: usize,
        with_types: bool,
        graph: &EGraph<L>,
        data: &DashMap<EClassId, HashMap<usize, C>>,
        analysis_pending: &Arc<Mutex<UniqueQueue<EClassId>>>,
        type_cache: &TypeSizeCache,
        parents: &HashMap<EClassId, HashSet<EClassId>>,
    ) {
        // Potentially, this might lead to a situation where only one thread is working on the queue.
        // This has not been observed in practice, but it is a potential bottleneck.
        // Drop lock at the end of the scope
        while let Some(id) = { analysis_pending.lock().unwrap().pop() } {
            // let canonical_id = graph.canonicalize(id); // Only canonical ids are ever in here
            debug_assert_eq!(graph.canonicalize(id), id);
            let eclass = graph.class(id);

            // Check if we can calculate the analysis for any enode
            let available_data = eclass.nodes().iter().filter_map(|node| {
                // If all the childs eclass_children have data, we can calculate it!
                let all_ready = node.children().iter().all(|child_id| match child_id {
                    ExprChildId::Nat(_) | ExprChildId::Data(_) => true,
                    ExprChildId::EClass(eclass_id) => {
                        data.contains_key(&graph.canonicalize(*eclass_id))
                    }
                });
                all_ready.then(|| {
                    // Get the type overhead for this e-class
                    let type_overhead = if with_types {
                        1 + type_cache.get_type_size(eclass.ty())
                    } else {
                        0
                    };
                    TermCount::make_node_data(
                        limit,
                        graph,
                        node.children(),
                        data,
                        type_cache,
                        type_overhead,
                    )
                })
            });

            // If we have some info, we add that info to our storage.
            // Otherwise we have absolutely no info about the nodes so we can only put them back onto the queue.
            // and hope for a better time later.
            if let Some(computed_data) = available_data.reduce(|mut a, b| {
                Self::merge(&mut a, b);
                a
            }) {
                // If we have gained new information, put the parents onto the queue.
                // They need to be re-evaluated.
                // Only once we have reached a fixpoint we can stop updating the parents.
                if data.get(&id).is_none_or(|v| *v != computed_data) {
                    if let Some(parent_set) = parents.get(&id) {
                        analysis_pending
                            .lock()
                            .unwrap()
                            .extend(parent_set.iter().copied());
                    }
                    data.insert(id, computed_data);
                }
            } else {
                assert!(!eclass.nodes().is_empty());
                analysis_pending.lock().unwrap().insert(id);
            }
        }
    }

    /// Merge two term count data maps.
    fn merge(a: &mut HashMap<usize, C>, b: HashMap<usize, C>) {
        for (size, count) in b {
            a.entry(size).and_modify(|c| *c += &count).or_insert(count);
        }
    }

    /// Compute term counts for a single e-node.
    fn make_node_data(
        limit: usize,
        graph: &EGraph<L>,
        children: &[ExprChildId],
        data: &DashMap<EClassId, HashMap<usize, C>>,
        type_cache: &TypeSizeCache,
        type_overhead: usize,
    ) -> HashMap<usize, C> {
        // Base size: 1 for the node itself + type overhead
        let base_size = 1 + type_overhead;

        if children.is_empty() {
            if base_size <= limit {
                return HashMap::from([(base_size, C::one())]);
            }
            return HashMap::new();
        }

        let Some(budget) = limit.checked_sub(base_size) else {
            return HashMap::new();
        };

        let histograms = children
            .iter()
            .map(|c| TermCount::get_child_data(graph, *c, data, type_cache))
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

        for class_id in graph.class_ids() {
            for node in graph.class(class_id).nodes() {
                for child_id in node.children() {
                    if let ExprChildId::EClass(child_eclass_id) = child_id {
                        let canonical_child = graph.canonicalize(*child_eclass_id);
                        parents.entry(canonical_child).or_default().insert(class_id);
                    }
                }
            }
        }

        parents
    }

    /// Get the count data for a child, handling Nat/Data/EClass variants.
    fn get_child_data(
        graph: &EGraph<L>,
        child_id: ExprChildId,
        data: &DashMap<EClassId, HashMap<usize, C>>,
        type_cache: &TypeSizeCache,
    ) -> HashMap<usize, C> {
        match child_id {
            ExprChildId::Nat(nat_id) => {
                // Nat nodes have a fixed size (no choices)
                let size = type_cache.get_nat_size(nat_id);
                HashMap::from([(size, C::one())])
            }
            ExprChildId::Data(data_id) => {
                // Data type nodes have a fixed size (no choices)
                let size = type_cache.get_data_size(data_id);
                HashMap::from([(size, C::one())])
            }
            ExprChildId::EClass(eclass_id) => {
                // E-class children use the precomputed data
                let canonical_id = graph.canonicalize(eclass_id);
                data.get(&canonical_id)
                    .map(|r| r.clone())
                    .unwrap_or_default()
            }
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
    pub fn sample_root(
        &self,
        size: usize,
        samples: u64,
        seed: u64,
    ) -> impl ParallelIterator<Item = TreeNode<L>> {
        self.sample_class(self.graph.root(), size, samples, seed)
    }

    #[must_use]
    pub fn sample_class(
        &self,
        id: EClassId,
        size: usize,
        samples: u64,
        seed: u64,
    ) -> impl ParallelIterator<Item = TreeNode<L>> {
        (0..samples).into_par_iter().map(move |sample| {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            rng.set_stream(sample);
            self.sample(id, size, &mut rng)
        })
    }

    /// Get the histogram for a child (size -> count).
    fn child_histogram(&self, child_id: ExprChildId) -> Cow<'_, HashMap<usize, C>> {
        match child_id {
            ExprChildId::Nat(nat_id) => Cow::Owned(HashMap::from([(
                self.type_sizes.get_nat_size(nat_id),
                C::one(),
            )])),
            ExprChildId::Data(data_id) => Cow::Owned(HashMap::from([(
                self.type_sizes.get_data_size(data_id),
                C::one(),
            )])),
            ExprChildId::EClass(eclass_id) => match self.of_eclass(eclass_id) {
                Some(h) => Cow::Borrowed(h),
                None => Cow::Owned(HashMap::default()),
            },
        }
    }

    fn type_overhead(&self, eclass: &EClass<L>) -> usize {
        if self.with_types {
            1 + self.type_sizes.get_type_size(eclass.ty())
        } else {
            0
        }
    }

    #[must_use]
    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> TreeNode<L> {
        let canonical_id = self.graph.canonicalize(id);
        let eclass = self.graph.class(canonical_id);
        let nodes = eclass.nodes();
        let type_overhead = self.type_overhead(eclass);
        let child_budget = size - 1 - type_overhead;
        let cached = &self.suffix_cache[&canonical_id];

        // Pick a node weighted by how many terms of the target size it produces.
        let weights = cached
            .iter()
            .map(|suffix| {
                suffix[0]
                    .get(&child_budget)
                    .cloned()
                    .unwrap_or_else(C::zero)
            })
            .collect::<Vec<_>>();
        let pick_idx = WeightedIndex::new(&weights).unwrap().sample(rng);

        let pick = &nodes[pick_idx];
        let children = pick.children();
        let suffix = &cached[pick_idx];

        if children.is_empty() {
            return TreeNode::new_typed(
                pick.label().clone(),
                vec![],
                Some(TreeNode::from_eclass(self.graph, canonical_id)),
            );
        }
        // Sequentially sample a size for each child, weighting by:
        //   count(child_i, s) * suffix_count(i+1, remaining - s)
        let mut remaining = child_budget;
        let mut child_sizes = Vec::with_capacity(children.len());

        for (i, &c_id) in children.iter().enumerate() {
            let histogram = self.child_histogram(c_id);
            let candidates = histogram
                .iter()
                .filter_map(|(&s, count)| {
                    remaining
                        .checked_sub(s)
                        .and_then(|r| suffix[i + 1].get(&r))
                        .map(|rest_count| (s, count.to_owned() * rest_count))
                })
                .collect::<Vec<_>>();

            let dist = WeightedIndex::new(candidates.iter().map(|(_, w)| w)).unwrap();
            let chosen_size = candidates[dist.sample(rng)].0;

            child_sizes.push(chosen_size);
            remaining -= chosen_size;
        }

        TreeNode::new_typed(
            pick.label().clone(),
            children
                .iter()
                .zip(child_sizes)
                .map(|(c_id, s)| match c_id {
                    ExprChildId::Nat(nat_id) => TreeNode::from_nat(self.graph, *nat_id),
                    ExprChildId::Data(data_id) => TreeNode::from_data(self.graph, *data_id),
                    ExprChildId::EClass(eclass_id) => self.sample(*eclass_id, s, rng),
                })
                .collect(),
            Some(TreeNode::from_eclass(self.graph, canonical_id)),
        )
    }

    /// Build suffix convolution tables for all e-classes at the maximum budget.
    fn build_suffix_cache(
        limit: usize,
        graph: &EGraph<L>,
        data: &DashMap<EClassId, HashMap<usize, C>>,
        type_cache: &TypeSizeCache,
    ) -> HashMap<EClassId, Vec<Vec<HashMap<usize, C>>>> {
        let max_budget = limit.saturating_sub(1);
        data.par_iter()
            .map(|entry| {
                let id = *entry.key();
                let nodes = graph.class(id).nodes();
                let per_node = nodes
                    .iter()
                    .map(|n| {
                        let histograms = n
                            .children()
                            .iter()
                            .map(|&c_id| Self::get_child_data(graph, c_id, data, type_cache))
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
    fn suffix_convolutions<H: Borrow<HashMap<usize, C>>>(
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

/// Cache for type node sizes to avoid repeated computation.
///
/// Built eagerly before parallelism via [`TypeSizeCache::build`] so that
/// the parallel phase only needs a shared `&TypeSizeCache` (no locking).
#[derive(Debug, Default, Clone)]
struct TypeSizeCache {
    nats: HashMap<NatId, usize>,
    data: HashMap<DataId, usize>,
    funs: HashMap<FunId, usize>,
}

impl TypeSizeCache {
    /// Pre-compute sizes for every nat, data, and fun type node in the e-graph.
    fn build<L: Label>(graph: &EGraph<L>) -> Self {
        let mut cache = Self::default();
        for id in graph.nat_ids() {
            cache.ensure_nat(graph, id);
        }
        for id in graph.data_ids() {
            cache.ensure_data(graph, id);
        }
        for id in graph.fun_ids() {
            cache.ensure_fun(graph, id);
        }
        cache
    }

    fn get_type_size(&self, id: TypeChildId) -> usize {
        match id {
            TypeChildId::Nat(nat_id) => self.nats[&nat_id],
            TypeChildId::Type(fun_id) => self.funs[&fun_id],
            TypeChildId::Data(data_id) => self.data[&data_id],
        }
    }

    fn get_nat_size(&self, id: NatId) -> usize {
        self.nats[&id]
    }

    fn get_data_size(&self, id: DataId) -> usize {
        self.data[&id]
    }

    // -- eager population helpers (called only during `new`) --

    fn ensure_nat<L: Label>(&mut self, graph: &EGraph<L>, id: NatId) {
        if self.nats.contains_key(&id) {
            return;
        }
        let node = graph.nat(id);
        for &child_id in node.children() {
            self.ensure_nat(graph, child_id);
        }
        let size = 1 + node
            .children()
            .iter()
            .map(|&c| self.nats[&c])
            .sum::<usize>();
        self.nats.insert(id, size);
    }

    fn ensure_data<L: Label>(&mut self, graph: &EGraph<L>, id: DataId) {
        if self.data.contains_key(&id) {
            return;
        }
        let node = graph.data_ty(id);
        for &child_id in node.children() {
            match child_id {
                DataChildId::Nat(nat_id) => self.ensure_nat(graph, nat_id),
                DataChildId::DataType(data_id) => self.ensure_data(graph, data_id),
            }
        }
        let size = 1 + node
            .children()
            .iter()
            .map(|&c| match c {
                DataChildId::Nat(nat_id) => self.nats[&nat_id],
                DataChildId::DataType(data_id) => self.data[&data_id],
            })
            .sum::<usize>();
        self.data.insert(id, size);
    }

    fn ensure_fun<L: Label>(&mut self, graph: &EGraph<L>, id: FunId) {
        if self.funs.contains_key(&id) {
            return;
        }
        let node = graph.fun_ty(id);
        for &child_id in node.children() {
            match child_id {
                TypeChildId::Nat(nat_id) => self.ensure_nat(graph, nat_id),
                TypeChildId::Data(data_id) => self.ensure_data(graph, data_id),
                TypeChildId::Type(fun_id) => self.ensure_fun(graph, fun_id),
            }
        }
        let size = 1 + node
            .children()
            .iter()
            .map(|&c| match c {
                TypeChildId::Nat(nat_id) => self.nats[&nat_id],
                TypeChildId::Data(data_id) => self.data[&data_id],
                TypeChildId::Type(fun_id) => self.funs[&fun_id],
            })
            .sum::<usize>();
        self.funs.insert(id, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EClass;
    use crate::nodes::{ENode, NatNode};
    use num::BigUint;

    fn eid(i: usize) -> ExprChildId {
        ExprChildId::EClass(EClassId::new(i))
    }

    fn dummy_ty() -> TypeChildId {
        TypeChildId::Nat(NatId::new(0))
    }

    fn dummy_nat_nodes() -> HashMap<NatId, NatNode<String>> {
        let mut nats = HashMap::new();
        nats.insert(NatId::new(0), NatNode::leaf("0".to_owned()));
        nats
    }

    fn cfv(classes: Vec<EClass<String>>) -> HashMap<EClassId, EClass<String>> {
        classes
            .into_iter()
            .enumerate()
            .map(|(i, c)| (EClassId::new(i), c))
            .collect()
    }

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
    fn size_limit_filters() {
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

        // Limit = 1, so f(a) with size 2 should be filtered out
        let term_count = TermCount::<BigUint, _>::new(1, false, &graph);

        // Class 1 should have data (size 1)
        assert!(term_count.data.contains_key(&EClassId::new(1)));
        assert_eq!(term_count.data[&EClassId::new(1)][&1], BigUint::from(1u32));

        // Class 0 should be empty (size 2 exceeds limit)
        assert!(
            term_count
                .data
                .get(&EClassId::new(0))
                .is_none_or(|d| d.is_empty())
        );
    }
}
