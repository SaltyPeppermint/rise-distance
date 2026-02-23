//! Term counting analysis for e-graphs.
//!
//! Counts the number of terms up to a given size that can be extracted from each e-class.

use std::borrow::{Borrow, Cow};
use std::iter::Product;

use hashbrown::{HashMap, HashSet};
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
use crate::tree::{PartialChild, PartialTree, tree_node_to_partial};
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
    /// # Arguments
    /// * `limit` - Maximum term size to count
    /// * `with_types` - If true, include type annotations in size calculations
    #[must_use]
    #[expect(clippy::missing_panics_doc)]
    pub fn new(limit: usize, with_types: bool, graph: &EGraph<L>) -> TermCount<'_, C, L> {
        // Build parent map and type size cache
        let parents = Self::build_parent_map(graph);
        let type_cache = TypeSizeCache::build(graph);

        // Find leaf classes (classes with at least one leaf node)
        let mut pending: UniqueQueue<EClassId> = graph
            .class_ids()
            .filter(|&id| {
                graph
                    .class(id)
                    .nodes()
                    .iter()
                    .any(|n| n.children().is_empty())
            })
            .collect();

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
                            let type_overhead = if with_types {
                                1 + type_cache.get_type_size(eclass.ty())
                            } else {
                                0
                            };
                            Self::make_node_data_from_map(
                                limit,
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

        let suffix_cache = Self::build_suffix_cache_from_map(limit, graph, &data, &type_cache);
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
        limit: usize,
        graph: &EGraph<L>,
        children: &[ExprChildId],
        data: &HashMap<EClassId, HashMap<usize, C>>,
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
            ExprChildId::Data(data_id) => {
                let size = type_cache.get_data_size(data_id);
                HashMap::from([(size, C::one())])
            }
            ExprChildId::EClass(eclass_id) => {
                let canonical_id = graph.canonicalize(eclass_id);
                data.get(&canonical_id).cloned().unwrap_or_default()
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
    pub fn of_root(&self) -> Option<&HashMap<usize, C>> {
        self.of_eclass(self.graph.root())
    }

    #[must_use]
    fn sample_root(
        &self,
        size: usize,
        samples: u64,
        seed: u64,
    ) -> impl ParallelIterator<Item = TreeNode<L>> {
        self.sample_class(self.graph.root(), size, samples, seed)
    }

    #[must_use]
    fn sample_class(
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

    /// Sample unique terms across a range of sizes from root.
    ///
    /// See `sample_unique` for more info
    #[must_use]
    pub fn sample_unique_root<F: Fn(usize) -> u64 + Sync + Send>(
        &self,
        min_size: usize,
        max_size: usize,
        samples_per_size: F,
    ) -> HashSet<TreeNode<L>> {
        self.sample_unique(self.graph.root(), min_size, max_size, samples_per_size)
    }

    /// Sample unique terms across a range of sizes.
    ///
    /// For each size in `[min_size, max_size]` that the root e-class actually has
    /// terms for, samples `samples_per_size` terms and deduplicates them.
    ///
    /// Only sizes present in the root's histogram are sampled. The root e-class
    /// may have gaps in its reachable sizes (e.g. terms only at sizes 5, 7, 9),
    /// and calling `sample` with a size that has no terms would cause all node
    /// weights to be zero, panicking with `AllWeightsZero`.
    #[must_use]
    pub fn sample_unique<F: Fn(usize) -> u64 + Sync + Send>(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
        samples_per_size: F,
    ) -> HashSet<TreeNode<L>> {
        let canon_id = self.graph.canonicalize(id);
        self.data
            .get(&canon_id)
            .into_iter()
            .flat_map(|h| h.keys().filter(|&&s| s >= min_size && s <= max_size))
            .par_bridge()
            .flat_map(|&size| self.sample_root(size, samples_per_size(size), size as u64))
            .collect()
    }

    /// Sample unique terms across a range of sizes from root, maximizing
    /// structural overlap with a reference tree.
    ///
    /// See `sample_unique_overlap` for more info.
    #[must_use]
    pub fn sample_unique_root_overlap<F: Fn(usize) -> u64 + Sync + Send>(
        &self,
        ref_tree: &TreeNode<L>,
        min_size: usize,
        max_size: usize,
        samples_per_size: F,
    ) -> HashSet<TreeNode<L>> {
        self.sample_unique_overlap(
            self.graph.root(),
            ref_tree,
            min_size,
            max_size,
            samples_per_size,
        )
    }

    /// Sample unique terms across a range of sizes, maximizing structural
    /// overlap with a reference tree.
    ///
    /// Like `sample_unique`, but uses `sample_overlap` instead of `sample` so
    /// that the locked-in structure from the reference tree is preserved in
    /// every sample. The partial tree is built once and reused across all
    /// sizes. Sizes where the fixed overlap exceeds the target are skipped.
    #[must_use]
    pub fn sample_unique_overlap<F: Fn(usize) -> u64 + Sync + Send>(
        &self,
        id: EClassId,
        ref_tree: &TreeNode<L>,
        min_size: usize,
        max_size: usize,
        samples_per_size: F,
    ) -> HashSet<TreeNode<L>> {
        let canon_id = self.graph.canonicalize(id);
        let Some(partial) = self.match_ref_tree(canon_id, ref_tree) else {
            return HashSet::new();
        };
        let fixed = partial.fixed_size(self.with_types);

        self.data
            .get(&canon_id)
            .into_iter()
            .flat_map(|h| {
                h.keys()
                    .filter(|&&s| s >= min_size && s <= max_size && s >= fixed)
                    .copied()
            })
            .par_bridge()
            .flat_map(|size| {
                let remaining = size - fixed;
                let samples = samples_per_size(size);
                let partial = partial.clone();
                (0..samples).into_par_iter().filter_map(move |sample| {
                    let mut rng = ChaCha12Rng::seed_from_u64(size as u64);
                    rng.set_stream(sample);
                    self.fill_holes_sampling(remaining, &mut rng, partial.clone())
                })
            })
            .collect()
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

    pub fn sample_root_overlap(
        &self,
        ref_tree: &TreeNode<L>,
        target_size: usize,
        samples: u64,
        seed: u64,
    ) -> Option<impl ParallelIterator<Item = TreeNode<L>>> {
        self.sample_overlap(self.graph.root(), ref_tree, target_size, samples, seed)
    }

    /// Sample a term of exactly `target_size` from `eclass_id`, maximizing
    /// structural overlap with `ref_tree`.
    ///
    /// Matches `ref_tree` against the e-class to lock in shared structure,
    /// then jointly samples sizes for unmatched subtrees (holes) and fills them.
    /// Returns `None` if no valid term of the target size can be produced
    /// or the tree does not match at the root
    #[must_use]
    pub fn sample_overlap(
        &self,
        eclass_id: EClassId,
        ref_tree: &TreeNode<L>,
        target_size: usize,
        samples: u64,
        seed: u64,
    ) -> Option<impl ParallelIterator<Item = TreeNode<L>>> {
        let canonical_id = self.graph.canonicalize(eclass_id);

        // Build partial tree
        let partial = self.match_ref_tree(canonical_id, ref_tree)?;

        // Budget accounting
        let fixed = partial.fixed_size(self.with_types);

        if fixed > target_size {
            return None;
        }
        let remaining = target_size - fixed;

        Some((0..samples).into_par_iter().filter_map(move |sample| {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            rng.set_stream(sample);
            self.fill_holes_sampling(remaining, &mut rng, partial.clone())
        }))
    }

    /// Match a reference tree against an e-class, producing a partial tree
    /// that maximizes structural overlap with the reference.
    ///
    /// At each e-class, finds e-nodes whose label matches the `ref_tree`'s label.
    /// If multiple match, tries all and picks the one with the largest
    /// `resolved_count`. Returns `None` if no e-node matches (caller creates a Hole).
    fn match_ref_tree(
        &self,
        eclass_id: EClassId,
        ref_tree: &TreeNode<L>,
    ) -> Option<PartialTree<L>> {
        let canonical_id = self.graph.canonicalize(eclass_id);
        let eclass = self.graph.class(canonical_id);
        let ty = Some(TreeNode::<L>::from_eclass(self.graph, canonical_id));

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
                        let nat_tree = TreeNode::<L>::from_nat(self.graph, nat_id);
                        partial_children
                            .push(PartialChild::Resolved(tree_node_to_partial(&nat_tree)));
                        ref_idx += 1;
                    }
                    ExprChildId::Data(data_id) => {
                        let data_tree = TreeNode::<L>::from_data(self.graph, data_id);
                        partial_children
                            .push(PartialChild::Resolved(tree_node_to_partial(&data_tree)));
                        ref_idx += 1;
                    }
                    ExprChildId::EClass(child_eclass_id) => {
                        if ref_idx < ref_children.len() {
                            match self.match_ref_tree(child_eclass_id, &ref_children[ref_idx]) {
                                Some(pt) => {
                                    partial_children.push(PartialChild::Resolved(pt));
                                }
                                None => {
                                    partial_children.push(PartialChild::Hole(
                                        self.graph.canonicalize(child_eclass_id),
                                    ));
                                }
                            }
                        } else {
                            partial_children
                                .push(PartialChild::Hole(self.graph.canonicalize(child_eclass_id)));
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

    /// Jointly samples sizes for unmatched subtrees (holes) and fills them.
    /// Returns `None` if no valid term of the target size can be produced.
    pub fn fill_holes_sampling<R: Rng + SeedableRng>(
        &self,
        remaining: usize,
        rng: &mut R,
        partial: PartialTree<L>,
    ) -> Option<TreeNode<L>> {
        let holes = partial.holes();

        if holes.is_empty() {
            if remaining == 0 {
                let mut empty_iter = std::iter::empty();
                return Some(partial.fill(&mut empty_iter));
            }
            return None;
        }

        // Build histograms for each hole and jointly sample sizes
        let hole_histograms = holes
            .iter()
            .map(|&hole_id| {
                self.child_histogram(ExprChildId::EClass(hole_id))
                    .into_owned()
            })
            .collect::<Vec<_>>();

        let suffix = Self::suffix_convolutions(&hole_histograms, remaining);

        // Check feasibility
        if suffix[0].get(&remaining).is_none_or(|c| *c == C::zero()) {
            return None;
        }

        Some(self.fill_inner(remaining, rng, partial, &holes, &hole_histograms, &suffix))
    }

    fn fill_inner<R: Rng + SeedableRng>(
        &self,
        remaining: usize,
        rng: &mut R,
        partial: PartialTree<L>,
        holes: &[EClassId],
        hole_histograms: &[HashMap<usize, C>],
        suffix: &[HashMap<usize, C>],
    ) -> TreeNode<L> {
        // Sequentially pick a size for each hole
        let mut budget_left = remaining;
        let mut hole_sizes = Vec::with_capacity(holes.len());

        // TODO CHECK THAT THIS IS REALLY UNIFORM
        for (i, hist) in hole_histograms.iter().enumerate() {
            let candidates: Vec<_> = hist
                .iter()
                .filter_map(|(&s, count)| {
                    budget_left
                        .checked_sub(s)
                        .and_then(|r| suffix[i + 1].get(&r))
                        .map(|rest_count| (s, count.to_owned() * rest_count))
                })
                .collect();

            let dist = WeightedIndex::new(candidates.iter().map(|(_, w)| w)).unwrap();
            let chosen = candidates[dist.sample(rng)].0;
            hole_sizes.push(chosen);
            budget_left -= chosen;
        }

        // Fill each hole
        let filled = holes
            .iter()
            .zip(hole_sizes.iter())
            .map(|(&hole_id, &hole_size)| self.sample(hole_id, hole_size, rng))
            .collect::<Vec<_>>();

        let mut fill_iter = filled.into_iter();
        partial.fill(&mut fill_iter)
    }

    /// Build suffix convolution tables for all e-classes at the maximum budget.
    fn build_suffix_cache_from_map(
        limit: usize,
        graph: &EGraph<L>,
        data: &HashMap<EClassId, HashMap<usize, C>>,
        type_cache: &TypeSizeCache,
    ) -> HashMap<EClassId, Vec<Vec<HashMap<usize, C>>>> {
        let max_budget = limit.saturating_sub(1);
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

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        let partial = tc.match_ref_tree(EClassId::new(0), &ref_tree).unwrap();
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

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "(f c)".parse().unwrap();

        let partial = tc.match_ref_tree(EClassId::new(0), &ref_tree).unwrap();
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

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "b".parse().unwrap();

        assert!(tc.match_ref_tree(EClassId::new(0), &ref_tree).is_none());
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

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        let partial = tc.match_ref_tree(EClassId::new(0), &ref_tree).unwrap();
        // Should pick the f->class1 node which fully matches (f a)
        assert_eq!(partial.resolved_count(), 2);
        assert!(partial.holes().is_empty());
    }

    #[test]
    fn sample_with_overlap_exact_match() {
        // Class 0: f(class1)
        // Class 1: leaf "a"
        // ref_tree: (f a) — exact match, target_size=2
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
        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        let results: Vec<_> = tc
            .sample_overlap(EClassId::new(0), &ref_tree, 2, 1, 42)
            .unwrap()
            .collect();
        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.size(false), 2);
        assert_eq!(result.label(), "f");
        assert_eq!(result.children()[0].label(), "a");
    }

    #[test]
    fn sample_with_overlap_with_holes() {
        // Class 0: f(class1, class2)
        // Class 1: leaf "a", leaf "x"
        // Class 2: leaf "b", leaf "y"
        // ref_tree: (f a z) — "a" matches class1, "z" does NOT match class2
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("x".to_owned())],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("b".to_owned()), ENode::leaf("y".to_owned())],
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
        let ref_tree: TreeNode<String> = "(f a z)".parse().unwrap();

        // target_size = 3: f(1) + a(1) + hole(1) = 3
        let results: Vec<_> = tc
            .sample_overlap(EClassId::new(0), &ref_tree, 3, 1, 42)
            .unwrap()
            .collect();
        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.size(false), 3);
        assert_eq!(result.label(), "f");
        // First child should be "a" (matched)
        assert_eq!(result.children()[0].label(), "a");
        // Second child should be either "b" or "y" (sampled from class 2)
        let second = result.children()[1].label();
        assert!(second == "b" || second == "y");
    }

    #[test]
    fn sample_with_overlap_budget_too_small() {
        // Class 0: f(class1)
        // Class 1: leaf "a"
        // ref_tree: (f a) — fixed size = 2, but target = 1
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
        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        assert!(
            tc.sample_overlap(EClassId::new(0), &ref_tree, 1, 1, 42)
                .is_none()
        );
    }

    #[test]
    fn sample_with_overlap_no_match_returns_none() {
        // Class 0: leaf "a", leaf "b"
        // ref_tree: "z" — no match at root, returns None
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
        let ref_tree: TreeNode<String> = "z".parse().unwrap();

        assert!(
            tc.sample_overlap(EClassId::new(0), &ref_tree, 1, 1, 42)
                .is_none()
        );
    }
}
