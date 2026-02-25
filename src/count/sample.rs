use std::borrow::Cow;

use hashbrown::{HashMap, HashSet};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use super::Counter;
use super::TermCount;
use crate::TreeNode;
use crate::ids::{EClassId, ExprChildId};
use crate::nodes::Label;

impl<C: Counter, L: Label> TermCount<'_, C, L> {
    #[must_use]
    pub(crate) fn sample_root(
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
    pub fn sample_unique_root(
        &self,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<TreeNode<L>> {
        self.sample_unique(self.graph.root(), min_size, max_size, samples_per_size)
    }

    /// Sample unique terms across a range of sizes.
    ///
    /// For each size in `[min_size, max_size]` that the root e-class actually has
    /// terms for, samples `samples_fn` terms and deduplicates them.
    ///
    /// Only sizes present in the root's histogram are sampled. The root e-class
    /// may have gaps in its reachable sizes (e.g. terms only at sizes 5, 7, 9),
    /// and calling `sample` with a size that has no terms would cause all node
    /// weights to be zero, panicking with `AllWeightsZero`.
    #[must_use]
    pub fn sample_unique(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<TreeNode<L>> {
        let canon_id = self.graph.canonicalize(id);
        self.data
            .get(&canon_id)
            .into_iter()
            .flat_map(|h| h.keys().filter(|&&s| s >= min_size && s <= max_size))
            .par_bridge()
            .flat_map(|&size| self.sample_root(size, samples_per_size[&size], size as u64))
            .collect()
    }

    #[must_use]
    pub(crate) fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> TreeNode<L> {
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
                TreeNode::from_eclass(self.graph, canonical_id),
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
            TreeNode::from_eclass(self.graph, canonical_id),
        )
    }

    /// Get the histogram for a child (size -> count).
    pub(crate) fn child_histogram(&self, child_id: ExprChildId) -> Cow<'_, HashMap<usize, C>> {
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
}
