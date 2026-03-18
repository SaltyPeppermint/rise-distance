use hashbrown::{HashMap, HashSet};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;

use crate::TreeNode;
use crate::count::{Counter, TermCount};
use crate::ids::{EClassId, ExprChildId};
use crate::nodes::Label;
use crate::sampling::Sampler;

pub struct CountSampler<'a, C: Counter, L: Label>(&'a TermCount<'a, C, L>);

impl<'a, C: Counter, L: Label> CountSampler<'a, C, L> {
    #[must_use]
    pub fn new(term_count: &'a TermCount<'a, C, L>) -> Self {
        Self(term_count)
    }
}

impl<C: Counter, L: Label> Sampler<L> for CountSampler<'_, C, L> {
    fn root(&self) -> EClassId {
        self.0.graph.root()
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
    fn sample_unique(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<TreeNode<L>> {
        let canon_id = self.0.graph.canonicalize(id);
        self.0
            .data
            .get(&canon_id)
            .into_iter()
            .flat_map(|h| h.keys().filter(|&&s| s >= min_size && s <= max_size))
            .par_bridge()
            .flat_map(|&size| self.sample_root(size, samples_per_size[&size], size as u64))
            .collect()
    }

    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> TreeNode<L> {
        let canonical_id = self.0.graph.canonicalize(id);
        let eclass = self.0.graph.class(canonical_id);
        let child_budget = size - 1 - self.0.type_overhead(eclass);
        let cached = &self.0.suffix_cache[&canonical_id];

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

        let pick = &eclass.nodes()[pick_idx];
        let suffix = &cached[pick_idx];

        // Sequentially sample a size for each child, weighting by:
        //   count(child_i, s) * suffix_count(i+1, remaining - s)
        let mut remaining = child_budget;
        let mut child_sizes = Vec::with_capacity(pick.children().len());

        for (i, &c_id) in pick.children().iter().enumerate() {
            let histogram = self.0.child_histogram(c_id);
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
            pick.children()
                .iter()
                .zip(child_sizes)
                .map(|(c_id, s)| match c_id {
                    ExprChildId::Nat(nat_id) => TreeNode::from_nat(self.0.graph, *nat_id),
                    ExprChildId::Data(data_id) => TreeNode::from_data(self.0.graph, *data_id),
                    ExprChildId::EClass(eclass_id) => self.sample(*eclass_id, s, rng),
                })
                .collect(),
            TreeNode::from_eclass(self.0.graph, canonical_id),
        )
    }
}
