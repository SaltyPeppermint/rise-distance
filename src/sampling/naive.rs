use crate::count::{Counter, TermCount};
use crate::ids::ExprChildId;
use crate::sampling::Sampler;
use crate::{EClassId, Graph, Label, TreeNode};

use hashbrown::{HashMap, HashSet};
use rand::Rng;
use rand::prelude::*;
use rayon::prelude::*;

pub struct NaiveSampler<'a, 'b, C: Counter, L: Label> {
    term_count: &'a TermCount<C>,
    graph: &'b Graph<L>,
}

impl<'a, 'b, C: Counter, L: Label> NaiveSampler<'a, 'b, C, L> {
    #[must_use]
    pub fn new(term_count: &'a TermCount<C>, graph: &'b Graph<L>) -> Self {
        Self { term_count, graph }
    }
}

impl<C: Counter, L: Label> Sampler<L> for NaiveSampler<'_, '_, C, L> {
    fn root(&self) -> EClassId {
        self.graph.root()
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
        let canon_id = self.graph.canonicalize(id);
        self.term_count
            .data
            .get(&canon_id)
            .into_iter()
            .flat_map(|h| h.keys().filter(|&&s| s >= min_size && s <= max_size))
            .par_bridge()
            .flat_map(|&size| {
                self.sample_root(size, samples_per_size[&size], size.try_into().unwrap())
            })
            .collect()
    }

    /// Here we sample with no regard for how many terms of a given size are in the
    /// `EClass` / `ENodes` children
    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> TreeNode<L> {
        let canon_id = self.graph.canonicalize(id);
        let eclass = self.graph.class(canon_id);
        let child_budget = size - 1 - self.term_count.type_overhead(eclass);
        let cached = &self.term_count.suffix_cache[&canon_id];

        // Pick a node from all the children that support the remaining budget.
        let pick_idx = cached
            .iter()
            .enumerate()
            .filter_map(|(idx, suffix)| suffix[0].contains_key(&child_budget).then_some(idx))
            .choose(rng)
            .unwrap();

        let pick = &eclass.nodes()[pick_idx];
        let suffix = &cached[pick_idx];

        // Sequentially sample a size for each child,
        // always making sure that the child supports that size
        let mut remaining = child_budget;
        let mut child_sizes = Vec::with_capacity(pick.children().len());

        for (i, &c_id) in pick.children().iter().enumerate() {
            let histogram = self.term_count.child_histogram(c_id, self.graph);
            let chosen_size = histogram
                .iter()
                .filter_map(|(&s, _)| {
                    remaining
                        .checked_sub(s)
                        .and_then(|r| suffix[i + 1].contains_key(&r).then_some(r))
                })
                .choose(rng)
                .unwrap();

            child_sizes.push(chosen_size);
            remaining -= chosen_size;
        }

        TreeNode::new_typed(
            pick.label().clone(),
            pick.children()
                .iter()
                .zip(child_sizes)
                .map(|(c_id, s)| match c_id {
                    ExprChildId::Nat(nat_id) => TreeNode::from_nat(self.graph, *nat_id),
                    ExprChildId::Data(data_id) => TreeNode::from_data(self.graph, *data_id),
                    ExprChildId::EClass(eclass_id) => self.sample(*eclass_id, s, rng),
                })
                .collect(),
            TreeNode::from_eclass(self.graph, canon_id),
        )
    }
}
