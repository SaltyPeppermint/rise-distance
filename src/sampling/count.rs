use rand::distributions::WeightedIndex;
use rand::prelude::*;

use crate::count::{Counter, TermCount};
use crate::ids::{EClassId, ExprChildId};
use crate::nodes::Label;
use crate::sampling::Sampler;
use crate::{Graph, TreeNode};

pub struct CountSampler<'a, 'b, C: Counter, L: Label> {
    term_count: &'a TermCount<C>,
    graph: &'b Graph<L>,
}

impl<'a, 'b, C: Counter, L: Label> CountSampler<'a, 'b, C, L> {
    #[must_use]
    pub fn new(term_count: &'a TermCount<C>, graph: &'b Graph<L>) -> Self {
        Self { term_count, graph }
    }
}

impl<C: Counter, L: Label> Sampler<L> for CountSampler<'_, '_, C, L> {
    fn root(&self) -> EClassId {
        self.graph.root()
    }

    fn possible_size(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
    ) -> impl Iterator<Item = usize> + Send {
        let canon_id = self.graph.canonicalize(id);
        self.term_count
            .data
            .get(&canon_id)
            .into_iter()
            .flat_map(move |h| h.keys().filter(move |&&s| s >= min_size && s <= max_size))
            .copied()
    }

    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> TreeNode<L> {
        let canonical_id = self.graph.canonicalize(id);
        let eclass = self.graph.class(canonical_id);
        let child_budget = size - 1 - self.term_count.type_overhead(eclass);
        let cached = &self.term_count.suffix_cache[&canonical_id];

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
            let histogram = self.term_count.child_histogram(c_id, self.graph);
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
                    ExprChildId::Nat(nat_id) => TreeNode::from_nat(self.graph, *nat_id),
                    ExprChildId::Data(data_id) => TreeNode::from_data(self.graph, *data_id),
                    ExprChildId::EClass(eclass_id) => self.sample(*eclass_id, s, rng),
                })
                .collect(),
            TreeNode::from_eclass(self.graph, canonical_id),
        )
    }
}
