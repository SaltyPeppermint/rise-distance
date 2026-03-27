use hashbrown::{HashMap, HashSet};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

use crate::Graph;
use crate::count::{Counter, TermCount};
use crate::ids::{EClassId, ExprChildId};
use crate::nodes::Label;
use crate::sampling::Sampler;
use crate::tree::OriginTree;

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

impl<C: Counter, L: Label> Sampler for CountSampler<'_, '_, C, L> {
    type Label = L;

    fn root(&self) -> EClassId {
        self.graph.root()
    }

    fn possible_size(&self, id: EClassId, size: usize, samples: u64) -> bool {
        super::common::possible_size(self.term_count, self.graph, id, size, samples)
    }

    fn sample_batch(
        &self,
        id: EClassId,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<OriginTree<L>> {
        super::common::sample_batch(self, id, samples_per_size)
    }

    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> OriginTree<L> {
        let canon_id = self.graph.canonicalize(id);
        let eclass = self.graph.class(canon_id);
        let child_budget = size - 1 - self.term_count.type_overhead(eclass);
        let cached = &self.term_count.suffix_cache[&canon_id];

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

        // Sample a size for each child (weighted by count * suffix) and recurse in one pass.
        let mut remaining = child_budget;
        let children = pick
            .children()
            .iter()
            .enumerate()
            .map(|(i, &c_id)| {
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
                remaining -= chosen_size;

                match c_id {
                    ExprChildId::Nat(nat_id) => OriginTree::from_nat(self.graph, nat_id),
                    ExprChildId::Data(data_id) => OriginTree::from_data(self.graph, data_id),
                    ExprChildId::EClass(eclass_id) => self.sample(eclass_id, chosen_size, rng),
                }
            })
            .collect();

        OriginTree::new_typed(
            pick.label().clone(),
            children,
            OriginTree::from_eclass(self.graph, canon_id),
            canon_id.into(),
        )
    }
}
