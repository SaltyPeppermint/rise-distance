use hashbrown::HashSet;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::count::{Counter, TermCount};
use crate::ids::ExprChildId;
use crate::sampling::Sampler;
use crate::tree::OriginTree;
use crate::{EClassId, Graph, Label};

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

impl<C: Counter, L: Label> Sampler for NaiveSampler<'_, '_, C, L> {
    type Label = L;

    fn root(&self) -> EClassId {
        self.graph.root()
    }

    fn possible_size(&self, id: EClassId, size: usize, samples: u64) -> bool {
        super::common::possible_size(self.term_count, self.graph, id, size, samples)
    }

    fn sample_batch<const PARALLEL: bool>(
        &self,
        id: EClassId,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
    ) -> HashSet<OriginTree<L>> {
        super::common::sample_batch::<PARALLEL, _>(self, id, samples_per_size, seed)
    }

    /// Sample uniformly: each feasible choice gets equal weight.
    fn sample(&self, id: EClassId, size: usize, rng: &mut ChaCha12Rng) -> OriginTree<L> {
        let canon_id = self.graph.canonicalize(id);
        let eclass = self.graph.class(canon_id);
        let child_budget = size - 1 - self.term_count.type_overhead(eclass);
        let cached = &self.term_count.suffix_cache[&canon_id];

        // Pick a node uniformly from those that can produce the target size.
        let pick_idx = cached
            .iter()
            .enumerate()
            .filter_map(|(idx, suffix)| suffix[0].contains_key(&child_budget).then_some(idx))
            .choose(rng)
            .unwrap();

        let pick = &eclass.nodes()[pick_idx];
        let suffix = &cached[pick_idx];

        // Sample a feasible size for each child and recurse in one pass.
        let mut remaining = child_budget;
        let children = pick
            .children()
            .iter()
            .enumerate()
            .map(|(i, &c_id)| {
                let histogram = self.term_count.child_histogram(c_id, self.graph);
                let chosen_size = histogram
                    .iter()
                    .filter_map(|(&s, _)| {
                        remaining
                            .checked_sub(s)
                            .and_then(|r| suffix[i + 1].contains_key(&r).then_some(s))
                    })
                    .choose(rng)
                    .unwrap();
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
