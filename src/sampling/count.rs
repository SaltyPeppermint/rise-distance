use egg::{EGraph, Id, RecExpr};
use hashbrown::HashSet;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::MyLanguage;
use crate::count::{Counter, TermCount};
use crate::sampling::Sampler;
use crate::{MyAnalysis, OriginLang, stack_children};

pub struct CountSampler<'a, 'b, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    term_count: &'a TermCount<C>,
    graph: &'b EGraph<L, N>,
    root: Id,
}

impl<'a, 'b, C, L, N> CountSampler<'a, 'b, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    #[must_use]
    pub fn new(term_count: &'a TermCount<C>, graph: &'b EGraph<L, N>, root: Id) -> Self {
        Self {
            term_count,
            graph,
            root,
        }
    }
}

impl<C, L, N> Sampler<L> for CountSampler<'_, '_, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn root(&self) -> Id {
        self.root
    }

    fn possible_size(&self, id: Id, size: usize, samples: u64) -> bool {
        super::common::possible_size(self.term_count, self.graph, id, size, samples)
    }

    fn sample_batch<const PARALLEL: bool, F>(
        &self,
        id: Id,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
        check: &F,
    ) -> HashSet<RecExpr<OriginLang<L>>>
    where
        F: Fn(&RecExpr<OriginLang<L>>) -> bool + Sync,
    {
        super::common::sample_batch::<PARALLEL, _, _, _>(self, id, samples_per_size, seed, check)
    }

    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let canon_id = self.graph.find(id);
        let eclass = &self.graph[canon_id];
        let child_budget = size - 1; //- self.term_count.type_overhead(&canon_id);
        let cached = &self.term_count.suffix_cache()[&canon_id];

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

        let pick = &eclass.nodes[pick_idx];
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
                self.sample(c_id, chosen_size, rng)
                // match c_id {
                //     ExprChildId::Nat(nat_id) => OriginTree::from_nat(self.graph, nat_id),
                //     ExprChildId::Data(data_id) => OriginTree::from_data(self.graph, data_id),
                //     ExprChildId::EClass(eclass_id) => self.sample(eclass_id, chosen_size, rng),
                // }
            })
            .collect::<Vec<_>>();

        // OriginTree::new_typed(
        //     pick.clone(),
        //     children,
        //     OriginTree::from_eclass(self.graph, canon_id),
        //     canon_id.into(),
        // )
        stack_children(&children, OriginLang::new(pick.clone(), canon_id))
    }
}
