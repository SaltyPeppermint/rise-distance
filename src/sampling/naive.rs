use egg::{EGraph, Id, RecExpr};
use hashbrown::HashSet;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::count::{Counter, TermCount};
use crate::sampling::{Sampler, common};
use crate::{MyAnalysis, MyLanguage, OriginLang, stack_children};
// use crate::tree::OriginTree;

pub struct NaiveSampler<'a, 'b, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    term_count: &'a TermCount<C>,
    graph: &'b EGraph<L, N>,
    root: Id,
}

impl<'a, 'b, C, L, N> NaiveSampler<'a, 'b, C, L, N>
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

impl<C, L, N> Sampler<L> for NaiveSampler<'_, '_, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn root(&self) -> Id {
        self.root
    }

    fn possible_size(&self, id: Id, size: usize, samples: u64) -> bool {
        common::possible_size(self.term_count, self.graph, id, size, samples)
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
        common::sample_batch::<PARALLEL, _, _, _>(self, id, samples_per_size, seed, check)
    }

    /// Sample uniformly: each feasible choice gets equal weight.
    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let canon_id = self.graph.find(id);
        let eclass = &self.graph[canon_id];
        let child_budget = size - 1; // - self.term_count.type_overhead(&canon_id);
        let cached = &self.term_count.suffix_cache()[&canon_id];

        // Pick a node uniformly from those that can produce the target size.
        let pick_idx = cached
            .iter()
            .enumerate()
            .filter_map(|(idx, suffix)| suffix[0].contains_key(&child_budget).then_some(idx))
            .choose(rng)
            .unwrap();

        let pick = &eclass.nodes[pick_idx];
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

                self.sample(c_id, chosen_size, rng)
                // match c_id {
                //     ExprChildId::Nat(nat_id) => OriginTree::from_nat(self.graph, nat_id),
                //     ExprChildId::Data(data_id) => OriginTree::from_data(self.graph, data_id),
                //     ExprChildId::EClass(eclass_id) => self.sample(eclass_id, chosen_size, rng),
                // }
            })
            .collect::<Vec<_>>();

        // OriginTree::new_typed(
        //     pick.label().clone(),
        //     children,
        //     OriginTree::from_eclass(self.graph, canon_id),
        //     canon_id.into(),
        // )
        stack_children(&children, OriginLang::new(pick.clone(), canon_id))
    }

    fn sample_batch_root<const PARALLEL: bool, F>(
        &self,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
        check: &F,
    ) -> HashSet<egg::RecExpr<OriginLang<L>>>
    where
        F: Fn(&egg::RecExpr<OriginLang<L>>) -> bool + Sync,
    {
        self.sample_batch::<PARALLEL, _>(self.root(), samples_per_size, seed, check)
    }
}
