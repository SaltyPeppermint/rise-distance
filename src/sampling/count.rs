use egg::{Analysis, EGraph, Id, Language};
use hashbrown::HashSet;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::count::{Counter, TermCount};
use crate::egg::TypeAnalysisWrapper;
use crate::origin::{OriginExpr, OriginNode};
use crate::sampling::Sampler;

pub struct CountSampler<'a, 'b, C, L, N>
where
    C: Counter,
    L: Language + Send + Sync,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Send + Sync,
    N::Data: Send + Sync,
{
    term_count: &'a TermCount<C>,
    graph: &'b EGraph<L, TypeAnalysisWrapper<N>>,
    root: Id,
}

impl<'a, 'b, C, L, N> CountSampler<'a, 'b, C, L, N>
where
    C: Counter,
    L: Language + Send + Sync,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Send + Sync,
    N::Data: Send + Sync,
{
    #[must_use]
    pub fn new(
        term_count: &'a TermCount<C>,
        graph: &'b EGraph<L, TypeAnalysisWrapper<N>>,
        root: Id,
    ) -> Self {
        Self {
            term_count,
            graph,
            root,
        }
    }

    fn sample_inner(
        &self,
        expr: &mut OriginExpr<L>,
        id: Id,
        size: usize,
        rng: &mut ChaCha12Rng,
    ) -> Id {
        let canon_id = self.graph.find(id);
        let eclass = &self.graph[canon_id];
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

        let mut pick = eclass.nodes[pick_idx].clone();
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

                self.sample_inner(expr, c_id, chosen_size, rng)
            })
            .collect::<Vec<_>>();

        let ty = eclass.data.add_type(self.graph, expr);
        for (i, c_id) in pick.children_mut().iter_mut().enumerate() {
            *c_id = children[i];
        }
        let on = OriginNode {
            node: pick,
            ty,
            origin: canon_id.into(),
        };
        expr.add(on)
    }
}

impl<C, L, N> Sampler for CountSampler<'_, '_, C, L, N>
where
    C: Counter,
    L: Language + Send + Sync,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Send + Sync,
    N::Data: Send + Sync,
{
    type Lang = L;

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
    ) -> HashSet<OriginExpr<Self::Lang>>
    where
        F: Fn(&OriginExpr<Self::Lang>) -> bool + Sync,
    {
        super::common::sample_batch::<PARALLEL, _, _>(self, id, samples_per_size, seed, check)
    }

    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> OriginExpr<Self::Lang> {
        let mut expr = OriginExpr::default();
        self.sample_inner(&mut expr, id, size, rng);
        expr
    }

    fn sample_batch_root<const PARALLEL: bool, F>(
        &self,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
        check: &F,
    ) -> HashSet<OriginExpr<Self::Lang>>
    where
        F: Fn(&OriginExpr<Self::Lang>) -> bool + Sync,
    {
        self.sample_batch::<PARALLEL, _>(self.root(), samples_per_size, seed, check)
    }
}
