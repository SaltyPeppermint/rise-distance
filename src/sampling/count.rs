use egg::{EGraph, Id, RecExpr};
use hashbrown::HashSet;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::MyLanguage;
use crate::count::{Counter, TermCount};
use crate::sampling::{Sampler, common};
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
                // TODO: re-add types or clean up
                // match c_id {
                //     ExprChildId::Nat(nat_id) => OriginTree::from_nat(self.graph, nat_id),
                //     ExprChildId::Data(data_id) => OriginTree::from_data(self.graph, data_id),
                //     ExprChildId::EClass(eclass_id) => self.sample(eclass_id, chosen_size, rng),
                // }
            })
            .collect::<Vec<_>>();

        // TODO: re-add types or clean up
        // OriginTree::new_typed(
        //     pick.clone(),
        //     children,
        //     OriginTree::from_eclass(self.graph, canon_id),
        //     canon_id.into(),
        // )
        stack_children(&children, OriginLang::new(pick.clone(), canon_id))
    }
}

#[cfg(test)]
mod tests {
    use egg::EGraph;
    use num::BigUint;

    use super::*;
    use crate::egg::Math;
    use crate::lower;
    use crate::utils::combined_rng;

    fn sym(name: &str) -> Math {
        Math::Symbol(name.into())
    }

    #[test]
    fn sample_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let tc = TermCount::<BigUint>::new(10, &graph);
        let sampler = CountSampler::new(&tc, &graph, root);

        let mut rng = combined_rng([42]);
        let term = sampler.sample(root, 1, &mut rng);
        assert_eq!(lower(term).to_string(), "a");
    }

    #[test]
    fn sample_picks_valid_choice() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let tc = TermCount::<BigUint>::new(10, &graph);
        let sampler = CountSampler::new(&tc, &graph, a);

        for s in 0..50_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(sampler.sample(a, 1, &mut rng)).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn sample_batch_finds_unique() {
        // (+ a b) where a in {a1, a2} and b in {b1, b2, b3} => 6 unique terms of size 3.
        // possible_size requires count > samples, so request fewer than 6 to keep going.
        let mut graph = EGraph::<Math, ()>::new(());
        let a1 = graph.add(sym("a1"));
        let a2 = graph.add(sym("a2"));
        graph.union(a1, a2);
        let b1 = graph.add(sym("b1"));
        let b2 = graph.add(sym("b2"));
        let b3 = graph.add(sym("b3"));
        graph.union(b1, b2);
        graph.union(b1, b3);
        let root = graph.add(Math::Add([a1, b1]));
        graph.rebuild();

        let tc = TermCount::<BigUint>::new(10, &graph);
        let sampler = CountSampler::new(&tc, &graph, root);

        let result = sampler.sample_batch_root::<false, _>(&[(3, 5)], [1, 2], &|_| true);
        assert!(!result.is_empty());
        assert!(result.len() <= 6);
    }

    #[test]
    fn sample_batch_check_filters() {
        // Reject samples whose displayed form contains "a1".
        let mut graph = EGraph::<Math, ()>::new(());
        let a1 = graph.add(sym("a1"));
        let a2 = graph.add(sym("a2"));
        graph.union(a1, a2);
        let b1 = graph.add(sym("b1"));
        let b2 = graph.add(sym("b2"));
        let b3 = graph.add(sym("b3"));
        graph.union(b1, b2);
        graph.union(b1, b3);
        let root = graph.add(Math::Add([a1, b1]));
        graph.rebuild();

        let tc = TermCount::<BigUint>::new(10, &graph);
        let sampler = CountSampler::new(&tc, &graph, root);

        let result = sampler.sample_batch_root::<false, _>(&[(3, 5)], [1, 2], &|t| {
            !lower(t.clone()).to_string().contains("a1")
        });
        for s in &result {
            assert!(!lower(s.clone()).to_string().contains("a1"));
        }
    }
}
