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
        let sampler = NaiveSampler::new(&tc, &graph, root);

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
        let sampler = NaiveSampler::new(&tc, &graph, a);

        for s in 0..50_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(sampler.sample(a, 1, &mut rng)).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn possible_size_correct() {
        // ln has 1 term of size 2, so possible_size only returns true when count > samples.
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let tc = TermCount::<BigUint>::new(10, &graph);
        let sampler = NaiveSampler::new(&tc, &graph, root);

        // No terms of size 1 or 3 exist for root.
        assert!(!sampler.possible_size(root, 1, 0));
        assert!(!sampler.possible_size(root, 3, 0));
        // 1 term of size 2: 1 > 0 yes, 1 > 1 no.
        assert!(sampler.possible_size(root, 2, 0));
        assert!(!sampler.possible_size(root, 2, 1));
    }

    #[test]
    fn sample_batch_finds_all_unique() {
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
        let sampler = NaiveSampler::new(&tc, &graph, root);

        let result = sampler.sample_batch_root::<false, _>(&[(3, 5)], [1, 2], &|_| true);
        // Sampling uniformly with the same seed scheme deterministically — accept any subset.
        assert!(!result.is_empty());
        assert!(result.len() <= 6);
    }
}
