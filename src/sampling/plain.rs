use egg::{EGraph, Id, RecExpr};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

// TODO: reenable zs_min_distance sampler
// pub use zs_min_distance::ZSDistanceSampler;

use crate::count::{Counter, PlainTermCount};
use crate::sampling::Weigher;
use crate::{MyAnalysis, MyLanguage, OriginLang, Sampler, stack_children};

pub struct PlainSampler<'a, 'b, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    term_count: &'a PlainTermCount<C>,
    graph: &'b EGraph<L, N>,
    root: Id,
    weigher: W,
}

impl<'a, 'b, C, L, N, W> PlainSampler<'a, 'b, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    #[must_use]
    pub fn new(
        term_count: &'a PlainTermCount<C>,
        graph: &'b EGraph<L, N>,
        root: Id,
        weigher: W,
    ) -> Self {
        Self {
            term_count,
            graph,
            root,
            weigher,
        }
    }
}

impl<C, L, N, W> Sampler<C, L, N> for PlainSampler<'_, '_, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    fn root(&self) -> Id {
        self.root
    }

    fn possible_size(&self, id: Id, size: usize, samples: u64) -> bool {
        let canon_id = self.graph.find(id);
        let Some(count) = self
            .term_count
            .data()
            .get(&canon_id)
            .and_then(|h| h.get(&size))
        else {
            return false;
        };
        samples.try_into().is_ok_and(|s: C| count > &s)
    }

    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let canon_id = self.graph.find(id);
        let eclass = &self.graph[canon_id];
        let child_budget = size - 1;
        let cached = &self.term_count.suffix_cache()[&canon_id];

        let weights: Vec<C> = cached
            .iter()
            .map(|suffix| {
                suffix[0]
                    .get(&child_budget)
                    .map_or_else(C::zero, |count| self.weigher.node_weight(count))
            })
            .collect();
        let pick_idx = WeightedIndex::new(&weights).unwrap().sample(rng);

        let pick = &eclass.nodes[pick_idx];
        let suffix = &cached[pick_idx];

        let mut remaining = child_budget;
        let children = pick
            .children()
            .iter()
            .enumerate()
            .map(|(i, &c_id)| {
                let histogram = self.term_count.child_histogram(c_id, self.graph);
                let candidates: Vec<(usize, C)> = histogram
                    .into_iter()
                    .flatten()
                    .filter_map(|(&s, count)| {
                        remaining
                            .checked_sub(s)
                            .and_then(|r| suffix[i + 1].get(&r))
                            .map(|rest_count| (s, self.weigher.child_weight(count, rest_count)))
                    })
                    .collect();

                let dist = WeightedIndex::new(candidates.iter().map(|(_, w)| w)).unwrap();
                let chosen_size = candidates[dist.sample(rng)].0;
                remaining -= chosen_size;
                self.sample(c_id, chosen_size, rng)
            })
            .collect::<Vec<_>>();

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
    use crate::sampling::{CountWeigher, NaiveWeigher};
    use crate::utils::combined_rng;

    fn sym(name: &str) -> Math {
        Math::Symbol(name.into())
    }

    #[test]
    fn naive_sample_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, root, NaiveWeigher);

        let mut rng = combined_rng([42]);
        let term = sampler.sample(root, 1, &mut rng);
        assert_eq!(lower(term).to_string(), "a");
    }

    #[test]
    fn naive_sample_picks_valid_choice() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, a, NaiveWeigher);

        for s in 0..50_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(sampler.sample(a, 1, &mut rng)).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn naive_possible_size_correct() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, root, NaiveWeigher);

        assert!(!sampler.possible_size(root, 1, 0));
        assert!(!sampler.possible_size(root, 3, 0));
        assert!(sampler.possible_size(root, 2, 0));
        assert!(!sampler.possible_size(root, 2, 1));
    }

    #[test]
    fn naive_sample_batch_finds_all_unique() {
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

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, root, NaiveWeigher);

        let result = sampler.sample_batch_root::<false>(&[(3, 5)], [1, 2]);
        assert!(!result.is_empty());
        assert!(result.len() <= 6);
    }

    #[test]
    fn count_sample_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, root, CountWeigher);

        let mut rng = combined_rng([42]);
        let term = sampler.sample(root, 1, &mut rng);
        assert_eq!(lower(term).to_string(), "a");
    }

    #[test]
    fn count_sample_picks_valid_choice() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, a, CountWeigher);

        for s in 0..50_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(sampler.sample(a, 1, &mut rng)).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn count_sample_batch_finds_unique() {
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

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, root, CountWeigher);

        let result = sampler.sample_batch_root::<false>(&[(3, 5)], [1, 2]);
        assert!(!result.is_empty());
        assert!(result.len() <= 6);
    }

    #[test]
    fn count_sample_batch_check_filters() {
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

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, root, CountWeigher);

        let result = sampler.sample_batch_root::<false>(&[(3, 5)], [1, 2]);
        for s in &result {
            assert!(!lower(s.clone()).to_string().contains("a1"));
        }
    }
}
