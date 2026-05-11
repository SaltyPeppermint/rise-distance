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

    fn term_sizes(&self, id: Id) -> Vec<usize> {
        let canon_id = self.graph.find(id);
        self.term_count
            .data()
            .get(&canon_id)
            .map(|h| h.keys().copied().collect())
            .unwrap_or_default()
    }

    fn enumerate_size(&self, id: Id, size: usize) -> Vec<RecExpr<OriginLang<L>>> {
        self.term_count.enumerate_size(self.graph, id, size)
    }

    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let canon_id = self.graph.find(id);
        let eclass = &self.graph[canon_id];
        let child_budget = size - 1;
        let cached = &self.term_count.suffix_cache()[&canon_id];

        let weights = cached
            .iter()
            .map(|suffix| {
                suffix[0]
                    .get(&child_budget)
                    .map_or_else(C::zero, |count| self.weigher.node_weight(count))
            })
            .collect::<Vec<_>>();
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
                let candidates = histogram
                    .into_iter()
                    .flatten()
                    .filter_map(|(&s, count)| {
                        remaining
                            .checked_sub(s)
                            .and_then(|r| suffix[i + 1].get(&r))
                            .map(|rest_count| (s, self.weigher.child_weight(count, rest_count)))
                    })
                    .collect::<Vec<_>>();

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
    use hashbrown::HashSet;
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
    fn n_smallest_returns_all_when_available() {
        // E-class {a, b}: two distinct size-1 terms.
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, a, NaiveWeigher);

        let got = sampler.n_smallest(a, 2).unwrap();
        let strs: HashSet<String> = got.into_iter().map(|t| lower(t).to_string()).collect();
        assert_eq!(strs, HashSet::from_iter(["a".to_owned(), "b".to_owned()]));
    }

    #[test]
    fn n_smallest_none_when_not_enough() {
        // Only one term reachable.
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, a, NaiveWeigher);

        assert!(sampler.n_smallest(a, 2).is_none());
        assert_eq!(sampler.n_smallest(a, 1).unwrap().len(), 1);
    }

    #[test]
    fn n_smallest_crosses_sizes() {
        // x's e-class has `x` (size 1) and `(ln y)` (size 2) via union.
        // Asking for 2 should yield both; smallest comes first.
        let mut graph = EGraph::<Math, ()>::new(());
        let x = graph.add(sym("x"));
        let y = graph.add(sym("y"));
        let lny = graph.add(Math::Ln(y));
        graph.union(x, lny);
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, x, NaiveWeigher);

        let got = sampler.n_smallest(x, 2).unwrap();
        let two: HashSet<String> = got.into_iter().map(|t| lower(t).to_string()).collect();
        assert_eq!(
            two,
            HashSet::from_iter(["x".to_owned(), "(ln y)".to_owned()])
        );

        // Asking for just 1 should give the smallest (size 1).
        let got_one = sampler.n_smallest(x, 1).unwrap();
        let one: HashSet<String> = got_one.into_iter().map(|t| lower(t).to_string()).collect();
        assert_eq!(one, HashSet::from_iter(["x".to_owned()]));
    }

    #[test]
    fn n_smallest_is_stable() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, a, NaiveWeigher);

        let first = sampler.n_smallest(a, 2).unwrap();
        let second = sampler.n_smallest(a, 2).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn n_smallest_is_monotone() {
        // Build an e-class with at least 6 distinct terms.
        // a, b, c unioned -> 3 size-1 terms.
        // ln(a) reachable via the unified class -> size-2 terms: (ln a), (ln b), (ln c).
        // That's 6 distinct terms total.
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        let c = graph.add(sym("c"));
        graph.union(a, b);
        graph.union(a, c);
        let lna = graph.add(Math::Ln(a));
        graph.union(a, lna);
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let sampler = PlainSampler::new(&tc, &graph, a, NaiveWeigher);

        let six = sampler.n_smallest(a, 6).unwrap();
        let three = sampler.n_smallest(a, 3).unwrap();
        let two = sampler.n_smallest(a, 2).unwrap();
        assert_eq!(six.len(), 6);
        assert_eq!(three.len(), 3);
        assert_eq!(two.len(), 2);
        assert!(two.is_subset(&three));
        assert!(three.is_subset(&six));

        // Repeated calls return the exact same set.
        assert_eq!(sampler.n_smallest(a, 3).unwrap(), three);
        assert_eq!(sampler.n_smallest(a, 2).unwrap(), two);
    }
}
