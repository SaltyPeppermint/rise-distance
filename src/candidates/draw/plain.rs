use egg::{EGraph, Id, RecExpr};
use hashbrown::HashMap;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::Counter;
use crate::candidates::count::CountData;
use crate::candidates::draw::{Drawer, Weigher};
use crate::{MyAnalysis, MyLanguage, OriginLang, stack_children};

pub struct PlainDrawer<'a, 'b, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    term_count: &'a CountData<C>,
    graph: &'b EGraph<L, N>,
    root: Id,
    weigher: W,
}

impl<'a, 'b, C, L, N, W> PlainDrawer<'a, 'b, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    #[must_use]
    pub const fn new(
        term_count: &'a CountData<C>,
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

impl<C, L, N, W> Drawer<C, L, N> for PlainDrawer<'_, '_, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    fn root(&self) -> Id {
        self.root
    }

    fn find(&self, id: Id) -> Id {
        self.graph.find(id)
    }

    fn size_histogram(&self, id: Id) -> Option<&HashMap<usize, C>> {
        self.term_count.data.get(&id)
    }

    fn draw(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let canon_id = self.graph.find(id);
        let eclass = &self.graph[canon_id];
        let child_budget = size - 1;
        let cached = &self.term_count.suffix[&canon_id];

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
                let histogram = self.term_count.data.get(&self.graph.find(c_id));
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
                self.draw(c_id, chosen_size, rng)
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
    use crate::candidates::count::{CountData, count_terms_rooted, root_budgets};
    use crate::candidates::draw::{CountWeigher, NaiveWeigher};
    use crate::langs::math::Math;
    use crate::lower;
    use crate::utils::combined_rng;
    use crate::utils::sym;

    fn rooted_counts(max_size: usize, graph: &EGraph<Math, ()>, root: Id) -> CountData<BigUint> {
        let budgets = root_budgets(graph, root, max_size);
        count_terms_rooted(graph, &budgets)
    }

    #[test]
    fn naive_draw_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let tc = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&tc, &graph, root, NaiveWeigher);

        let mut rng = combined_rng([42]);
        let term = drawer.draw(root, 1, &mut rng);
        assert_eq!(lower(term).to_string(), "a");
    }

    #[test]
    fn naive_draw_picks_valid_choice() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let tc = rooted_counts(10, &graph, a);
        let drawer = PlainDrawer::new(&tc, &graph, a, NaiveWeigher);

        for s in 0..50_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(drawer.draw(a, 1, &mut rng)).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn naive_possible_size_correct() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let tc = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&tc, &graph, root, NaiveWeigher);

        assert!(!drawer.possible_size(root, 1, 0));
        assert!(!drawer.possible_size(root, 3, 0));
        assert!(drawer.possible_size(root, 2, 0));
        assert!(!drawer.possible_size(root, 2, 1));
    }

    #[test]
    fn naive_draw_batch_finds_all_unique() {
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

        let tc = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&tc, &graph, root, NaiveWeigher);

        let result = drawer.draw_root_batch(&[(3, 5)], [1, 2]).unwrap();
        assert!(result.len() <= 6);
    }

    #[test]
    fn count_weighted_draw_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let tc = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&tc, &graph, root, CountWeigher);

        let mut rng = combined_rng([42]);
        let term = drawer.draw(root, 1, &mut rng);
        assert_eq!(lower(term).to_string(), "a");
    }

    #[test]
    fn count_weighted_draw_picks_valid_choice() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let tc = rooted_counts(10, &graph, a);
        let drawer = PlainDrawer::new(&tc, &graph, a, CountWeigher);

        for s in 0..50_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(drawer.draw(a, 1, &mut rng)).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn count_draw_batch_finds_unique() {
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

        let tc = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&tc, &graph, root, CountWeigher);

        let result = drawer.draw_root_batch(&[(3, 5)], [1, 2]).unwrap();

        assert!(result.len() <= 6);
    }

    #[test]
    fn draw_batch_returns_partial_when_size_undersupplied() {
        // Root Add([a-class, b-class]) has 2 * 3 = 6 distinct terms of size 3.
        // Asking for far more than that must return the 6 that exist rather
        // than collapsing the whole batch to None (the empty-pool bug).
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

        let tc = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&tc, &graph, root, CountWeigher);

        let result = drawer
            .draw_root_batch(&[(3, 1000)], [1, 2])
            .expect("undersupplied size should still yield its available terms");
        assert_eq!(
            result.len(),
            6,
            "should return exactly the 6 distinct terms"
        );
    }

    #[test]
    fn draw_batch_none_only_when_all_sizes_empty() {
        // Root Ln(a) has exactly one term of size 2 and nothing at size 5.
        // A batch mixing a satisfiable size with an empty one keeps the
        // satisfiable one; a batch of only empty sizes returns None.
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let tc = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&tc, &graph, root, NaiveWeigher);

        let mixed = drawer
            .draw_root_batch(&[(2, 5), (5, 5)], [1, 2])
            .expect("a non-empty size keeps the batch alive");
        assert_eq!(mixed.len(), 1);

        assert!(
            drawer.draw_root_batch(&[(5, 5)], [1, 2]).is_none(),
            "a wholly empty frontier still returns None"
        );
    }
}
