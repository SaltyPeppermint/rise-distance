//! Independent weighted frontier-candidate drawing.
//!
//! [`IndependentFrontierDrawer`] preserves the draw behavior of the
//! former frontier drawer: every requested term is drawn independently, and a
//! [`Weigher`] controls whether feasible derivation choices are balanced
//! locally or weighted by their term counts. The frontier constraint itself
//! lives in [`super::space::FrontierSpace`] and is shared with other
//! frontier selection policies.

use egg::{EGraph, Id, RecExpr};
use hashbrown::HashMap;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use super::space::{
    BranchSite, ChildSizeChoice, ChildSizeSite, FrontierBranch, FrontierPolicy, FrontierSpace,
    FrontierState,
};
use crate::Counter;
use crate::candidates::exact::count::NovelTermCount;
use crate::candidates::exact::draw::{ExactDrawer, Weigher};
use crate::{MyAnalysis, MyLanguage, OriginLang};

/// Draws each frontier term independently using the supplied local weighting
/// policy.
///
/// `CountWeigher` draws proportionally to the number of complete terms below
/// each derivation choice. `NaiveWeigher` gives every feasible local choice
/// equal weight. Neither policy coordinates choices across a batch; use
/// `BalancedFrontierDrawer` when coverage across drawn candidates matters.
pub struct IndependentFrontierDrawer<'a, 'g, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    space: FrontierSpace<'a, 'g, C, L, N>,
    root: Id,
    weigher: W,
}

impl<'a, 'g, C, L, N, W> IndependentFrontierDrawer<'a, 'g, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    #[must_use]
    pub const fn new(
        counts: &'a NovelTermCount<C>,
        graph: &'g EGraph<L, N>,
        root: Id,
        weigher: W,
    ) -> Self {
        Self {
            space: FrontierSpace::new(counts, graph),
            root,
            weigher,
        }
    }
}

struct IndependentPolicy<'a, W> {
    weigher: &'a W,
}

impl<C, W> FrontierPolicy<C> for IndependentPolicy<'_, W>
where
    C: Counter,
    W: Weigher<C>,
{
    fn pick_branch(
        &mut self,
        _site: BranchSite,
        choices: &[FrontierBranch<'_, C>],
        rng: &mut ChaCha12Rng,
    ) -> usize {
        WeightedIndex::new(
            choices
                .iter()
                .map(|choice| self.weigher.node_weight(&choice.count)),
        )
        .expect("frontier branch weights contain a positive choice")
        .sample(rng)
    }

    fn pick_child_size(
        &mut self,
        _site: ChildSizeSite<'_>,
        choices: &[ChildSizeChoice<C>],
        rng: &mut ChaCha12Rng,
    ) -> usize {
        WeightedIndex::new(choices.iter().map(|choice| {
            self.weigher
                .child_weight(&choice.child_count, &choice.rest_count)
        }))
        .expect("frontier child-size weights contain a positive choice")
        .sample(rng)
    }
}

impl<C, L, N, W> ExactDrawer<C, L, N> for IndependentFrontierDrawer<'_, '_, C, L, N, W>
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
        self.space.graph().find(id)
    }

    fn size_histogram(&self, id: Id) -> Option<&HashMap<usize, C>> {
        self.space.counts().data().get(&self.find(id))
    }

    fn draw(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let mut policy = IndependentPolicy {
            weigher: &self.weigher,
        };
        self.space
            .construct(id, size, FrontierState::OutsidePrev, &mut policy, rng)
    }
}

#[cfg(test)]
mod tests {
    use egg::EGraph;
    use num::BigUint;

    use super::*;
    use crate::candidates::exact::draw::CountWeigher;
    use crate::langs::math::Math;
    use crate::lower;
    use crate::utils::{combined_rng, sym};

    #[test]
    fn independent_frontier_draw_picks_only_frontier_term() {
        // prev: a, b, ln(a) (no union).
        // curr: same plus union(a, b). Now ln(b) is extractable from curr's
        // root but not from any prev class.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let root = curr.add(Math::Ln(a));
        curr.rebuild();
        let prev = curr.clone();
        let _ = b;

        curr.union(a, b);
        curr.rebuild();

        let novel = NovelTermCount::<BigUint>::rooted_for_tests(5, &curr, &prev, root);
        let drawer = IndependentFrontierDrawer::new(&novel, &curr, root, CountWeigher);

        for seed in 0..50_u64 {
            let mut rng = combined_rng([seed]);
            let term = lower(drawer.draw(root, 2, &mut rng)).to_string();
            assert_eq!(term, "(ln b)", "got non-frontier candidate: {term}");
        }
    }

    #[test]
    fn independent_frontier_draw_union_diagonal() {
        // prev: Add(a, b)
        // curr: same plus union(a, b). Add(merged, merged) extracts 4 terms;
        // only Add(a, b) is in prev.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let root = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.rebuild();

        let novel = NovelTermCount::<BigUint>::rooted_for_tests(5, &curr, &prev, root);
        let drawer = IndependentFrontierDrawer::new(&novel, &curr, root, CountWeigher);

        for seed in 0..100_u64 {
            let mut rng = combined_rng([seed]);
            let term = lower(drawer.draw(root, 3, &mut rng)).to_string();
            assert_ne!(term, "(+ a b)", "produced non-frontier term");
            assert!(
                ["(+ a a)", "(+ b a)", "(+ b b)"].contains(&term.as_str()),
                "unexpected term: {term}"
            );
        }
    }

    #[test]
    fn independent_frontier_possible_size_excludes_old_terms() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        graph.rebuild();

        let novel = NovelTermCount::<BigUint>::rooted_for_tests(5, &graph, &graph, a);
        let drawer = IndependentFrontierDrawer::new(&novel, &graph, a, CountWeigher);

        assert!(!drawer.possible_size(a, 1, 0));
    }
}
