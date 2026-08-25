//!  Weighted frontier-candidate drawing.
//!
//! [`FrontierDrawer`] preserves the draw behavior of the
//! former frontier drawer: every requested term is drawn independently, and a
//! [`Weigher`] controls whether feasible derivation choices are uniform locally
//! or weighted by their term counts.
//! The drawer chooses between productions returned by
//! [`FrontierSpace`], so they cannot accidentally construct a term that
//! violates the requested frontier state.

use egg::{EGraph, Id, RecExpr};
use hashbrown::HashMap;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::Counter;
use crate::candidates::count::{NodeMatch, NovelTermCount};
use crate::candidates::draw::{Drawer, Weigher};
use crate::candidates::{convolve_at, suffix_convolutions};
use crate::{MyAnalysis, MyLanguage, OriginLang, stack_children};

/// Draws each frontier term independently using the supplied local weighting
/// policy.
///
/// `CountWeigher` draws proportionally to the number of complete terms below
/// each derivation choice. `NaiveWeigher` gives every feasible local choice
/// equal weight. Neither policy coordinates choices across a batch
pub struct FrontierDrawer<'a, 'g, C, L, N, W>
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

impl<'a, 'g, C, L, N, W> FrontierDrawer<'a, 'g, C, L, N, W>
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

impl<C, L, N, W> Drawer<C, L, N> for FrontierDrawer<'_, '_, C, L, N, W>
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
        self.space
            .construct(id, size, FrontierState::OutsidePrev, &self.weigher, rng)
    }
}

/// Whether a current-graph extraction is outside the previous graph or agrees
/// with one particular previous e-class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FrontierState {
    OutsidePrev,
    InsidePrev(Id),
}

/// One feasible root-production/profile choice at a `(class, size, state)`.
pub(crate) struct FrontierBranch<'a, C> {
    pub node_idx: usize,
    pub child_states: Vec<FrontierState>,
    pub count: C,
    child_hists: Vec<&'a HashMap<usize, C>>,
}

/// One feasible size for the current child, together with the counts needed
/// by count-proportional policies.
pub(crate) struct ChildSizeChoice<C> {
    pub size: usize,
    pub child_count: C,
    pub rest_count: C,
}

/// The constrained derivation space.
///
/// `OutsidePrev` corresponds to the old `Novel` recursion mode and
/// `InsidePrev(pc)` to the old `AgreeWith(pc)` mode. Counts and match data come
/// from [`NovelTermCount`]; this type turns them into feasible productions and
/// performs policy-directed recursive construction.
struct FrontierSpace<'a, 'g, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    counts: &'a NovelTermCount<C>,
    graph: &'g EGraph<L, N>,
}

impl<'a, 'g, C, L, N> FrontierSpace<'a, 'g, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    const fn new(counts: &'a NovelTermCount<C>, graph: &'g EGraph<L, N>) -> Self {
        Self { counts, graph }
    }

    const fn graph(&self) -> &'g EGraph<L, N> {
        self.graph
    }

    pub const fn counts(&self) -> &'a NovelTermCount<C> {
        self.counts
    }

    fn pick_branch<W>(
        weigher: &W,
        choices: &[FrontierBranch<'_, C>],
        rng: &mut ChaCha12Rng,
    ) -> usize
    where
        W: Weigher<C>,
    {
        WeightedIndex::new(
            choices
                .iter()
                .map(|choice| weigher.node_weight(&choice.count)),
        )
        .expect("frontier branch weights contain a positive choice")
        .sample(rng)
    }

    fn pick_child_size<W>(
        weigher: &W,
        choices: &[ChildSizeChoice<C>],
        rng: &mut ChaCha12Rng,
    ) -> usize
    where
        W: Weigher<C>,
    {
        WeightedIndex::new(
            choices
                .iter()
                .map(|choice| weigher.child_weight(&choice.child_count, &choice.rest_count)),
        )
        .expect("frontier child-size weights contain a positive choice")
        .sample(rng)
    }

    /// Construct a term in `state` using only feasible frontier productions.
    fn construct<W>(
        &self,
        id: Id,
        size: usize,
        state: FrontierState,
        weigher: &W,
        rng: &mut ChaCha12Rng,
    ) -> RecExpr<OriginLang<L>>
    where
        W: Weigher<C>,
    {
        let curr = self.graph().find(id);

        let branches = self.branches(curr, size, state);
        assert!(
            !branches.is_empty(),
            "frontier state has at least one feasible production"
        );

        let branch_idx = Self::pick_branch(weigher, &branches, rng);
        let branch = &branches[branch_idx];
        let node = &self.graph()[curr].nodes[branch.node_idx];
        let child_budget = size - 1;
        let suffix = suffix_convolutions(&branch.child_hists, child_budget);

        let mut remaining = child_budget;
        let mut children = Vec::with_capacity(node.children().len());
        for (child_index, &child_id) in node.children().iter().enumerate() {
            let choices = branch.child_hists[child_index]
                .iter()
                .filter_map(|(&child_size, child_count)| {
                    let rest_size = remaining.checked_sub(child_size)?;
                    let rest_count = suffix[child_index + 1].get(&rest_size)?;
                    (*rest_count != C::zero()).then(|| ChildSizeChoice {
                        size: child_size,
                        child_count: child_count.clone(),
                        rest_count: rest_count.clone(),
                    })
                })
                .collect::<Vec<_>>();

            assert!(
                !choices.is_empty(),
                "chosen frontier production has a feasible child-size split"
            );
            let size_idx = Self::pick_child_size(weigher, &choices, rng);
            let child_size = choices[size_idx].size;
            remaining -= child_size;
            let child = self.construct(
                child_id,
                child_size,
                branch.child_states[child_index],
                weigher,
                rng,
            );
            children.push(child);
        }

        stack_children(&children, OriginLang::new(node.clone(), curr))
    }

    fn branches(&self, curr: Id, size: usize, state: FrontierState) -> Vec<FrontierBranch<'_, C>> {
        match state {
            FrontierState::OutsidePrev => self.outside_branches(curr, size),
            FrontierState::InsidePrev(prev) => self.inside_branches(curr, size, prev),
        }
    }

    fn inside_branches(&self, curr: Id, size: usize, prev: Id) -> Vec<FrontierBranch<'_, C>> {
        let eclass = &self.graph()[curr];
        let child_budget = size - 1;

        eclass
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(node_idx, node)| {
                self.counts
                    .matches_of(self.graph, curr, node_idx)
                    .iter()
                    .filter(move |m| m.prev_class == prev)
                    .filter_map(move |m| {
                        let child_hists = node
                            .children()
                            .iter()
                            .zip(m.prev_children.iter())
                            .map(|(child, &pc)| self.counts.joint_histogram(self.graph, *child, pc))
                            .collect::<Option<Vec<_>>>()?;
                        let count = convolve_at::<C>(&child_hists, child_budget)?;
                        Some(FrontierBranch {
                            node_idx,
                            child_states: m
                                .prev_children
                                .iter()
                                .copied()
                                .map(FrontierState::InsidePrev)
                                .collect(),
                            count,
                            child_hists,
                        })
                    })
            })
            .collect()
    }

    fn outside_branches(&self, curr: Id, size: usize) -> Vec<FrontierBranch<'_, C>> {
        let eclass = &self.graph()[curr];
        let child_budget = size - 1;

        eclass
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(node_idx, node)| {
                let matches = self.counts.matches_of(self.graph, curr, node_idx);
                let children = node.children();
                let slot_options = children
                    .iter()
                    .map(|child| {
                        let mut options = vec![FrontierState::OutsidePrev];
                        options.extend(
                            self.counts
                                .cover_of(self.graph, *child)
                                .iter()
                                .copied()
                                .map(FrontierState::InsidePrev),
                        );
                        options
                    })
                    .collect::<Vec<_>>();

                enumerate_profiles(&slot_options)
                    .into_iter()
                    .filter(|profile| !completes_some_match(profile, matches))
                    .filter_map(move |child_states| {
                        let child_hists = children
                            .iter()
                            .zip(child_states.iter())
                            .map(|(child, state)| match state {
                                FrontierState::OutsidePrev => {
                                    self.counts.novel_histogram(self.graph, *child)
                                }
                                FrontierState::InsidePrev(pc) => {
                                    self.counts.joint_histogram(self.graph, *child, *pc)
                                }
                            })
                            .collect::<Option<Vec<_>>>()?;
                        let count = convolve_at::<C>(&child_hists, child_budget)?;
                        Some(FrontierBranch {
                            node_idx,
                            child_states,
                            count,
                            child_hists,
                        })
                    })
            })
            .collect()
    }
}

fn enumerate_profiles<T: Clone>(slot_options: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut profiles = vec![Vec::new()];
    for slot in slot_options {
        let mut next = Vec::with_capacity(profiles.len() * slot.len());
        for prefix in &profiles {
            for option in slot {
                let mut profile = prefix.clone();
                profile.push(option.clone());
                next.push(profile);
            }
        }
        profiles = next;
    }
    profiles
}

fn completes_some_match(profile: &[FrontierState], matches: &[NodeMatch]) -> bool {
    matches.iter().any(|m| {
        profile.len() == m.prev_children.len()
            && profile
                .iter()
                .zip(m.prev_children.iter())
                .all(|(state, &pc)| *state == FrontierState::InsidePrev(pc))
    })
}

#[cfg(test)]
mod tests {
    use egg::EGraph;
    use num::BigUint;

    use super::*;
    use crate::candidates::draw::CountWeigher;
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
        let drawer = FrontierDrawer::new(&novel, &curr, root, CountWeigher);

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
        let drawer = FrontierDrawer::new(&novel, &curr, root, CountWeigher);

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
        let drawer = FrontierDrawer::new(&novel, &graph, a, CountWeigher);

        assert!(!drawer.possible_size(a, 1, 0));
    }
}
