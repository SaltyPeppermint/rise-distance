//! Weighted frontier-candidate drawing.
//!
//! Every requested term is drawn independently, and a [`Weigher`] controls
//! whether feasible derivation choices are uniform locally or weighted by their
//! term counts.

use egg::{EGraph, Id, RecExpr};
use hashbrown::HashMap;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::Counter;
use crate::candidates::count::{NodeMatch, NovelTermCount, RootBudgets};
use crate::candidates::count::{
    NodeMatches, count_terms_rooted, enumerate_matches_rooted, find_novel_root_sizes,
    prune_matches, root_budgets,
};
use crate::candidates::draw::{CountWeigher, DrawerPackage, UniformWeigher};
use crate::candidates::draw::{Drawer, Weigher};
use crate::candidates::greedy_distribute_alloc;
use crate::candidates::{convolve_at, suffix_convolutions};
use crate::cli::Policy;
use crate::eqsat::EqsatResult;
use crate::{MyAnalysis, MyLanguage, OriginLang, stack_children};

/// Draws each frontier term independently using the supplied local weighting
/// policy.
///
/// `CountWeigher` draws proportionally to the number of complete terms below
/// each derivation choice. `NaiveWeigher` gives every feasible local choice
/// equal weight. Neither policy coordinates choices across a batch.
pub struct FrontierDrawer<'a, 'g, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    counts: &'a NovelTermCount<C>,
    graph: &'g EGraph<L, N>,
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
            counts,
            graph,
            root,
            weigher,
        }
    }

    fn pick_branch(&self, choices: &[Branch<'_, C>], rng: &mut ChaCha12Rng) -> usize {
        WeightedIndex::new(
            choices
                .iter()
                .map(|choice| self.weigher.node_weight(&choice.count)),
        )
        .expect("frontier branch weights contain a positive choice")
        .sample(rng)
    }

    fn histogram(&self, child: Id, state: State) -> Option<&HashMap<usize, C>> {
        match state {
            State::Novel => self.counts.novel_histogram(self.graph, child),
            State::SharedWith(prev) => self.counts.joint_histogram(self.graph, child, prev),
        }
    }

    fn make_branch(
        &self,
        curr: Id,
        node_idx: usize,
        child_states: Vec<State>,
        child_budget: usize,
    ) -> Option<Branch<'_, C>> {
        let child_hists = self.graph[curr].nodes[node_idx]
            .children()
            .iter()
            .copied()
            .zip(child_states.iter().copied())
            .map(|(child, state)| self.histogram(child, state))
            .collect::<Option<Vec<_>>>()?;
        let count = convolve_at::<C>(&child_hists, child_budget)?;
        Some(Branch {
            node_idx,
            child_states,
            count,
            child_hists,
        })
    }

    /// Construct a term in `state` using only feasible frontier productions.
    fn construct(
        &self,
        id: Id,
        size: usize,
        state: State,
        rng: &mut ChaCha12Rng,
    ) -> RecExpr<OriginLang<L>> {
        let curr = self.graph.find(id);

        let branches = self.branches(curr, size, state);
        assert!(
            !branches.is_empty(),
            "frontier state has at least one feasible production"
        );

        let branch_idx = self.pick_branch(&branches, rng);
        let branch = &branches[branch_idx];
        let node = &self.graph[curr].nodes[branch.node_idx];
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
                    (*rest_count != C::zero()).then(|| {
                        (
                            child_size,
                            self.weigher.child_weight(child_count, rest_count),
                        )
                    })
                })
                .collect::<Vec<_>>();

            assert!(
                !choices.is_empty(),
                "chosen frontier production has a feasible child-size split"
            );
            let size_idx = WeightedIndex::new(choices.iter().map(|(_, weight)| weight))
                .expect("frontier child-size weights contain a positive choice")
                .sample(rng);
            let child_size = choices[size_idx].0;
            remaining -= child_size;
            let child = self.construct(child_id, child_size, branch.child_states[child_index], rng);
            children.push(child);
        }

        stack_children(&children, OriginLang::new(node.clone(), curr))
    }

    fn branches(&self, curr: Id, size: usize, state: State) -> Vec<Branch<'_, C>> {
        match state {
            State::Novel => self.novel_branches(curr, size),
            State::SharedWith(prev) => self.shared_branches(curr, size, prev),
        }
    }

    fn shared_branches(&self, curr: Id, size: usize, prev: Id) -> Vec<Branch<'_, C>> {
        let eclass = &self.graph[curr];
        let child_budget = size - 1;

        eclass
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(node_idx, _)| {
                self.counts
                    .matches_of(self.graph, curr, node_idx)
                    .iter()
                    .filter(move |m| m.prev_class == prev)
                    .filter_map(move |matched| {
                        let child_states = matched
                            .prev_children
                            .iter()
                            .copied()
                            .map(State::SharedWith)
                            .collect();
                        self.make_branch(curr, node_idx, child_states, child_budget)
                    })
            })
            .collect()
    }

    fn novel_branches(&self, curr: Id, size: usize) -> Vec<Branch<'_, C>> {
        let eclass = &self.graph[curr];
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
                        let mut options = vec![State::Novel];
                        options.extend(
                            self.counts
                                .cover_of(self.graph, *child)
                                .iter()
                                .copied()
                                .map(State::SharedWith),
                        );
                        options
                    })
                    .collect::<Vec<_>>();

                enumerate_profiles(&slot_options)
                    .into_iter()
                    .filter(|profile| !completes_some_match(profile, matches))
                    .filter_map(move |child_states| {
                        self.make_branch(curr, node_idx, child_states, child_budget)
                    })
            })
            .collect()
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
        self.graph.find(id)
    }

    fn size_histogram(&self, id: Id) -> Option<&HashMap<usize, C>> {
        self.counts.data().get(&self.find(id))
    }

    fn draw(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        self.construct(id, size, State::Novel, rng)
    }
}

/// Whether an extraction is novel or shared with one previous e-class.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Novel,
    SharedWith(Id),
}

/// One feasible root-production/profile choice at a `(class, size, state)`.
struct Branch<'a, C> {
    node_idx: usize,
    child_states: Vec<State>,
    count: C,
    child_hists: Vec<&'a HashMap<usize, C>>,
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

fn completes_some_match(profile: &[State], matches: &[NodeMatch]) -> bool {
    matches.iter().any(|m| {
        profile.len() == m.prev_children.len()
            && profile
                .iter()
                .zip(m.prev_children.iter())
                .all(|(state, &pc)| *state == State::SharedWith(pc))
    })
}

/// Final e-graph and complete count tables for frontier candidate construction.
///
/// Construction consumes [`EqsatResult`] and discards its run metadata.
pub struct FrontierPackage<C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    egraph: EGraph<L, N>,
    counts: NovelTermCount<C>,
    min_size: usize,
    max_size: usize,
    root: Id,
}

impl<C, L, N> FrontierPackage<C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    /// Build counts through `max_size` relative to the previous boundary.
    /// Returns `None` for an empty frontier.
    #[must_use]
    pub fn build(result: EqsatResult<L, N>, max_size: usize) -> Option<FrontierPackage<C, L, N>> {
        let curr = result.curr();
        let root = curr.find(result.root());
        let budgets = root_budgets(curr, root, max_size);

        let prev = result.prev_index();
        let mut matches = enumerate_matches_rooted(curr, &prev, &budgets);
        drop(prev);
        prune_matches(curr, &mut matches, &budgets);
        Self::from_rooted_matches(result, max_size, matches, &budgets)
    }

    /// Finish [`build`](Self::build) from matches already restricted
    /// to the supplied final root budgets.
    fn from_rooted_matches(
        result: EqsatResult<L, N>,
        max_size: usize,
        matches: NodeMatches,
        budgets: &RootBudgets,
    ) -> Option<FrontierPackage<C, L, N>> {
        let (egraph, root) = result.into_curr();
        let plain = count_terms_rooted(&egraph, budgets);
        let counts = NovelTermCount::from_rooted_matches(&egraph, plain, matches, budgets);

        let root = egraph.find(root);
        let histogram = counts.data().get(&root)?;

        let min_size = histogram.keys().min().copied().unwrap_or(1);
        Some(FrontierPackage {
            egraph,
            counts,
            min_size,
            max_size,
            root,
        })
    }

    /// Build a package ending at the `novel_size_goal`-th novel root size.
    ///
    /// The exact scan stops at the cap `start_size + max_retries * retry_step`.
    /// See `docs/counting/novel_size_search.md`.
    ///
    /// # Errors
    ///
    /// Returns the cap if the scan or package finds too few novel sizes.
    ///
    /// # Panics
    ///
    /// Panics if `novel_size_goal` is zero or writing to `log` fails.
    pub fn build_through_novel_sizes<W: std::fmt::Write>(
        result: EqsatResult<L, N>,
        start_size: usize,
        max_retries: usize,
        retry_step: usize,
        min_extractable: usize,
        log: &mut W,
    ) -> Result<(usize, Self), usize> {
        let cap = start_size + max_retries * retry_step;

        let prev = result.prev_index();
        let curr = result.curr();
        let root = curr.find(result.root());
        let cap_budgets = root_budgets(curr, root, cap);
        let mut matches = enumerate_matches_rooted(curr, &prev, &cap_budgets);

        drop(prev);

        let max_size =
            match find_novel_root_sizes(curr, root, &matches, min_extractable, &cap_budgets) {
                Ok(max_size) => max_size,
                Err(term_count) => {
                    writeln!(
                        log,
                        "found insufficient ({term_count}) extractable terms with cap={cap}",
                    )
                    .unwrap();
                    return Err(cap);
                }
            };

        let final_budgets = root_budgets(curr, root, max_size);
        prune_matches(curr, &mut matches, &final_budgets);

        let Some(package) = Self::from_rooted_matches(result, max_size, matches, &final_budgets)
        else {
            writeln!(
                log,
                "package construction found no novel terms (max_size={max_size})"
            )
            .unwrap();
            return Err(cap);
        };
        if package.root_histogram().len() < min_extractable {
            writeln!(
                log,
                "package construction found fewer than {min_extractable} novel sizes \
                 (max_size={max_size})"
            )
            .unwrap();
            return Err(cap);
        }

        Ok((max_size, package))
    }
}

impl<C, L, N> DrawerPackage<C, L, N> for FrontierPackage<C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    /// Novel root-term counts by size.
    ///
    /// # Panics
    ///
    /// Panics if package construction violated its root-histogram invariant.
    fn root_histogram(&self) -> &HashMap<usize, C> {
        self.counts
            .data()
            .get(&self.root)
            .expect("root histogram present iff build returned Some")
    }

    /// Draw exact root candidates absent from the previous boundary.
    fn draw_candidates(
        &self,
        count: usize,
        policy: Policy,
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let histogram = self.counts.data().get(&self.root)?;

        let requests = greedy_distribute_alloc(self.min_size, self.max_size, count, histogram);

        match policy {
            Policy::Uniform => {
                FrontierDrawer::new(&self.counts, &self.egraph, self.root, UniformWeigher)
                    .draw_root_batch(&requests, seed)
            }
            Policy::Count => {
                FrontierDrawer::new(&self.counts, &self.egraph, self.root, CountWeigher)
                    .draw_root_batch(&requests, seed)
            } //   Policy::SmallestOverall => Some(vec![
              //         PlainDrawer::new(self.counts.plain(), &self.egraph, self.root, NaiveWeigher)
              //             .smallest(self.root),
              //     ]),
              //     Policy::SmallestNovel => Some(vec![
              //         FrontierDrawer::new(&self.counts, &self.egraph, self.root, NaiveWeigher)
              //             .smallest(self.root),
              //     ]),
        }
    }

    fn root(&self) -> Id {
        self.root
    }
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
    fn build_through_novel_sizes_runs_analysis_at_kth_novel_size() {
        // Unioning `a` with the root of (+ a b) creates a cycle: the root
        // class extracts a, (+ a b), (+ (+ a b) b), ... (sizes 1, 3, 5, ...).
        // `a` and (+ a b) already exist in prev, so the novel sizes are
        // 5, 7, 9, ... asking for 3 sizes must yield max_size = 9.
        let mut curr = EGraph::<Math, ()>::new(());
        curr.enable_union_event_recording();
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let apb = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev_raw_node_count = curr.nodes().len();
        let prev_union_event_count = curr.union_event_count();

        curr.union(a, apb);
        curr.rebuild();

        let result =
            EqsatResult::new_for_tests(curr, apb, prev_raw_node_count, prev_union_event_count);
        let mut log = String::new();
        let (used_max_size, package) = FrontierPackage::<BigUint, _, _>::build_through_novel_sizes(
            result, 3, 10, 2, 3, &mut log,
        )
        .expect("build_through_novel_sizes should succeed");

        assert_eq!(used_max_size, 9, "log:\n{log}");
        assert_eq!(package.max_size, 9);
        assert_eq!(package.min_size, 5);
        let mut keys = package.root_histogram().keys().copied().collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, vec![5, 7, 9]);
        assert!(
            package
                .root_histogram()
                .values()
                .all(|c| *c == BigUint::from(1u32))
        );
    }

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
