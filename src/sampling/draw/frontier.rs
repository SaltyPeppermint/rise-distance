use egg::{Analysis, EGraph, Id, Language, RecExpr};
use num_bigint::BigUint;
use rand_chacha::ChaCha12Rng;
use smallvec::SmallVec;

use crate::cli::Policy;
use crate::eqsat::EqsatResult;
use crate::origin::OriginLang;
use crate::previous::PrevId;
use crate::sampling::count::novel::{NodeMatch, NovelTermCount};
use crate::sampling::draw::{self, DrawCursor, Drawer, DrawingError, SamplePackage, SampledTrie};
use crate::sampling::weigher::{Choice, Weigher};
use crate::sampling::{self, INLINE_ARITY};
use crate::utils::{self, HashMap};

/// Draws frontier terms without replacement using the supplied local weighting
/// policy and a pre-order [`SampledTrie`] of previously drawn derivations.
///
/// `Count` weights each remaining branch proportionally to its unsampled
/// completions, giving exact uniform sampling without replacement. `Uniform`
/// gives every non-exhausted local choice equal weight.
struct FrontierDrawer<'a, 'g, L: Language, N: Analysis<L>> {
    counts: &'a NovelTermCount,
    graph: &'g EGraph<L, N>,
    root: Id,
    weigher: Weigher,
}

impl<'a, 'g, L: Language, N: Analysis<L>> FrontierDrawer<'a, 'g, L, N> {
    fn new(
        counts: &'a NovelTermCount,
        graph: &'g EGraph<L, N>,
        root: Id,
        weigher: Weigher,
    ) -> Self {
        Self {
            counts,
            graph,
            root: graph.find(root),
            weigher,
        }
    }

    fn histogram(&self, child: Id, state: State) -> Option<&HashMap<usize, BigUint>> {
        let canon = self.graph.find(child);
        match state {
            State::Novel => self.counts.novel_histogram(canon),
            State::SharedWith(prev) => self.counts.joint_histogram(canon, prev),
        }
    }

    fn make_branch(
        &self,
        curr: Id,
        node_idx: usize,
        child_states: SmallVec<[State; INLINE_ARITY]>,
        child_budget: usize,
    ) -> Option<Branch<'_>> {
        let child_hists = self.graph[curr].nodes[node_idx]
            .children()
            .iter()
            .copied()
            .zip(child_states.iter().copied())
            .map(|(child, state)| self.histogram(child, state))
            .collect::<Option<SmallVec<[_; INLINE_ARITY]>>>()?;
        let count = sampling::convolve_at(&child_hists, child_budget)?;
        Some(Branch {
            node_idx,
            child_states,
            count,
            child_hists,
        })
    }

    /// Construct a term in `state` using only feasible frontier productions
    /// that still have unsampled completions in `cursor`.
    fn construct(
        &self,
        id: Id,
        size: usize,
        state: State,
        rest_total: Option<&BigUint>,
        cursor: &mut DrawCursor<'_>,
    ) -> RecExpr<OriginLang<L>> {
        let curr = self.graph.find(id);
        let active_rest = cursor.active_rest(rest_total);

        let branches = self.branches(curr, size, state);
        assert!(
            !branches.is_empty(),
            "frontier state has at least one feasible production"
        );

        let choices = branches
            .iter()
            .enumerate()
            .map(|(branch_idx, branch)| Choice::branch(branch_idx, &branch.count))
            .collect::<Vec<_>>();
        let branch_idx = self.weigher.pick(&choices, active_rest, cursor);
        cursor.step(branch_idx);

        let branch = &branches[branch_idx];
        let node = &self.graph[curr].nodes[branch.node_idx];
        let children_ids = node.children();
        let children = draw::draw_children(
            self.weigher,
            &branch.child_hists,
            size - 1,
            active_rest,
            cursor,
            |i, child_size, child_rest, cur| {
                self.construct(
                    children_ids[i],
                    child_size,
                    branch.child_states[i],
                    child_rest,
                    cur,
                )
            },
        );

        utils::stack_children(&children, OriginLang::new(node.clone(), curr))
    }

    fn branches(&self, curr: Id, size: usize, state: State) -> Vec<Branch<'_>> {
        match state {
            State::Novel => self.novel_branches(curr, size),
            State::SharedWith(prev) => self.shared_branches(curr, size, prev),
        }
    }

    fn shared_branches(&self, curr: Id, size: usize, prev: PrevId) -> Vec<Branch<'_>> {
        self.graph[curr]
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(node_idx, _)| {
                self.counts
                    .matches_of(curr, node_idx)
                    .iter()
                    .filter(move |m| m.prev_class == prev)
                    .filter_map(move |matched| {
                        let child_states = matched
                            .prev_children
                            .iter()
                            .copied()
                            .map(State::SharedWith)
                            .collect();
                        self.make_branch(curr, node_idx, child_states, size - 1)
                    })
            })
            .collect()
    }

    fn novel_branches(&self, curr: Id, size: usize) -> Vec<Branch<'_>> {
        self.graph[curr]
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(node_idx, node)| {
                let matches = self.counts.matches_of(curr, node_idx);
                let slot_options = node.children().iter().map(|&child| {
                    std::iter::once(State::Novel).chain(
                        self.counts
                            .cover_of(self.graph.find(child))
                            .iter()
                            .copied()
                            .map(State::SharedWith),
                    )
                });

                sampling::cartesian_product(slot_options)
                    .into_iter()
                    .filter(|profile| !completes_some_match(profile, matches))
                    .filter_map(move |child_states| {
                        self.make_branch(curr, node_idx, child_states, size - 1)
                    })
            })
            .collect()
    }
}

impl<L: Language, N: Analysis<L>> Drawer<L> for FrontierDrawer<'_, '_, L, N> {
    fn root_histogram(&self) -> Option<&HashMap<usize, BigUint>> {
        self.counts.novel_histogram(self.root)
    }

    fn draw_next(
        &self,
        size: usize,
        sampled: &mut SampledTrie,
        rng: &mut ChaCha12Rng,
    ) -> RecExpr<OriginLang<L>> {
        let mut cursor = DrawCursor::new(sampled, rng);
        self.construct(
            self.root,
            size,
            State::Novel,
            Some(&BigUint::ONE),
            &mut cursor,
        )
    }
}

/// Whether an extraction is novel or shared with one previous e-class.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Novel,
    SharedWith(PrevId),
}

/// One feasible root-production/profile choice at a `(class, size, state)`.
struct Branch<'a> {
    node_idx: usize,
    child_states: SmallVec<[State; INLINE_ARITY]>,
    count: BigUint,
    child_hists: SmallVec<[&'a HashMap<usize, BigUint>; INLINE_ARITY]>,
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

/// Final e-graph and complete count tables for frontier sample construction.
///
/// Construction consumes [`EqsatResult`] and discards its run metadata.
pub struct FrontierPackage<L: Language, N: Analysis<L>> {
    egraph: EGraph<L, N>,
    counts: NovelTermCount,
    root: Id,
}

impl<L: Language, N: Analysis<L>> FrontierPackage<L, N> {
    fn from_counts(
        egraph: EGraph<L, N>,
        root: Id,
        counts: NovelTermCount,
    ) -> Option<FrontierPackage<L, N>> {
        counts.novel_histogram(root)?;
        Some(FrontierPackage {
            egraph,
            counts,
            root,
        })
    }

    /// Build a package ending at the smallest novel root size that makes at
    /// least `min_extractable` novel terms available.
    ///
    /// The exact scan grows budgets geometrically starting at
    /// `min_size(root) + search_step` and stops at the cap
    /// `start_size + search_steps`.
    /// See `docs/counting/novel_size_search.md`.
    ///
    /// # Errors
    ///
    /// Returns the cap if the scan or package finds too few novel terms.
    pub fn build(
        result: EqsatResult<L, N>,
        start_size: usize,
        search_step: usize,
        search_steps: usize,
        min_extractable: usize,
    ) -> Result<(usize, Self), usize> {
        let cap = start_size + search_steps;
        let (egraph, root, prev_boundary) = result.into_curr_and_boundary();
        let root = egraph.find(root);

        let (max_size, counts) = match NovelTermCount::build(
            &egraph,
            || prev_boundary.prev_index(&egraph),
            root,
            cap,
            search_step,
            min_extractable,
        ) {
            Ok(found) => found,
            Err(term_count) => {
                eprintln!(
                    "found insufficient ({term_count}) extractable terms with cap={cap} when {min_extractable} were expected"
                );
                return Err(cap);
            }
        };

        let Some(package) = Self::from_counts(egraph, root, counts) else {
            eprintln!("package construction found no novel terms (max_size={max_size})");
            return Err(cap);
        };

        Ok((max_size, package))
    }
}

impl<L: Language, N: Analysis<L>> SamplePackage<L> for FrontierPackage<L, N> {
    /// Novel root-term counts by size.
    ///
    /// # Panics
    ///
    /// Panics if package construction violated its root-histogram invariant.
    fn root_histogram(&self) -> &HashMap<usize, BigUint> {
        self.counts
            .novel_histogram(self.root)
            .expect("root histogram present iff build returned Some")
    }

    /// Draw exact root samples absent from the previous boundary.
    fn draw_samples(
        &self,
        count: usize,
        policy: Policy,
        seed: [u64; 2],
    ) -> Result<Vec<RecExpr<OriginLang<L>>>, DrawingError> {
        FrontierDrawer::new(&self.counts, &self.egraph, self.root, policy.into())
            .draw_for_policy(count, policy, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::langs::math::{self, Math};
    use crate::origin;
    use Weigher::{Count, Uniform};

    #[test]
    fn build_runs_analysis_at_kth_novel_size() {
        // Unioning `a` with the root of (+ a b) creates a cycle: the root
        // class extracts a, (+ a b), (+ (+ a b) b), ... (sizes 1, 3, 5, ...).
        // `a` and (+ a b) already exist in prev, so the novel sizes are
        // 5, 7, 9, ... asking for 3 sizes must yield max_size = 9.
        let mut curr = EGraph::<Math, ()>::new(());
        curr.enable_union_event_recording();
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        let apb = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev_raw_node_count = curr.nodes().len();
        let prev_union_event_count = curr.union_event_count();

        curr.union(a, apb);
        curr.rebuild();

        let result =
            EqsatResult::new_for_tests(curr, apb, prev_raw_node_count, prev_union_event_count);
        let (used_max_size, package) =
            FrontierPackage::build(result, 3, 2, 10, 3).expect("build should succeed");

        assert_eq!(used_max_size, 9);
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
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        let root = curr.add(Math::Ln(a));
        curr.rebuild();
        let prev = curr.clone();
        let _ = b;

        curr.union(a, b);
        curr.rebuild();

        let (_, novel) = NovelTermCount::build(&curr, || &prev, root, 5, 2, 1).unwrap();
        let drawer = FrontierDrawer::new(&novel, &curr, root, Count);

        for seed in 0..50_u64 {
            let terms = drawer.draw_batch(&[(2, 1)], [seed, 0]).unwrap();
            let term = origin::lower(terms[0].clone()).to_string();
            assert_eq!(term, "(ln b)", "got non-frontier sample: {term}");
        }
    }

    #[test]
    fn independent_frontier_draw_union_diagonal() {
        // prev: Add(a, b)
        // curr: same plus union(a, b). Add(merged, merged) extracts 4 terms;
        // only Add(a, b) is in prev.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        let root = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.rebuild();

        let (_, novel) = NovelTermCount::build(&curr, || &prev, root, 5, 2, 1).unwrap();
        let drawer = FrontierDrawer::new(&novel, &curr, root, Count);

        for seed in 0..100_u64 {
            let terms = drawer.draw_batch(&[(3, 1)], [seed, 0]).unwrap();
            let term = origin::lower(terms[0].clone()).to_string();
            assert_ne!(term, "(+ a b)", "produced non-frontier term");
            assert!(
                ["(+ a a)", "(+ b a)", "(+ b b)"].contains(&term.as_str()),
                "unexpected term: {term}"
            );
        }
    }

    #[test]
    fn independent_frontier_possible_size_excludes_old_terms() {
        // prev: a, b, ln(a) with a == ln(a) (so size 1 is old).
        // curr: also union(a, b) so ln(b) is novel at size 2, while size 1
        // only contains old leaves.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        let root = curr.add(Math::Ln(a));
        curr.union(a, root);
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.rebuild();

        let (_, novel) = NovelTermCount::build(&curr, || &prev, root, 5, 2, 1).unwrap();
        let drawer = FrontierDrawer::new(&novel, &curr, root, Count);

        assert!(drawer.draw_batch(&[(1, 1)], [0, 0]).is_err());
        assert!(drawer.draw_batch(&[(2, 1)], [0, 0]).is_ok());
    }

    #[test]
    fn frontier_exhaustive_draw_both_policies_and_stable_order() {
        // prev: Add(a, b) with a, b, c distinct.
        // curr: union(a, b) and union(a, c) -> Add(M, M) has 9 terms of size 3;
        // only Add(a, b) existed in prev, so 8 novel terms exist.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        let c = curr.add(math::sym("c"));
        let root = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.union(a, c);
        curr.rebuild();

        let root = curr.find(root);
        let (_, novel) = NovelTermCount::build(&curr, || &prev, root, 5, 2, 8).unwrap();
        assert_eq!(
            novel.novel_histogram(root).unwrap()[&3],
            BigUint::from(8u32)
        );

        let count_terms = FrontierDrawer::new(&novel, &curr, root, Count)
            .draw_batch(&[(3, 8)], [4, 2])
            .expect("frontier count exhaustive draw");
        let uniform_terms = FrontierDrawer::new(&novel, &curr, root, Uniform)
            .draw_batch(&[(3, 8)], [4, 2])
            .expect("frontier uniform exhaustive draw");

        assert_eq!(count_terms.len(), 8);
        assert_eq!(count_terms, uniform_terms);

        let lowered = count_terms
            .iter()
            .map(|t| origin::lower(t.clone()).to_string())
            .collect::<Vec<_>>();
        assert!(!lowered.contains(&"(+ a b)".to_owned()));
        let unique = lowered
            .iter()
            .cloned()
            .collect::<crate::utils::HashSet<_>>();
        assert_eq!(unique.len(), 8);
    }

    #[test]
    fn frontier_skewed_uniform_draw_succeeds_without_rejection() {
        // prev: a single leaf `old` under a 10-step chain of `cos`, plus `sin(s)`.
        // curr: union 16 fresh leaves `n0..n15` into `old` so the deep branch
        // has 16 novel terms while each intermediate level gets 1 novel term via
        // a newly unioned `sin(new_i)` branch -> skewed cardinality under Uniform.
        let mut curr = EGraph::<Math, ()>::new(());
        let old = curr.add(math::sym("old"));
        let mut cur = old;
        let mut old_spine = Vec::new();
        for i in 0..10 {
            let s = curr.add(math::sym(&format!("s{i}")));
            let cos_node = curr.add(Math::Cos(cur));
            let sin_node = curr.add(Math::Sin(s));
            curr.union(cos_node, sin_node);
            old_spine.push((cos_node, s));
            cur = cos_node;
        }
        curr.rebuild();
        let prev = curr.clone();

        for i in 0..16 {
            let fresh = curr.add(math::sym(&format!("n{i}")));
            curr.union(old, fresh);
        }
        // Make each level's `sin` branch also have a single novel term of
        // matching size by unioning a fresh leaf into `s9` (root-level sin has
        // size 2, not size 11). Instead build a matching-size novel branch at
        // every level on top of `s0` so all novel terms have size 11.
        let mut chain_leaf = curr.add(math::sym("fresh_s"));
        for &(level_class, _) in &old_spine {
            let branch = curr.add(Math::Sin(chain_leaf));
            curr.union(level_class, branch);
            chain_leaf = curr.add(Math::Ln(chain_leaf));
        }
        curr.rebuild();

        let root = curr.find(cur);
        let target_size = 11;
        let (_, novel) = NovelTermCount::build(&curr, || &prev, root, target_size, 2, 26).unwrap();
        // 16 deep `cos^10(n_i)` + 10 narrow `cos^k(sin(ln^...))` = 26 novel terms.
        assert_eq!(
            novel.novel_histogram(root).unwrap()[&target_size],
            BigUint::from(26u32)
        );

        let uniform_first = FrontierDrawer::new(&novel, &curr, root, Uniform)
            .draw_batch(&[(target_size, 26)], [17, 19])
            .expect("skewed frontier uniform draw must succeed");
        let uniform_second = FrontierDrawer::new(&novel, &curr, root, Uniform)
            .draw_batch(&[(target_size, 26)], [17, 19])
            .expect("repeated skewed frontier uniform draw must succeed");

        assert_eq!(uniform_first.len(), 26);
        assert_eq!(uniform_first, uniform_second);
        for term in &uniform_first {
            assert!(
                prev.lookup_expr(&origin::lower(term.clone())).is_none(),
                "drawn term must be absent from prev"
            );
        }
    }
}
