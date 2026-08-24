use egg::{EGraph, Id, RecExpr};
use hashbrown::HashMap;

use crate::Counter;
use crate::candidates::exact::count::{
    NodeMatches, NovelTermCount, count_terms_rooted, enumerate_matches_rooted,
    find_novel_root_sizes_rooted, prune_matches, root_budgets,
};
use crate::candidates::exact::draw::{
    BalancedFrontierDrawer, CountWeigher, ExactDrawer, IndependentFrontierDrawer, NaiveWeigher,
    PlainDrawer,
};
use crate::candidates::{ExactSelectionPolicy, SizeAllocation};
use crate::eqsat::EqsatResult;
use crate::{MyAnalysis, MyLanguage, OriginLang};

/// Final e-graph and complete count tables for exact candidate construction.
///
/// Construction consumes [`EqsatResult`] and discards its run metadata.
pub struct ExactCandidatePackage<C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    egraph: EGraph<L, N>,
    tc: NovelTermCount<C>,
    min_size: usize,
    max_size: usize,
    root: Id,
}

impl<C, L, N> ExactCandidatePackage<C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    /// Build counts through `max_size` relative to the previous boundary.
    /// Returns `None` for an empty frontier.
    #[must_use]
    pub fn build(
        result: EqsatResult<L, N>,
        max_size: usize,
    ) -> Option<ExactCandidatePackage<C, L, N>> {
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
        budgets: &crate::candidates::exact::count::RootBudgets,
    ) -> Option<ExactCandidatePackage<C, L, N>> {
        let (egraph, root) = result.into_curr();
        let plain = count_terms_rooted(&egraph, budgets);
        let tc = NovelTermCount::from_rooted_matches(&egraph, plain, matches, budgets);

        let root = egraph.find(root);
        let histogram = tc.data().get(&root)?;

        let min_size = histogram.keys().min().copied().unwrap_or(1);
        Some(ExactCandidatePackage {
            egraph,
            tc,
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
        novel_size_goal: usize,
        log: &mut W,
    ) -> Result<(usize, Self), usize> {
        assert!(novel_size_goal > 0, "novel_size_goal must be nonzero");

        let cap = start_size + max_retries * retry_step;

        let prev = result.prev_index();
        let curr = result.curr();
        let root = curr.find(result.root());
        let cap_budgets = root_budgets(curr, root, cap);
        let mut matches = enumerate_matches_rooted(curr, &prev, &cap_budgets);

        drop(prev);

        let novel_sizes =
            find_novel_root_sizes_rooted(curr, root, &matches, novel_size_goal, &cap_budgets);
        if novel_sizes.len() < novel_size_goal {
            writeln!(
                log,
                "found {found} of {novel_size_goal} novel sizes (max_size={cap})",
                found = novel_sizes.len()
            )
            .unwrap();
            return Err(cap);
        }
        let max_size = novel_sizes[novel_size_goal - 1];
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
        if package.root_histogram().len() < novel_size_goal {
            writeln!(
                log,
                "package construction found fewer than {novel_size_goal} novel sizes \
                 (max_size={max_size})"
            )
            .unwrap();
            return Err(cap);
        }

        Ok((max_size, package))
    }

    /// Log the stats about the root into `out`.
    ///
    /// # Panics
    ///
    /// Panics if there are no terms in the root, or if writing to `out` fails.
    pub fn log_root_counts<W: std::fmt::Write>(&self, out: &mut W) {
        let histogram = self
            .tc
            .data()
            .get(&self.root)
            .expect("Somehow the root does not contain any terms?");
        let mut sorted_hist = histogram
            .iter()
            .map(|(a, b)| (*a, b.to_owned()))
            .collect::<Vec<_>>();
        sorted_hist.sort_unstable_by_key(|(size, _)| *size);
        writeln!(out, "Terms in frontier:").unwrap();
        for (k, v) in &sorted_hist {
            writeln!(out, "{v} terms of size {k}").unwrap();
        }
    }

    /// Draw exact root candidates absent from the previous boundary.
    #[must_use]
    pub fn draw_frontier_candidates(
        &self,
        count: usize,
        allocation: SizeAllocation,
        selection_policy: ExactSelectionPolicy,
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let histogram = self.tc.data().get(&self.root)?;

        let requests = allocation.allocate(histogram, self.min_size, self.max_size, count);

        match selection_policy {
            ExactSelectionPolicy::Naive => {
                IndependentFrontierDrawer::new(&self.tc, &self.egraph, self.root, NaiveWeigher)
                    .draw_root_batch(&requests, seed)
            }
            ExactSelectionPolicy::Independent => {
                IndependentFrontierDrawer::new(&self.tc, &self.egraph, self.root, CountWeigher)
                    .draw_root_batch(&requests, seed)
            }
            ExactSelectionPolicy::Balanced => {
                BalancedFrontierDrawer::new(&self.tc, &self.egraph, self.root)
                    .draw_root_batch(&requests, seed)
            }
        }
    }

    /// Draw frontier candidates with batch-local coverage balancing per size.
    #[must_use]
    pub fn draw_balanced_frontier_candidates(
        &self,
        count: usize,
        allocation: SizeAllocation,
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let histogram = self.tc.data().get(&self.root)?;
        let requests = allocation.allocate(histogram, self.min_size, self.max_size, count);

        BalancedFrontierDrawer::new(&self.tc, &self.egraph, self.root)
            .draw_root_batch(&requests, seed)
    }

    /// [`Self::draw_balanced_frontier_candidates`] with explicit coverage
    /// penalties.
    #[must_use]
    pub fn draw_balanced_frontier_candidates_with_config(
        &self,
        count: usize,
        allocation: SizeAllocation,
        seed: [u64; 2],
        config: crate::candidates::BalanceConfig,
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let histogram = self.tc.data().get(&self.root)?;
        let requests = allocation.allocate(histogram, self.min_size, self.max_size, count);

        BalancedFrontierDrawer::with_config(&self.tc, &self.egraph, self.root, config)
            .draw_root_batch(&requests, seed)
    }

    #[must_use]
    pub fn smallest_candidate(&self, id: Id, novel: bool) -> RecExpr<OriginLang<L>> {
        if novel {
            IndependentFrontierDrawer::new(&self.tc, &self.egraph, self.root, NaiveWeigher)
                .smallest(id)
        } else {
            PlainDrawer::new(self.tc.plain(), &self.egraph, self.root, NaiveWeigher).smallest(id)
        }
    }

    #[must_use]
    pub const fn root(&self) -> Id {
        self.root
    }

    /// Novel root-term counts by size.
    ///
    /// # Panics
    ///
    /// Panics if package construction violated its root-histogram invariant.
    #[must_use]
    pub fn root_histogram(&self) -> &HashMap<usize, C> {
        self.tc
            .data()
            .get(&self.root)
            .expect("root histogram present iff build returned Some")
    }
}

#[cfg(test)]
mod tests {
    use egg::EGraph;
    use num::BigUint;

    use super::*;
    use crate::langs::math::Math;
    use crate::utils::sym;

    #[test]
    fn build_through_novel_sizes_runs_exact_analysis_at_kth_novel_size() {
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
        let (used_max_size, package) =
            ExactCandidatePackage::<BigUint, _, _>::build_through_novel_sizes(
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
    fn balanced_selection_policy_covers_union_profiles() {
        let mut curr = EGraph::<Math, ()>::new(());
        curr.enable_union_event_recording();
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let root = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev_raw_node_count = curr.nodes().len();
        let prev_union_event_count = curr.union_event_count();
        curr.union(a, b);
        curr.rebuild();

        let result =
            EqsatResult::new_for_tests(curr, root, prev_raw_node_count, prev_union_event_count);
        let package =
            ExactCandidatePackage::<BigUint, _, _>::build(result, 3).expect("frontier package");
        let terms = package
            .draw_frontier_candidates(
                3,
                SizeAllocation::Greedy,
                ExactSelectionPolicy::Balanced,
                [5, 8],
            )
            .expect("balanced frontier terms");
        let lowered = terms
            .into_iter()
            .map(|term| crate::lower(term).to_string())
            .collect::<hashbrown::HashSet<_>>();

        assert_eq!(
            lowered,
            hashbrown::HashSet::from([
                "(+ a a)".to_owned(),
                "(+ b a)".to_owned(),
                "(+ b b)".to_owned(),
            ])
        );
    }
}
