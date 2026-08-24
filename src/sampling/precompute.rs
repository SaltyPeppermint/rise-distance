use egg::{EGraph, Id, RecExpr};
use hashbrown::HashMap;

use crate::Counter;
use crate::eqsat::EqsatResult;
use crate::sampling::count::{
    NodeMatches, NovelTermCount, count_terms_rooted, enumerate_matches_rooted,
    find_novel_root_sizes_rooted, prune_matches, root_budgets,
};
use crate::sampling::sampler::{
    BalancedFrontierSampler, CountWeigher, IndependentFrontierSampler, NaiveWeigher, PlainSampler,
    Sampler,
};
use crate::sampling::{Distribution, SampleStrategy};
use crate::{MyAnalysis, MyLanguage, OriginLang};

/// Final e-graph and counting tables for frontier sampling.
///
/// Construction consumes [`EqsatResult`] and discards its run metadata.
pub struct PrecomputePackage<C, L, N>
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

impl<C, L, N> PrecomputePackage<C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    /// Build counts through `max_size` relative to the previous boundary.
    /// Returns `None` for an empty frontier.
    #[must_use]
    pub fn precompute(
        result: EqsatResult<L, N>,
        max_size: usize,
    ) -> Option<PrecomputePackage<C, L, N>> {
        let curr = result.curr();
        let root = curr.find(result.root());
        let budgets = root_budgets(curr, root, max_size);

        let prev = result.prev_index();
        let mut matches = enumerate_matches_rooted(curr, &prev, &budgets);
        drop(prev);
        prune_matches(curr, &mut matches, &budgets);
        Self::build_rooted_package(result, max_size, matches, &budgets)
    }

    /// Finish [`precompute`](Self::precompute) from matches already restricted
    /// to the supplied final root budgets.
    fn build_rooted_package(
        result: EqsatResult<L, N>,
        max_size: usize,
        matches: NodeMatches,
        budgets: &crate::sampling::count::RootBudgets,
    ) -> Option<PrecomputePackage<C, L, N>> {
        let (egraph, root) = result.into_curr();
        let plain = count_terms_rooted(&egraph, budgets);
        let tc = NovelTermCount::from_rooted_matches(&egraph, plain, matches, budgets);

        let root = egraph.find(root);
        let histogram = tc.data().get(&root)?;

        let min_size = histogram.keys().min().copied().unwrap_or(1);
        Some(PrecomputePackage {
            egraph,
            tc,
            min_size,
            max_size,
            root,
        })
    }

    /// Build a package ending at the `sizes`-th novel root size.
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
    /// Panics if `sizes` is zero or writing to `log` fails.
    pub fn backoff_precompute<W: std::fmt::Write>(
        result: EqsatResult<L, N>,
        start_size: usize,
        max_retries: usize,
        retry_step: usize,
        sizes: usize,
        log: &mut W,
    ) -> Result<(usize, Self), usize> {
        assert!(sizes > 0, "sizes must be nonzero");

        let cap = start_size + max_retries * retry_step;

        let prev = result.prev_index();
        let curr = result.curr();
        let root = curr.find(result.root());
        let cap_budgets = root_budgets(curr, root, cap);
        let mut matches = enumerate_matches_rooted(curr, &prev, &cap_budgets);

        drop(prev);

        let novel_sizes = find_novel_root_sizes_rooted(curr, root, &matches, sizes, &cap_budgets);
        if novel_sizes.len() < sizes {
            writeln!(
                log,
                "found {found} of {sizes} novel sizes (max_size={cap})",
                found = novel_sizes.len()
            )
            .unwrap();
            return Err(cap);
        }
        let max_size = novel_sizes[sizes - 1];
        let final_budgets = root_budgets(curr, root, max_size);
        prune_matches(curr, &mut matches, &final_budgets);

        let Some(pp) = Self::build_rooted_package(result, max_size, matches, &final_budgets) else {
            writeln!(
                log,
                "package construction found no novel terms (max_size={max_size})"
            )
            .unwrap();
            return Err(cap);
        };
        if pp.root_histogram().len() < sizes {
            writeln!(
                log,
                "package construction found fewer than {sizes} novel sizes (max_size={max_size})"
            )
            .unwrap();
            return Err(cap);
        }

        Ok((max_size, pp))
    }

    /// Log the stats about the root into `out`.
    ///
    /// # Panics
    ///
    /// Panics if there are no terms in the root, or if writing to `out` fails.
    pub fn log_root<W: std::fmt::Write>(&self, out: &mut W) {
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

    /// Sample root terms absent from the previous boundary.
    #[must_use]
    pub fn sample_frontier_terms(
        &self,
        count: usize,
        distribution: Distribution,
        sample_strategy: SampleStrategy,
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let histogram = self.tc.data().get(&self.root)?;

        let samples_per_size =
            distribution.samples_per_size(histogram, self.min_size, self.max_size, count);

        match sample_strategy {
            SampleStrategy::Naive => {
                IndependentFrontierSampler::new(&self.tc, &self.egraph, self.root, NaiveWeigher)
                    .sample_batch_root(&samples_per_size, seed)
            }
            SampleStrategy::Independent => {
                IndependentFrontierSampler::new(&self.tc, &self.egraph, self.root, CountWeigher)
                    .sample_batch_root(&samples_per_size, seed)
            }
            SampleStrategy::Balanced => {
                BalancedFrontierSampler::new(&self.tc, &self.egraph, self.root)
                    .sample_batch_root(&samples_per_size, seed)
            }
        }
    }

    /// Sample frontier terms with batch-local coverage balancing per size.
    #[must_use]
    pub fn sample_balanced_frontier_terms(
        &self,
        count: usize,
        distribution: Distribution,
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let histogram = self.tc.data().get(&self.root)?;
        let samples_per_size =
            distribution.samples_per_size(histogram, self.min_size, self.max_size, count);

        BalancedFrontierSampler::new(&self.tc, &self.egraph, self.root)
            .sample_batch_root(&samples_per_size, seed)
    }

    /// [`Self::sample_balanced_frontier_terms`] with explicit coverage
    /// penalties.
    #[must_use]
    pub fn sample_balanced_frontier_terms_with_config(
        &self,
        count: usize,
        distribution: Distribution,
        seed: [u64; 2],
        config: crate::sampling::BalanceConfig,
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let histogram = self.tc.data().get(&self.root)?;
        let samples_per_size =
            distribution.samples_per_size(histogram, self.min_size, self.max_size, count);

        BalancedFrontierSampler::with_config(&self.tc, &self.egraph, self.root, config)
            .sample_batch_root(&samples_per_size, seed)
    }

    #[must_use]
    pub fn smallest(&self, id: Id, novel: bool) -> RecExpr<OriginLang<L>> {
        if novel {
            IndependentFrontierSampler::new(&self.tc, &self.egraph, self.root, NaiveWeigher)
                .smallest(id)
        } else {
            PlainSampler::new(self.tc.plain(), &self.egraph, self.root, NaiveWeigher).smallest(id)
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
            .expect("root histogram present iff precompute returned Some")
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
    fn backoff_precompute_runs_exact_analysis_at_kth_novel_size() {
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
        let (used_max_size, pp) =
            PrecomputePackage::<BigUint, _, _>::backoff_precompute(result, 3, 10, 2, 3, &mut log)
                .expect("backoff_precompute should succeed");

        assert_eq!(used_max_size, 9, "log:\n{log}");
        assert_eq!(pp.max_size, 9);
        assert_eq!(pp.min_size, 5);
        let mut keys = pp.root_histogram().keys().copied().collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, vec![5, 7, 9]);
        assert!(
            pp.root_histogram()
                .values()
                .all(|c| *c == BigUint::from(1u32))
        );
    }

    #[test]
    fn balanced_sample_strategy_covers_union_profiles() {
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
            PrecomputePackage::<BigUint, _, _>::precompute(result, 3).expect("frontier package");
        let terms = package
            .sample_frontier_terms(3, Distribution::Greedy, SampleStrategy::Balanced, [5, 8])
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
