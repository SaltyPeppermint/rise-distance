use std::time::Instant;

use egg::{Id, RecExpr};
use hashbrown::HashMap;

use crate::Counter;
use crate::eqsat::EqsatResult;
use crate::sampling::count::{
    NodeMatches, NovelTermCount, PlainTermCount, enumerate_matches, probe_novel_root_sizes,
};
use crate::sampling::sampler::{CountWeigher, NaiveWeigher, NovelSampler, PlainSampler, Sampler};
use crate::sampling::{SampleStrategy, TermSampleDist};
use crate::{MyAnalysis, MyLanguage, OriginLang};

pub struct PrecomputePackage<'a, C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    tc: NovelTermCount<'a, C, L, N>,
    min_size: usize,
    max_size: usize,
    root: Id,
}

impl<'a, C, L, N> PrecomputePackage<'a, C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    /// Enumerate all frontier terms from `egraph` that are NOT present in `prev_raw_egg` for the sampling process later
    #[must_use]
    pub fn precompute(
        result: &'a EqsatResult<L, N>,
        max_size: usize,
    ) -> Option<PrecomputePackage<'a, C, L, N>> {
        let matches = enumerate_matches(result.curr(), result.prev());
        Self::precompute_with_matches(result, max_size, matches)
    }

    /// [`precompute`](Self::precompute) with the (size-independent) match
    /// enumeration precomputed by the caller, so repeated runs on the same
    /// egraph pair don't redo it.
    fn precompute_with_matches(
        result: &'a EqsatResult<L, N>,
        max_size: usize,
        matches: NodeMatches,
    ) -> Option<PrecomputePackage<'a, C, L, N>> {
        let tc = NovelTermCount::with_matches(
            max_size,
            result.curr(),
            result.prev(),
            PlainTermCount::rooted(max_size, result.curr(), &[result.root()]),
            matches,
        );

        let root = result.curr().find(result.root());
        let histogram = tc.data().get(&root)?;

        let min_size = histogram.keys().min().copied().unwrap_or(1);
        Some(PrecomputePackage {
            tc,
            min_size,
            max_size,
            root,
        })
    }

    /// One exact precompute attempt: succeeds iff the root has at least
    /// `sizes` distinct novel term sizes, clamping `max_size` to the
    /// `sizes`-th smallest so we don't sample from too many sizes.
    fn exact_attempt(
        result: &'a EqsatResult<L, N>,
        max_size: usize,
        matches: &NodeMatches,
        sizes: usize,
    ) -> Option<PrecomputePackage<'a, C, L, N>> {
        let mut pp = Self::precompute_with_matches(result, max_size, matches.clone())?;
        if pp.root_histogram().keys().len() < sizes {
            return None;
        }
        pp.max_size = **pp
            .root_histogram()
            .keys()
            .collect::<Vec<_>>()
            .select_nth_unstable(sizes - 1)
            .1;
        Some(pp)
    }

    /// Like [`precompute`](Self::precompute), but searches for the smallest
    /// `max_size` that yields at least `sizes` distinct novel term sizes at
    /// the root while running the expensive exact analysis only once.
    ///
    /// The search runs on cheap fingerprint counts (`u64` arithmetic modulo
    /// the prime `2^61 - 1`) in a single size-incremental pass: the counting
    /// DPs advance one size layer at a time, the root's novelty at each size
    /// is final as soon as its layer completes, and the pass stops at the
    /// `sizes`-th novel size — which is exactly the `max_size` the single
    /// exact run then uses, and the returned `usize`. `start_size`,
    /// `max_retries` and `retry_step` only bound the search: the probe gives
    /// up at `start_size + max_retries * retry_step` (they also still drive
    /// the exact fallback schedule below). See `docs/incremental_probe.md`.
    ///
    /// A nonzero fingerprint proves a nonzero exact count, so the probe can
    /// only err by *missing* a size whose exact count is divisible by
    /// `2^61 - 1` (probability on the order of `2^-61` per size). The exact
    /// run is therefore re-verified, and on a mismatch (or if the probe
    /// finds too few sizes) this falls back to the old exact backoff loop,
    /// so the result is always exact.
    ///
    /// # Errors
    ///
    /// Errors if no `max_size` up to the cap yields enough novel sizes. The
    /// error value is the cap (`start_size + max_retries * retry_step`).
    pub fn backoff_precompute<W: std::fmt::Write>(
        result: &'a EqsatResult<L, N>,
        start_size: usize,
        max_retries: usize,
        retry_step: usize,
        sizes: usize,
        fallback: bool,
        log: &mut W,
    ) -> Result<(usize, PrecomputePackage<'a, C, L, N>), usize> {
        let curr = result.curr();
        let root = curr.find(result.root());
        // The match enumeration is independent of `max_size`; compute it once
        // and share it between the probe and all exact runs.
        let matches = enumerate_matches(curr, result.prev());

        let cap = start_size + max_retries * retry_step;

        let start = Instant::now();
        let novel_sizes = probe_novel_root_sizes(cap, curr, root, &matches, sizes);
        let probed = if novel_sizes.len() >= sizes {
            Some(novel_sizes[sizes - 1])
        } else {
            writeln!(
                log,
                "probe found {found} of {sizes} novel sizes (max_size={cap})",
                found = novel_sizes.len()
            )
            .unwrap();
            None
        };

        if let Some(max_size) = probed {
            if let Some(pp) = Self::exact_attempt(result, max_size, &matches, sizes) {
                return Ok((max_size, pp));
            }
            // Only reachable through a fingerprint collision in the probe.
            writeln!(
                log,
                "exact analysis found fewer novel sizes than the probe (max_size={max_size})"
            )
            .unwrap();
        }

        eprintln!(
            "AFTER {} SECONDS NOTHIGN FOUND WITH CHEAP METHOD",
            start.elapsed().as_secs_f64()
        );

        // Fallback
        if fallback {
            eprintln!("Running expensive fallback");
            (0..=max_retries)
                .map(|i| start_size + i * retry_step)
                .find_map(|size| {
                    if let Some(pp) = Self::exact_attempt(result, size, &matches, sizes) {
                        Some((size, pp))
                    } else {
                        writeln!(
                            log,
                            "goal precompute returned None (max_size={size}), retrying"
                        )
                        .unwrap();
                        None
                    }
                })
                .ok_or(cap)
        } else {
            eprintln!("Fallback disabled");
            Err(cap)
        }
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

    /// Sample frontier goal terms from `egraph` that are NOT present in `prev_raw_egg`.
    #[must_use]
    pub fn sample_frontier_terms(
        &self,
        count: usize,
        distribution: TermSampleDist,
        sample_strategy: SampleStrategy,
        seed: [u64; 2],
        novel: bool,
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let histogram = self.tc.data().get(&self.root)?;

        let samples_per_size =
            distribution.samples_per_size(histogram, self.min_size, self.max_size, count);

        match (sample_strategy, novel) {
            (SampleStrategy::Naive, true) => NovelSampler::new(&self.tc, self.root, NaiveWeigher)
                .sample_batch_root(&samples_per_size, seed),
            (SampleStrategy::Count, true) => NovelSampler::new(&self.tc, self.root, CountWeigher)
                .sample_batch_root(&samples_per_size, seed),
            (SampleStrategy::Naive, false) => {
                PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, NaiveWeigher)
                    .sample_batch_root(&samples_per_size, seed)
            }
            (SampleStrategy::Count, false) => {
                PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, CountWeigher)
                    .sample_batch_root(&samples_per_size, seed)
            }
        }
    }

    #[must_use]
    pub fn smallest(&self, id: Id, novel: bool) -> RecExpr<OriginLang<L>> {
        if novel {
            NovelSampler::new(&self.tc, self.root, NaiveWeigher).smallest(id)
        } else {
            PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, NaiveWeigher).smallest(id)
        }
    }

    #[must_use]
    pub const fn root(&self) -> Id {
        self.root
    }

    /// Histogram of novel root extractions by size. Guaranteed non-empty
    /// because [`Self::precompute`] returns `None` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the root histogram is somehow missing (would indicate a bug
    /// in `precompute`'s None check).
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
    use crate::test_utils::sym;

    #[test]
    fn backoff_precompute_runs_exact_analysis_at_kth_novel_size() {
        // Unioning `a` with the root of (+ a b) creates a cycle: the root
        // class extracts a, (+ a b), (+ (+ a b) b), ... (sizes 1, 3, 5, ...).
        // `a` and (+ a b) already exist in prev, so the novel sizes are
        // 5, 7, 9, ... asking for 3 sizes must yield max_size = 9.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let apb = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, apb);
        curr.rebuild();

        let result = EqsatResult::new_for_tests(prev, curr, apb);
        let mut log = String::new();
        let (used_max_size, pp) = PrecomputePackage::<BigUint, _, _>::backoff_precompute(
            &result, 3, 10, 2, 3, false, &mut log,
        )
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
}
