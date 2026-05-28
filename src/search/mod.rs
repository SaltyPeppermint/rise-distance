//! Sketch-based reachability search over a generic e-graph.
//!
//! Given a single start expression and a set of [`Sketch`] goals, search for a
//! point where the root e-class satisfies every goal. Two strategies are
//! offered via [`SearchMode`]:
//!
//! - [`SearchMode::Cut`]: grow the e-graph to a chosen iteration, sample the
//!   novel frontier there, then continue eqsat from those samples and verify.
//! - [`SearchMode::Brute`]: grow one continuous e-graph and check the sketches
//!   directly, no sampling or restart.

use std::fmt::Display;

use egg::{Id, Language, RecExpr, Rewrite};
use hashbrown::HashMap;

use crate::cli::ExperimentError;
use crate::cli::argparse::{EqsatConfig, SampleStrategy, TermSampleDist};
use crate::count::Counter;
use crate::langs::{
    EqsatMetadata, EqsatResult, Goal, MyAnalysis, MyLanguage, id0, run_eqsat, verify_reachability,
};
use crate::sampling::{CountWeigher, NaiveWeigher};
use crate::sketch::Sketch;
use crate::{
    NovelSampler, NovelTermCount, OriginLang, PlainSampler, PlainTermCount, Sampler, lower,
};

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
    N: MyAnalysis<L> + Clone,
    C: Counter,
{
    /// Enumerate all frontier terms from `egraph` that are NOT present in `prev_raw_egg` for the sampling process later
    #[must_use]
    pub fn precompute(
        result: &'a EqsatResult<L, N>,
        max_size: usize,
    ) -> Option<PrecomputePackage<'a, C, L, N>> {
        let tc = NovelTermCount::new(
            max_size,
            result.curr(),
            result.prev(),
            PlainTermCount::new(max_size, result.curr()),
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

    /// Log the stats about the root
    ///
    /// # Panics
    ///
    /// Panics if there are no terms in the root
    pub fn log_root(&self) {
        let histogram = self
            .tc
            .data()
            .get(&self.root)
            .expect("Somehow the root does not contain any terms?");
        let mut sorted_hist = histogram
            .iter()
            .map(|(a, b)| (*a, b.to_owned()))
            .collect::<Vec<_>>();
        sorted_hist.sort_unstable();
        println!("Terms in frontier:");
        for (k, v) in &sorted_hist {
            println!("{v} terms of size {k}");
        }
    }

    /// Sample frontier goal terms from `egraph` that are NOT present in `prev_raw_egg`.
    #[expect(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn sample_frontier_terms(
        &self,
        count: usize,
        distribution: TermSampleDist,
        sample_strategy: SampleStrategy,
        seed: [u64; 2],
        novel: bool,
    ) -> Result<Vec<RecExpr<OriginLang<L>>>, ExperimentError> {
        let histogram = self
            .tc
            .data()
            .get(&self.root)
            .ok_or(ExperimentError::InsufficientSamples)?;

        let samples_per_size =
            distribution.samples_per_size(histogram, self.min_size, self.max_size, count);
        let samples = match (sample_strategy, novel) {
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
        };
        let samples = samples?;
        assert!(samples.len() == count, "insufficient samples");
        Ok(samples)
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
    pub fn root(&self) -> Id {
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

/// Tunable knobs for the cut-and-sample search strategy.
#[derive(Copy, Clone, Debug, clap::Args)]
pub struct CutArgs {
    /// Iteration at which to cut the egraph, sample the novel frontier, and
    /// continue eqsat from those samples.
    #[arg(long, default_value_t = 6)]
    pub cut_iters: usize,

    /// Maximum frontier term size enumerated by [`PrecomputePackage`] when
    /// sampling guide terms at the cut.
    #[arg(long, default_value_t = 30)]
    pub max_size: usize,

    /// Maximum nodes in an egraph
    #[arg(long, default_value_t = 1_000_000)]
    pub max_nodes: usize,

    /// Maximum time in an egraph
    #[arg(long, default_value_t = 30.0)]
    pub max_time: f64,

    /// Number of novel frontier terms to sample at the cut point as the guide
    /// set to continue eqsat from.
    #[arg(long, default_value_t = 100)]
    pub sample_count: usize,
}

/// Knobs for the brute-force (no-cut) strategy.
#[derive(Copy, Clone, Debug, clap::Args)]
pub struct BruteArgs {
    /// Maximum eqsat iterations before giving up.
    #[arg(long, default_value_t = 100)]
    pub max_iters: usize,

    /// Maximum nodes in an egraph
    #[arg(long, default_value_t = 1_000_000)]
    pub max_nodes: usize,

    /// Maximum time in an egraph
    #[arg(long, default_value_t = 30.0)]
    pub max_time: f64,
}

/// Which search strategy to run.
#[derive(Copy, Clone, Debug)]
pub enum SearchMode {
    /// Cut at an iteration, sample the novel frontier, continue + verify.
    Cut(CutArgs),
    /// Grow one continuous egraph and check the sketches directly.
    Brute(BruteArgs),
}

/// Outcome of a sketch-based reachability search.
pub struct ReachResult<L: MyLanguage> {
    /// Whether all sketch goals were satisfied.
    pub reached: Option<RecExpr<L>>,
    /// For [`SearchMode::Cut`], the novel frontier terms sampled at the cut and
    /// used as guides. Empty for [`SearchMode::Brute`].
    pub sampled: Vec<RecExpr<L>>,

    /// Per-phase eqsat metadata. [`SearchMode::Cut`] yields up to two entries
    /// (the cut growth, then the verify run); [`SearchMode::Brute`] yields one
    /// (the verify run). Empty when the search bailed before running eqsat.
    pub eqsat_meta: Vec<EqsatMetadata>,
}

/// Search whether `start` can reach an e-graph state satisfying every sketch in
/// `sketch_goals`, using the strategy selected by `mode`.
///
/// `C` is the [`Counter`] used to enumerate the frontier histogram in the cut
/// strategy (e.g. `num::BigUint`); it is unused by the brute strategy.
#[must_use]
pub fn reach_sketches<L, N, C>(
    search_name: &str,
    start: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    sketch_goals: Sketch<L>,
    mode: SearchMode,
) -> ReachResult<L>
where
    L: MyLanguage + Language + Display + 'static,
    N: MyAnalysis<L> + Default + Clone + 'static,
    N::Data: Clone,
    C: Counter,
{
    println!("Start:         {start}");
    println!("(Sketch) Goal: {sketch_goals}\n");
    match mode {
        SearchMode::Cut(args) => {
            reach_cut::<L, N, C>(search_name, start, rules, sketch_goals, args)
        }
        SearchMode::Brute(args) => reach_brute(search_name, start, rules, sketch_goals, args),
    }
}

/// Cut-and-sample strategy: grow to `cut_iters`, sample the novel frontier,
/// then continue eqsat from those samples and verify the sketches.
fn reach_cut<L, N, C>(
    search_name: &str,
    start: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    sketch_goals: Sketch<L>,
    args: CutArgs,
) -> ReachResult<L>
where
    L: MyLanguage + Language + Display + 'static,
    N: MyAnalysis<L> + Default + Clone + 'static,
    N::Data: Clone,
    C: Counter,
{
    println!("Doing search '{search_name}' via cut\n");
    let eqsat_config = EqsatConfig {
        max_iters: args.cut_iters,
        max_nodes: args.max_nodes,
        max_time: args.max_time,
        backoff_scheduler: false,
    };

    let Some(result) = run_eqsat::<L, N, _>(start, rules.iter(), &eqsat_config) else {
        println!("{search_name}: run_eqsat produced no distinct cut state");
        return ReachResult {
            reached: None,
            sampled: Vec::new(),
            eqsat_meta: Vec::new(),
        };
    };

    let cut_meta = EqsatMetadata::from_iterations(result.data());

    let Some(pp) = PrecomputePackage::<C, _, _>::precompute(&result, args.max_size) else {
        println!("{search_name}: precompute returned None (empty frontier)");
        return ReachResult {
            reached: None,
            sampled: Vec::new(),
            eqsat_meta: vec![cut_meta],
        };
    };
    // pp.log_root();

    let sampled = match pp.sample_frontier_terms(
        args.sample_count,
        TermSampleDist::UNIFORM,
        SampleStrategy::Count,
        [args.cut_iters as u64, 0],
        true,
    ) {
        Ok(s) => s,
        Err(e) => {
            println!("{search_name}: sampling failed ({e})");
            return ReachResult {
                reached: None,
                sampled: Vec::new(),
                eqsat_meta: vec![cut_meta],
            };
        }
    };

    println!(
        "Sampled {} terms after {} iterations!",
        sampled.len(),
        result.iters()
    );
    for s in &sampled {
        println!("{}", lower(s.to_owned()));
    }

    let verify = verify_reachability(
        &sampled,
        &Goal::Sketches(sketch_goals),
        rules,
        &eqsat_config,
        false,
    );

    let (reached, verify_iters) = match verify {
        Ok((iters, expr)) => (Some(expr), Some(iters)),
        Err(_) => (None, None),
    };

    let mut eqsat_meta = vec![cut_meta];
    if let Some(iters) = &verify_iters {
        eqsat_meta.push(EqsatMetadata::from_iterations(iters));
    }

    ReachResult {
        reached,
        sampled: sampled.into_iter().map(lower).collect(),
        eqsat_meta,
    }
}

/// Brute-force (no-cut) strategy: grow one continuous egraph from `start` and
/// check the sketches directly, no sampling or restart.
fn reach_brute<L, N>(
    search_name: &str,
    start: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    sketch_goals: Sketch<L>,
    args: BruteArgs,
) -> ReachResult<L>
where
    L: MyLanguage + Language + Display + 'static,
    N: MyAnalysis<L> + Default,
{
    println!("Doing search '{search_name}' via brute_force");
    let config = EqsatConfig {
        max_iters: args.max_iters,
        max_nodes: args.max_nodes,
        max_time: args.max_time,
        backoff_scheduler: false,
    };

    // Lift the plain start expr into an OriginLang guide (inverse of `lower`);
    // origin is irrelevant here since there is no full-union dedup.
    let guide = start
        .as_ref()
        .iter()
        .map(|n| OriginLang::new(n.clone(), id0()))
        .collect();

    let verify = verify_reachability(
        std::slice::from_ref(&guide),
        &Goal::Sketches(sketch_goals),
        rules,
        &config,
        false,
    );

    let (reached, verify_iters) = match verify {
        Ok((iters, expr)) => (Some(expr), Some(iters)),
        Err(_) => (None, None),
    };

    let eqsat_meta = verify_iters
        .iter()
        .map(|iters| EqsatMetadata::from_iterations(iters))
        .collect();

    ReachResult {
        reached,
        sampled: Vec::new(),
        eqsat_meta,
    }
}
