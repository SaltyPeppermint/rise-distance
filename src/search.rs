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

use egg::{Language, RecExpr, Rewrite};

use crate::Counter;
use crate::eqsat::{self, EqsatConfig, EqsatMetadata, Goal};
use crate::sampling::{PrecomputePackage, SampleStrategy, TermSampleDist};
use crate::sketch::Sketch;
use crate::{MyAnalysis, MyLanguage, OriginLang, id0, lower};

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
pub struct ReachResult<L: Language> {
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
    L: MyLanguage,
    N: MyAnalysis<L>,
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
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    println!("Doing search '{search_name}' via cut\n");
    let eqsat_config = EqsatConfig {
        max_iters: args.cut_iters,
        max_nodes: args.max_nodes,
        max_time: args.max_time,
        max_memory: None,
        backoff_scheduler: false,
    };

    let Some(result) = eqsat::run_eqsat::<L, N, _>(start, rules.iter(), &eqsat_config) else {
        println!("{search_name}: run_eqsat produced no distinct cut state");
        return ReachResult {
            reached: None,
            sampled: Vec::new(),
            eqsat_meta: Vec::new(),
        };
    };
    println!(
        "{search_name}: stopped with reason {:?}",
        result.stop_reason()
    );

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

    let Some(sampled) = pp.sample_frontier_terms(
        args.sample_count,
        TermSampleDist::GREEDY,
        SampleStrategy::Independent,
        [args.cut_iters as u64, 0],
    ) else {
        println!("{search_name}: sampling failed");
        return ReachResult {
            reached: None,
            sampled: Vec::new(),
            eqsat_meta: vec![cut_meta],
        };
    };

    println!(
        "Sampled {} terms after {} iterations!",
        sampled.len(),
        result.iters()
    );
    for s in &sampled {
        println!("{}", lower(s.to_owned()));
    }

    let verify = eqsat::verify_reachability(
        &sampled,
        &Goal::Sketches(sketch_goals),
        rules,
        &eqsat_config,
        false,
    );

    let (reached, verify_iters) = match verify {
        Ok(run) => (Some(run.target), Some(run.iterations)),
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
        max_memory: None,
        backoff_scheduler: false,
    };

    // Lift the plain start expr into an OriginLang guide (inverse of `lower`);
    // origin is irrelevant here since there is no full-union dedup.
    let guide = start
        .as_ref()
        .iter()
        .map(|n| OriginLang::new(n.clone(), id0()))
        .collect();

    let verify = eqsat::verify_reachability(
        std::slice::from_ref(&guide),
        &Goal::Sketches(sketch_goals),
        rules,
        &config,
        false,
    );

    let (reached, verify_iters) = match verify {
        Ok(run) => (Some(run.target), Some(run.iterations)),
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
