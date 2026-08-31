//! Sketch-based reachability search over a generic e-graph.
//!
//! Given a single start expression and a set of [`Sketch`] goals, search for a
//! point where the root e-class satisfies every goal. Two strategies are
//! offered via [`SearchMode`]:
//!
//! - [`SearchMode::Cut`]: grow the e-graph to a chosen iteration, construct
//!   novel frontier candidates, then continue eqsat from them and verify.
//! - [`SearchMode::Brute`]: grow one continuous e-graph and check the sketches
//!   directly, with no candidate restart.

use std::fmt::Display;

use egg::{Language, RecExpr, Rewrite};

use crate::Counter;
use crate::candidates::{DrawerPackage, FrontierPackage};
use crate::cli::Policy;
use crate::eqsat::{self, EqsatConfig, EqsatMetadata, Goal};
use crate::sketch::Sketch;
use crate::{MyAnalysis, MyLanguage, OriginLang, id0, lower};

/// Tunable knobs for the cut-and-restart search strategy.
#[derive(Copy, Clone, Debug, clap::Args)]
pub struct CutArgs {
    /// Iteration at which to cut the egraph, construct novel candidates, and
    /// continue eqsat from them.
    #[arg(long, default_value_t = 6)]
    pub cut_iters: usize,

    /// Maximum frontier term size enumerated by [`ExactCandidatePackage`] when
    /// constructing guide candidates at the cut.
    #[arg(long, default_value_t = 30)]
    pub max_size: usize,

    /// Maximum nodes in an egraph
    #[arg(long, default_value_t = 1_000_000)]
    pub max_nodes: usize,

    /// Maximum time in an egraph
    #[arg(long, default_value_t = 30.0)]
    pub max_time: f64,

    /// Number of novel frontier candidates to draw at the cut point as the guide
    /// set to continue eqsat from.
    #[arg(long, default_value_t = 100)]
    pub candidate_count: usize,
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
    /// Cut at an iteration, construct novel candidates, continue, and verify.
    Cut(CutArgs),
    /// Grow one continuous egraph and check the sketches directly.
    Brute(BruteArgs),
}

/// Outcome of a sketch-based reachability search.
pub struct ReachResult<L: Language> {
    /// Whether all sketch goals were satisfied.
    pub reached: Option<RecExpr<L>>,
    /// For [`SearchMode::Cut`], the novel candidates drawn at the cut and
    /// used as guides. Empty for [`SearchMode::Brute`].
    pub candidates: Vec<RecExpr<L>>,

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

/// Cut-and-restart strategy: grow to `cut_iters`, construct novel candidates,
/// then continue eqsat from them and verify the sketches.
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
        print_success_iters: false,
    };

    let Some(result) = eqsat::run_eqsat::<L, N, _>(start, rules.iter(), &eqsat_config) else {
        println!("{search_name}: run_eqsat produced no distinct cut state");
        return ReachResult {
            reached: None,
            candidates: Vec::new(),
            eqsat_meta: Vec::new(),
        };
    };
    println!(
        "{search_name}: stopped with reason {:?}",
        result.stop_reason()
    );

    let cut_meta = EqsatMetadata::from_iterations(result.data());
    let cut_iters = result.iters();

    let Some(package) = FrontierPackage::<C, _, _>::build(result, args.max_size) else {
        println!("{search_name}: exact candidate package found an empty frontier");
        return ReachResult {
            reached: None,
            candidates: Vec::new(),
            eqsat_meta: vec![cut_meta],
        };
    };
    // log_root_counts(package.root_histogram(), &mut log);

    let Some(candidates) = package.draw_candidates(
        args.candidate_count,
        Policy::Count,
        [args.cut_iters as u64, 0],
    ) else {
        println!("{search_name}: candidate drawing failed");
        return ReachResult {
            reached: None,
            candidates: Vec::new(),
            eqsat_meta: vec![cut_meta],
        };
    };

    println!(
        "Drew {} candidates after {} iterations!",
        candidates.len(),
        cut_iters
    );
    for candidate in &candidates {
        println!("{}", lower(candidate.to_owned()));
    }

    let verify = eqsat::guided_eqsat(
        &candidates,
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
        candidates: candidates.into_iter().map(lower).collect(),
        eqsat_meta,
    }
}

/// Brute-force (no-cut) strategy: grow one continuous egraph from `start` and
/// check the sketches directly, with no candidate restart.
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
        print_success_iters: false,
    };

    // Lift the plain start expr into an OriginLang guide (inverse of `lower`);
    // origin is irrelevant here since there is no full-union dedup.
    let guide = start
        .as_ref()
        .iter()
        .map(|n| OriginLang::new(n.clone(), id0()))
        .collect();

    let verify = eqsat::guided_eqsat(
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
        candidates: Vec::new(),
        eqsat_meta,
    }
}
