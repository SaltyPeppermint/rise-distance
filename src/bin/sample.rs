//! Produce the guide-samples menu for one start term.
//!
//! Replays guide-phase eqsat and draws the requested samples pool.
//! Arguments come from `guided_search.py`; output is a one-element JSON array
//! on stdout, or an empty array on failure. Logs go to stderr.

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use serde::Serialize;
use time::OffsetDateTime;

use rise_distance::cli::{Measured, Policy};
use rise_distance::eqsat::{self, EqsatConfig, EqsatResult};
use rise_distance::langs::{AvailableLanguages, MyAnalysis, MyLanguage, diospyros, math, prop};
use rise_distance::origin::{self, OriginLang};
use rise_distance::sampling::{AnalysisPackage, FrontierPackage, WholePackage};
use rise_distance::utils;

#[derive(Parser)]
#[command(
    about = "Construct the guide-samples menu for one start term (feeds guided_search.py)",
    after_help = "\
Prints a one-element `[Samples]` array to stdout (empty
on failure); logs go to stderr.
Example:
  sample --language math --start-term '(+ x 0)' \\
    --max-iters 38 --max-nodes 1000000 --max-time 10 \\
    --max-memory 2000000000 \\
    --policy count
"
)]
struct Args {
    /// Language used for eqsat.
    #[arg(long)]
    language: AvailableLanguages,

    /// Start-term s-expression whose guide phase gets replayed.
    #[arg(long)]
    start_term: String,

    /// Guide replay limits.
    #[command(flatten)]
    eqsat: EqsatConfig,

    /// Number of samples to draw.
    #[arg(long, default_value_t = 1000)]
    n_samples: usize,

    /// samples-construction seed, independent of the batch size.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// How much to grow `max_size` on each exact-size-search retry.
    #[arg(long, default_value_t = 2)]
    size_search_step: usize,

    /// Number of exact-size-search increments.
    #[arg(long, default_value_t = 200)]
    size_search_steps: usize,

    /// Policy used to draw samples.
    #[arg(long, value_enum)]
    policy: Policy,

    /// Take from the frontier
    #[arg(long, default_value_t = false)]
    frontier: bool,
}

fn main() {
    let args = Args::parse();

    eprintln!("Starting at {}", OffsetDateTime::now_local().unwrap());
    eprintln!("Language: {:?}", args.language);
    eprintln!("Start Term: {}", args.start_term);

    match args.language {
        AvailableLanguages::Diospyros => {
            main_inner(&args, &diospyros::rules(false, false));
        }
        AvailableLanguages::Math => {
            main_inner(&args, &math::rules());
        }
        AvailableLanguages::Prop => {
            main_inner(&args, &prop::rules());
        }
    }

    eprintln!("Finished at {}", OffsetDateTime::now_local().unwrap());
}

/// Build one language-specific seed's sample record and print it as JSON.
fn main_inner<L: MyLanguage, N: MyAnalysis<L>>(args: &Args, rules: &[Rewrite<L, N>]) {
    eprintln!(
        "\n=== Start Term: {} (max-iters={}) ===",
        args.start_term, args.eqsat.max_iters
    );

    let out = match build_sample_record(args, rules) {
        Ok(record) => vec![record],
        Err(e) => {
            eprintln!("ERROR OCCURRED:\n{e}");
            vec![]
        }
    };

    eprintln!(
        "Finished start term at {}",
        OffsetDateTime::now_local().unwrap()
    );

    serde_json::to_writer(std::io::stdout(), &Measured::now(out)).expect("write samples JSON");
    println!();
}

fn build_sample_record<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    rules: &[Rewrite<L, N>],
) -> Result<Samples<L>, String> {
    let seed_expr = args
        .start_term
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse start term '{}': {e}", args.start_term));

    // Replay the guide phase under the effective limits the driver computed;
    // the replay ends at whichever limit trips first.
    let result = eqsat::run_eqsat(&seed_expr, rules.iter(), &args.eqsat).ok_or("Eqsat failed")?;

    eprintln!("DEBUG: PEAK RSS AFTER EQSAT: {}", utils::peak_rss_bytes());
    let stop_reason = format!("{:?}", result.stop_reason());
    eprintln!("Guide replay stop reason: {stop_reason}");

    // Absolute process live heap before sample construction allocates more.
    let guide_memory = result.allocated();
    let guide_peak_live_heap = result.peak_allocated();
    let guide_nodes = result.curr().total_number_of_nodes();
    let guide_classes = result.curr().classes().len();
    let guide_iters = result.data().len();
    let guide_time = result.data().iter().map(|i| i.total_time).sum();
    eprintln!(
        "Guide egraph (replay): {guide_nodes} nodes, {guide_classes} classes, \
         {guide_memory} live-heap bytes"
    );

    let samples = if args.frontier {
        build_frontier_samples(args, result, args.policy, &seed_expr)?
    } else {
        build_whole_samples(args, result, args.policy, &seed_expr)?
    };
    eprintln!(
        "DEBUG: PEAK RSS AFTER SAMPLING: {}",
        utils::peak_rss_bytes()
    );
    Ok(Samples {
        start_term: args.start_term.clone(),
        policy: args.policy.to_string(),
        samples: samples.clone().into_iter().map(|e| e.to_vec()).collect(),
        samples_s_expr: samples.into_iter().map(origin::lower).collect(),
        guide_nodes,
        guide_classes,
        guide_iters,
        guide_time,
        guide_memory,
        guide_peak_live_heap,
        stop_reason,
    })
}

fn build_frontier_samples<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    result: EqsatResult<L, N>,
    policy: Policy,
    seed_expr: &RecExpr<L>,
) -> Result<Vec<RecExpr<OriginLang<L>>>, String> {
    let start_size = AstSize.cost_rec(seed_expr);
    let (max_size, package) = FrontierPackage::build_through_novel_sizes(
        result,
        start_size,
        args.size_search_steps,
        args.n_samples * 10, // More than 10x the terms should be present so we can easily sample
    )
    .map_err(|tried_max_size| {
        format!(
            "samples construction found too few novel sizes after {} retries \
                 (max_size={})",
            args.size_search_steps, tried_max_size
        )
    })?;
    eprintln!(
        "DEBUG: PEAK RSS AFTER ANALYSIS: {}",
        utils::peak_rss_bytes()
    );
    eprintln!("Sampling package succeeded with max_size {max_size}!");
    package.log_root_counts();
    let samples = package
        .draw_samples(args.n_samples, policy, [args.seed, 0])
        .unwrap_or_else(|e| {
            eprintln!(
                "WARNING: policy {policy} drew 0 n_samples ({e}) \
                     driver legs for this policy will have no guides to pick from"
            );
            Vec::new()
        });
    Ok(samples)
}

fn build_whole_samples<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    result: EqsatResult<L, N>,
    policy: Policy,
    seed_expr: &RecExpr<L>,
) -> Result<Vec<RecExpr<OriginLang<L>>>, String> {
    let start_size = AstSize.cost_rec(seed_expr);
    let (max_size, package) = WholePackage::build_through_sizes(
        result,
        start_size,
        args.size_search_steps,
        args.n_samples * 10, // More than 10x the terms should be present so we can easily sample
    )
    .map_err(|tried_max_size| {
        format!(
            "samples construction found too few terms after {} retries \
                 (max_size={})",
            args.size_search_steps, tried_max_size
        )
    })?;
    eprintln!(
        "DEBUG: PEAK RSS AFTER ANALYSIS: {}",
        utils::peak_rss_bytes()
    );
    eprintln!("Sampling package succeeded with max_size {max_size}!");
    package.log_root_counts();
    let samples = package
        .draw_samples(args.n_samples, policy, [args.seed, 0])
        .unwrap_or_else(|e| {
            eprintln!(
                "WARNING: policy {policy} drew 0 samples ({e}); \
                     driver legs for this policy will have no guides to pick from"
            );
            Vec::new()
        });
    Ok(samples)
}

/// For Serialization purposes we have to go via Vec instead of using `RecExpr`
#[derive(Serialize, Debug, Clone)]
#[expect(clippy::struct_field_names)]
struct Samples<L: MyLanguage> {
    start_term: String,
    policy: String,

    samples: Vec<Vec<OriginLang<L>>>,
    samples_s_expr: Vec<RecExpr<L>>,
    guide_nodes: usize,
    guide_classes: usize,
    guide_iters: usize,
    /// Total wall-clock time (seconds) of the guide-phase replay, so the driver
    /// can add the guide overhead to each leg's `total_time`.
    guide_time: f64,
    /// Guide-phase replay's absolute live allocation (bytes): jemalloc
    /// `stats.allocated` for the whole process, the same coordinate system the
    /// configured memory ceiling is expressed in. Includes heap the process
    /// already held before this run started.
    guide_memory: u64,
    /// Largest observed absolute live heap during guide replay.
    guide_peak_live_heap: u64,
    stop_reason: String,
}
