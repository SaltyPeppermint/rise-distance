//! Produce the guide-candidate menu for one seed.
//!
//! Replays guide-phase eqsat and constructs the requested candidate pools.
//! Arguments come from `guided_search.py`; output is a one-element JSON array
//! on stdout, or an empty array on failure. Logs go to stderr.

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use num::BigUint;
use time::OffsetDateTime;

use rise_distance::candidates::{DrawerPackage, FrontierPackage};
use rise_distance::cli::{Candidates, Policy};
use rise_distance::eqsat::{EqsatConfig, EqsatResult, run_eqsat};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::{MyAnalysis, MyLanguage, OriginLang, lower};

#[derive(Parser)]
#[command(
    about = "Construct the guide-candidate menu for one seed (feeds guided_search.py)",
    after_help = "\
Reads nothing on stdin: `--seed`, `--language`, and the replay's eqsat limits
come from argv. Prints a one-element `[SeedCandidates]` array to stdout (empty on
failure); logs go to stderr. `guided_search.py` fans out one invocation per seed,
passing the effective replay limits (search-phase limits overridden by its
`--stop-*` budget flags); the replay ends at whichever limit trips first.
Example:
  candidates --language math --seed '(+ x 0)' \\
    --max-iters 38 --max-nodes 1000000 --max-time 10 \\
    --max-memory 2000000000 \\
    --candidate-pool count
"
)]
struct Args {
    /// Language used for eqsat.
    #[arg(long)]
    language: AvailableLanguages,

    /// Seed s-expression.
    #[arg(long)]
    start_term: String,

    /// Guide replay limits.
    #[command(flatten)]
    eqsat: EqsatConfig,

    /// Candidates per construction pool. `Smallest` contributes one.
    #[arg(long, default_value_t = 1000)]
    n_candidates: usize,

    /// Candidate-construction seed, independent of the batch size.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// How much to grow `max_size` on each exact-size-search retry.
    #[arg(long, default_value_t = 5)]
    retry_step: usize,

    /// Number of exact-size-search increments.
    #[arg(long, default_value_t = 20)]
    max_retries: usize,

    /// Number of novel sizes the exact analysis must find.
    #[arg(long, default_value_t = 5)]
    size_goal: usize,

    /// Candidate policy to emit.
    #[arg(long, value_enum)]
    policy: Policy,
}

fn main() {
    let args = Args::parse();

    eprintln!("Starting at {}", OffsetDateTime::now_local().unwrap());
    eprintln!("Language: {:?}", args.language);
    eprintln!("Start Term: {}", args.start_term);

    match args.language {
        AvailableLanguages::Diospyros => {
            main_inner::<_, ()>(&args, &diospyros::rules(false, false));
        }
        AvailableLanguages::Math => {
            main_inner::<_, math::ConstantFold>(&args, &math::rules());
        }
        AvailableLanguages::Prop => {
            main_inner::<_, prop::ConstantFold>(&args, &prop::rules());
        }
    }

    eprintln!("Finished at {}", OffsetDateTime::now_local().unwrap());
}

/// Build one language-specific seed's candidate record and print it as JSON.
fn main_inner<L: MyLanguage, N: MyAnalysis<L>>(args: &Args, rules: &[Rewrite<L, N>]) {
    eprintln!(
        "\n=== Start Term: {} (max-iters={}) ===",
        args.start_term, args.eqsat.max_iters
    );

    let out = match build_candidate_record(args, rules) {
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

    serde_json::to_writer(std::io::stdout(), &out).expect("write candidates JSON");
    println!();
}

fn build_candidate_record<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    rules: &[Rewrite<L, N>],
) -> Result<Candidates<L>, String> {
    assert!(args.size_goal > 0, "--size-goal must be positive");

    let seed_expr = args
        .start_term
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse start term '{}': {e}", args.start_term));

    // Replay the guide phase under the effective limits the driver computed;
    // the replay ends at whichever limit trips first.
    let result = run_eqsat(&seed_expr, rules.iter(), &args.eqsat).ok_or("Eqsat failed")?;
    let stop_reason = format!("{:?}", result.stop_reason());
    eprintln!("Guide replay stop reason: {stop_reason}");

    // Absolute process live heap before candidate construction allocates more.
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

    let start_size = AstSize.cost_rec(&seed_expr);

    let candidates = build_full_analysis_candidates(args, result, args.policy, start_size)?;

    Ok(Candidates {
        start_term: args.start_term.clone(),
        policy: args.policy.to_string(),
        candidates: candidates.clone().into_iter().map(|e| e.to_vec()).collect(),
        candidate_s_expr: candidates.into_iter().map(lower).collect(),
        guide_nodes,
        guide_classes,
        guide_iters,
        guide_time,
        guide_memory,
        guide_peak_live_heap,
        stop_reason,
    })
}

fn build_full_analysis_candidates<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    result: EqsatResult<L, N>,
    policy: Policy,
    start_size: usize,
) -> Result<Vec<RecExpr<OriginLang<L>>>, String> {
    let mut root_log = String::new();
    let (max_size, package) = FrontierPackage::<BigUint, _, _>::build_through_novel_sizes(
        result,
        start_size,
        args.max_retries,
        args.retry_step,
        args.size_goal,
        &mut root_log,
    )
    .map_err(|tried_max_size| {
        format!(
            "exact candidate construction found too few novel sizes after {} retries \
                 (max_size={})",
            args.max_retries, tried_max_size
        )
    })?;
    eprintln!("Exact candidate package succeeded with max_size {max_size}!");
    package.log_root_counts(&mut root_log);
    eprint!("{root_log}");
    Ok(draw_candiates(args, policy, &package))
}

/// Draw one candidate pool; smallest-term pools contain one term.
fn draw_candiates<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    policy: Policy,
    package: &FrontierPackage<BigUint, L, N>,
) -> Vec<RecExpr<OriginLang<L>>> {
    package
        .draw_candidates(args.n_candidates, policy, [args.seed, policy.rng_salt()])
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: policy {policy} drew 0 candidates (empty novel frontier); \
                     driver legs for this policy will have no guides to pick from"
            );
            Vec::new()
        })
}
