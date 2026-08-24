//! Run one (seed, goal) pair's attempt loop: for each guide subset, union the
//! guides, saturate, and report goal reachability, stopping at the first reach.
//!
//! Stateless wrapper over [`verify_reachability`] — no guide egraph replay or
//! candidate construction. `guided_search.py` spawns this once per pair, passing the pair's
//! attempt subsets (serialized [`GuideExpr`] node lists) as a JSON array of
//! arrays on stdin and the goal on argv. Prints a JSON array of `LegResult`, one
//! per subset run — early-stopped, so possibly shorter than the input.

use std::io::Read;

use clap::Parser;
use egg::{RecExpr, Rewrite};
use serde::Serialize;

use rise_distance::cli::GuideExpr;
use rise_distance::eqsat::{
    EqsatConfig, Goal, GuideError, ReachedRun, verify_reachability, verify_unguided,
};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::{MyAnalysis, MyLanguage};

#[derive(Parser)]
#[command(
    about = "Run one (seed, goal) pair's attempt loop: union guides, saturate, report reachability",
    after_help = "\
Reads the pair's attempt subsets as a JSON array of guide-node-list arrays on
stdin and prints a JSON array of `LegResult` on stdout (one per subset run,
early-stopped at the first reach). `--goal`, `--language`, and the eqsat limits
come from argv. Example:
  echo '[[...],[...]]' \\
    | verify --language math --goal '(+ x 0)' --max-iters 200 \\
      --max-nodes 1000000 --max-time 10
"
)]
struct Args {
    /// Which language's rules to run under (from the folder's `goal_args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    /// The goal as a lowered s-expression string.
    #[arg(long)]
    goal: String,

    /// Run an ordinary single-seed baseline instead of reading guide subsets.
    #[arg(long)]
    seed: Option<String>,

    /// Use the full-union add for the leg egraph.
    #[arg(long)]
    full_union: bool,

    #[command(flatten)]
    eqsat: EqsatConfig,
}

/// One leg result, printed to stdout as JSON.
#[derive(Serialize)]
struct LegResult {
    reached: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    iters: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nodes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    classes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_applied: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_time: Option<f64>,
    /// This leg's absolute allocation (bytes): jemalloc `stats.allocated` for
    /// the whole process, the same coordinate system the configured memory
    /// ceiling is expressed in.
    #[serde(skip_serializing_if = "Option::is_none")]
    memory: Option<u64>,
    /// Largest sampled absolute live heap during the eqsat run.
    #[serde(skip_serializing_if = "Option::is_none")]
    peak_live_heap: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    panic: bool,
}

fn main() {
    let args = Args::parse();

    let subsets_json = if args.seed.is_none() {
        // The pair's attempt subsets come in on stdin as a JSON array of arrays
        // of serialized `GuideExpr` node lists.
        let mut json = String::new();
        std::io::stdin()
            .read_to_string(&mut json)
            .expect("read guide subsets from stdin");
        json
    } else {
        String::new()
    };

    let results = match args.language {
        AvailableLanguages::Diospyros if args.seed.is_some() => {
            vec![run_unguided::<_, ()>(
                &args,
                &diospyros::rules(false, false),
            )]
        }
        AvailableLanguages::Diospyros => {
            run_legs::<_, ()>(&subsets_json, &args, &diospyros::rules(false, false))
        }
        AvailableLanguages::Math if args.seed.is_some() => {
            vec![run_unguided::<_, math::ConstantFold>(&args, &math::rules())]
        }
        AvailableLanguages::Math => {
            run_legs::<_, math::ConstantFold>(&subsets_json, &args, &math::rules())
        }
        AvailableLanguages::Prop if args.seed.is_some() => {
            vec![run_unguided::<_, prop::ConstantFold>(&args, &prop::rules())]
        }
        AvailableLanguages::Prop => {
            run_legs::<_, prop::ConstantFold>(&subsets_json, &args, &prop::rules())
        }
    };

    serde_json::to_writer(std::io::stdout(), &results).expect("write leg results JSON");
    println!();
}

/// Run the pair's attempt loop: one leg per subset, stopping at the first
/// reach. Parses the goal once; a panicked leg surfaces as `panic: true` (caught
/// in [`verify_reachability`]) and the loop continues.
fn run_legs<L: MyLanguage, N: MyAnalysis<L>>(
    subsets_json: &str,
    args: &Args,
    rules: &[Rewrite<L, N>],
) -> Vec<LegResult> {
    let subsets: Vec<Vec<GuideExpr<L>>> =
        serde_json::from_str(subsets_json).expect("parse guide subset node lists");
    assert!(
        !subsets.is_empty(),
        "pair needs at least one attempt subset"
    );

    let goal_expr = args
        .goal
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse goal '{}': {e}", args.goal));
    let goal = Goal::Expr(goal_expr);

    let mut results = Vec::with_capacity(subsets.len());
    for guide_exprs in subsets {
        let result = run_leg(guide_exprs, &goal, args, rules);
        let reached = result.reached;
        results.push(result);
        if reached {
            break;
        }
    }
    results
}

fn run_leg<L: MyLanguage, N: MyAnalysis<L>>(
    guide_exprs: Vec<GuideExpr<L>>,
    goal: &Goal<L>,
    args: &Args,
    rules: &[Rewrite<L, N>],
) -> LegResult {
    assert!(!guide_exprs.is_empty(), "leg needs at least one guide");
    let guides: Vec<RecExpr<_>> = guide_exprs
        .into_iter()
        .map(GuideExpr::into_recexpr)
        .collect();

    result_to_leg(verify_reachability(
        &guides,
        goal,
        rules,
        &args.eqsat,
        args.full_union,
    ))
}

fn run_unguided<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    rules: &[Rewrite<L, N>],
) -> LegResult {
    let seed_text = args.seed.as_ref().expect("unguided mode needs --seed");
    let seed = seed_text
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed '{seed_text}': {e}"));
    let goal_expr = args
        .goal
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse goal '{}': {e}", args.goal));
    result_to_leg(verify_unguided(
        &seed,
        &Goal::Expr(goal_expr),
        rules,
        &args.eqsat,
    ))
}

fn result_to_leg<L: MyLanguage>(result: Result<ReachedRun<L>, GuideError>) -> LegResult {
    match result {
        Ok(run) => {
            let iterations = &run.iterations;
            LegResult {
                reached: true,
                iters: Some(iterations.len()),
                // True final egraph size (read off the rebuilt egraph after the
                // run), not the last iteration-boundary snapshot.
                nodes: Some(run.nodes),
                classes: Some(run.classes),
                total_applied: Some(
                    iterations
                        .iter()
                        .map(|i| i.applied.values().sum::<usize>())
                        .sum(),
                ),
                total_time: Some(iterations.iter().map(|i| i.total_time).sum()),
                memory: Some(run.allocated),
                peak_live_heap: Some(run.peak_allocated),
                stop_reason: None,
                panic: false,
            }
        }
        Err(GuideError::Unreached {
            stop_reason,
            final_allocated,
            peak_allocated,
        }) => LegResult {
            reached: false,
            iters: None,
            nodes: None,
            classes: None,
            total_applied: None,
            total_time: None,
            memory: Some(final_allocated),
            peak_live_heap: Some(peak_allocated),
            stop_reason: Some(format!("{stop_reason:?}")),
            panic: false,
        },
        Err(GuideError::PanicWhileAttempt) => LegResult {
            reached: false,
            iters: None,
            nodes: None,
            classes: None,
            total_applied: None,
            total_time: None,
            memory: None,
            peak_live_heap: None,
            stop_reason: None,
            panic: true,
        },
    }
}
