//! Run one (seed, goal) pair's attempt loop: for each guide subset, union the
//! guides, saturate, and report goal reachability, stopping at the first reach.
//!
//! Stateless wrapper over [`verify_reachability`] — no guide egraph replay or
//! precompute. `guided_search.py` spawns this once per pair, passing the pair's
//! attempt subsets (serialized [`GuideExpr`] node lists) as a JSON array of
//! arrays on stdin and the goal on argv. Prints a JSON array of `LegResult`, one
//! per subset run — early-stopped, so possibly shorter than the input.

use std::io::Read;

use clap::Parser;
use egg::{RecExpr, Rewrite};
use serde::Serialize;

use rise_distance::cli::GuideExpr;
use rise_distance::eqsat::{EqsatConfig, Goal, GuideError, verify_reachability};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::utils::live_heap_bytes;
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
      --max-nodes 1000000 --max-time 10 --backoff-scheduler
"
)]
struct Args {
    /// Which language's rules to run under (from the folder's `goal_args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    /// The goal as a lowered s-expression string.
    #[arg(long)]
    goal: String,

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
    /// This leg's live-heap growth (bytes): jemalloc `stats.allocated` after
    /// `verify_reachability` minus a sample taken just before the leg, so it
    /// isolates this leg's egraph from any baseline left by earlier legs in the
    /// shared process.
    #[serde(skip_serializing_if = "Option::is_none")]
    memory: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    panic: bool,
}

fn main() {
    let args = Args::parse();

    // The pair's attempt subsets come in on stdin as a JSON array of arrays of
    // serialized `GuideExpr` node lists; they're parsed against the concrete
    // language inside `run_legs`.
    let mut subsets_json = String::new();
    std::io::stdin()
        .read_to_string(&mut subsets_json)
        .expect("read guide subsets from stdin");

    let results = match args.language {
        AvailableLanguages::Diospyros => {
            run_legs::<_, ()>(&subsets_json, &args, &diospyros::rules(false, false))
        }
        AvailableLanguages::Math => {
            run_legs::<_, math::ConstantFold>(&subsets_json, &args, &math::rules())
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
    assert!(!subsets.is_empty(), "pair needs at least one attempt subset");

    let goal_expr = args
        .goal
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse goal '{}': {e}", args.goal));
    let goal = Goal::Expr(goal_expr);

    let mut results = Vec::with_capacity(subsets.len());
    for guide_exprs in subsets {
        // Baseline for this leg's memory delta (see `LegResult::memory`).
        let pre = live_heap_bytes();
        let result = run_leg(guide_exprs, &goal, pre, args, rules);
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
    pre_heap: u64,
    args: &Args,
    rules: &[Rewrite<L, N>],
) -> LegResult {
    assert!(!guide_exprs.is_empty(), "leg needs at least one guide");
    let guides: Vec<RecExpr<_>> = guide_exprs
        .into_iter()
        .map(GuideExpr::into_recexpr)
        .collect();

    match verify_reachability(&guides, goal, rules, &args.eqsat, args.full_union) {
        Ok((iterations, _target)) => {
            let last = iterations.last().expect("reached leg logged no iterations");
            LegResult {
                reached: true,
                iters: Some(iterations.len()),
                nodes: Some(last.egraph_nodes),
                classes: Some(last.egraph_classes),
                total_applied: Some(
                    iterations
                        .iter()
                        .map(|i| i.applied.values().sum::<usize>())
                        .sum(),
                ),
                total_time: Some(iterations.iter().map(|i| i.total_time).sum()),
                memory: Some(live_heap_bytes().saturating_sub(pre_heap)),
                stop_reason: None,
                panic: false,
            }
        }
        Err(GuideError::Unreached(stop)) => LegResult {
            reached: false,
            iters: None,
            nodes: None,
            classes: None,
            total_applied: None,
            total_time: None,
            memory: None,
            stop_reason: Some(format!("{stop:?}")),
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
            stop_reason: None,
            panic: true,
        },
    }
}
