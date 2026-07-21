//! Run a single search leg: union the chosen guides, saturate, and report
//! whether the goal became reachable within the eqsat limits.
//!
//! Stateless one-shot wrapper over [`verify_reachability`] — no guide egraph
//! replay, no precompute. `guided_search.py` spawns this once per leg, passing the
//! guide subset (as serialized [`GuideExpr`] node lists, origins intact) on
//! stdin and the goal on argv.

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
    about = "Run one search leg: union guides, saturate, report goal reachability",
    after_help = "\
Reads the guide subset as a JSON array on stdin and prints a JSON `LegResult`
on stdout. `--goal`, `--language`, and the eqsat limits come from argv. Example:
  echo '[...]' \\
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
    /// Live-heap bytes (jemalloc `stats.allocated`), sampled after
    /// `verify_reachability` returns.
    #[serde(skip_serializing_if = "Option::is_none")]
    memory: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    panic: bool,
}

fn main() {
    let args = Args::parse();

    // Guides come in on stdin as a JSON array of serialized `GuideExpr` node
    // lists; they're parsed against the concrete language inside `run_leg`.
    let mut guides_json = String::new();
    std::io::stdin()
        .read_to_string(&mut guides_json)
        .expect("read guides from stdin");

    let result = match args.language {
        AvailableLanguages::Diospyros => {
            run_leg::<_, ()>(&guides_json, &args, &diospyros::rules(false, false))
        }
        AvailableLanguages::Math => {
            run_leg::<_, math::ConstantFold>(&guides_json, &args, &math::rules())
        }
        AvailableLanguages::Prop => {
            run_leg::<_, prop::ConstantFold>(&guides_json, &args, &prop::rules())
        }
    };

    serde_json::to_writer(std::io::stdout(), &result).expect("write leg result JSON");
    println!();
}

fn run_leg<L: MyLanguage, N: MyAnalysis<L>>(
    guides_json: &str,
    args: &Args,
    rules: &[Rewrite<L, N>],
) -> LegResult {
    let guide_exprs: Vec<GuideExpr<L>> =
        serde_json::from_str(guides_json).expect("parse guide node lists");
    assert!(!guide_exprs.is_empty(), "leg needs at least one guide");
    let guides: Vec<RecExpr<_>> = guide_exprs
        .into_iter()
        .map(GuideExpr::into_recexpr)
        .collect();

    let goal_expr = args
        .goal
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse goal '{}': {e}", args.goal));
    let goal = Goal::Expr(goal_expr);

    match verify_reachability(&guides, &goal, rules, &args.eqsat, args.full_union) {
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
                memory: Some(live_heap_bytes()),
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
