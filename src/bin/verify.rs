//! Run a single search leg: union the chosen guides, saturate, and report
//! whether the goal became reachable within the eqsat limits.
//!
//! Stateless one-shot wrapper over [`verify_reachability`] — no guide egraph
//! replay, no precompute. `driver.py` spawns this once per leg, passing the
//! guide subset (as serialized [`GuideExpr`] node lists, origins intact) and the
//! goal on stdin.

use std::io::Read;

use clap::Parser;
use egg::{RecExpr, Rewrite};
use serde::{Deserialize, Serialize};

use rise_distance::cli::GuideExpr;
use rise_distance::eqsat::{EqsatConfig, Goal, GuideError, verify_reachability};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::{MyAnalysis, MyLanguage};

#[derive(Parser)]
#[command(
    about = "Run one search leg: union guides, saturate, report goal reachability",
    after_help = "\
Reads a JSON `LegRequest` on stdin and prints a JSON `LegResult` on stdout.
`--language` and the eqsat limits come from argv. Example:
  echo '{\"full_union\":true,\"goal\":\"(+ x 0)\",\"guides\":[...]}' \\
    | verify --language math --max-iters 200 --max-nodes 1000000 --max-time 10 \\
      --backoff-scheduler
"
)]
struct Args {
    /// Which language's rules to run under (from the folder's `args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    /// Use the experimental full-union add for the leg egraph.
    #[arg(long)]
    full_union: bool,

    #[command(flatten)]
    eqsat: EqsatConfig,
}

/// One leg request, read from stdin as JSON. `guides` are node lists so origins
/// survive the round trip; `goal` is a lowered s-expression string.
#[derive(Deserialize)]
struct LegRequest {
    goal: String,
    /// Guide subset as raw JSON, parsed against the concrete language below.
    guides: serde_json::Value,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    panic: bool,
}

fn main() {
    let args = Args::parse();

    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .expect("read leg request from stdin");
    let request: LegRequest = serde_json::from_str(&buf).expect("parse leg request JSON");

    let eqsat = args.eqsat;

    let result = match args.language {
        AvailableLanguages::Diospyros => {
            run_leg::<_, ()>(&request, &args, &eqsat, &diospyros::rules(false, false))
        }
        AvailableLanguages::Math => {
            run_leg::<_, math::ConstantFold>(&request, &args, &eqsat, &math::rules())
        }
        AvailableLanguages::Prop => {
            run_leg::<_, prop::ConstantFold>(&request, &args, &eqsat, &prop::rules())
        }
    };

    serde_json::to_writer(std::io::stdout(), &result).expect("write leg result JSON");
    println!();
}

fn run_leg<L: MyLanguage, N: MyAnalysis<L>>(
    request: &LegRequest,
    args: &Args,
    eqsat: &EqsatConfig,
    rules: &[Rewrite<L, N>],
) -> LegResult {
    let guide_exprs: Vec<GuideExpr<L>> =
        serde_json::from_value(request.guides.clone()).expect("parse guide node lists");
    assert!(!guide_exprs.is_empty(), "leg needs at least one guide");
    let guides: Vec<RecExpr<_>> = guide_exprs
        .into_iter()
        .map(GuideExpr::into_recexpr)
        .collect();

    let goal_expr = request
        .goal
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse goal '{}': {e}", request.goal));
    let goal = Goal::Expr(goal_expr);

    match verify_reachability(&guides, &goal, rules, eqsat, args.full_union) {
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
            stop_reason: None,
            panic: true,
        },
    }
}
