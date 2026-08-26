//! Run one leg: union a single guide subset, saturate, and report goal
//! reachability.
//!
//! Stateless wrapper over [`verify_reachability`] — no guide egraph replay or
//! candidate construction. `guided_search.py` spawns this once per leg, passing
//! that leg's guide subset (serialized [`GuideExpr`] node lists) as a JSON array
//! on stdin and the goal on argv. Prints the leg's `LegResult` as a JSON object.
//!
//! One leg per process is deliberate: peak RSS (`ru_maxrss`) is a per-process
//! lifetime high-water mark, so batching several legs into one invocation would
//! report the max over all of them while the unguided baseline — a single eqsat
//! run in its own process — reports just one. Keeping the work per process
//! identical across both arms is what makes the two peaks comparable. The driver
//! owns the attempt loop and its early stop.

use std::io::Read;

use clap::Parser;
use egg::{RecExpr, Rewrite};
use serde::Serialize;

use rise_distance::cli::GuideExpr;
use rise_distance::eqsat::{
    EqsatConfig, Goal, GuideError, ReachedRun, verify_reachability, verify_unguided,
};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::{MyAnalysis, MyLanguage, OriginLang};

#[derive(Parser)]
#[command(
    about = "Run one leg: union a single guide subset, saturate, report reachability",
    after_help = "\
Reads one leg's guide subset as a JSON array of guide-node-lists on stdin and
prints its `LegResult` as a JSON object. With `--start-term` it instead runs the
unguided baseline and reads nothing from stdin. `--goal-term`, `--language`, and
the eqsat limits come from argv. The driver runs the attempt loop, one process
per leg, so each process's peak RSS covers exactly one eqsat run — the same unit
as the unguided baseline. Example:
  echo '[...]' \\
    | verify --language math --goal-term '(+ x 0)' --max-iters 200 \\
      --max-nodes 1000000 --max-time 10
"
)]
struct Args {
    /// Which language's rules to run under (from the folder's `goal_args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    /// The goal as a lowered s-expression string.
    #[arg(long)]
    goal_term: String,

    /// Run an ordinary single-start-term baseline instead of reading guide subsets.
    #[arg(long)]
    start_term: Option<String>,

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

    let result = match args.language {
        AvailableLanguages::Diospyros => run::<_, ()>(&args, &diospyros::rules(false, false)),
        AvailableLanguages::Math => run::<_, math::ConstantFold>(&args, &math::rules()),
        AvailableLanguages::Prop => run::<_, prop::ConstantFold>(&args, &prop::rules()),
    };

    serde_json::to_writer(std::io::stdout(), &result).expect("write leg result JSON");
    println!();
}

/// Run this process's single eqsat: the unguided baseline when `--start-term`
/// is given, otherwise the guided leg whose subset arrives on stdin.
///
/// A panic surfaces as `panic: true` (caught inside the `verify_*` helpers)
/// rather than aborting, so the driver still gets a result for the attempt.
fn run<L: MyLanguage, N: MyAnalysis<L>>(args: &Args, rules: &[Rewrite<L, N>]) -> LegResult {
    let goal_expr = args
        .goal_term
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse goal term '{}': {e}", args.goal_term));
    let goal = Goal::Expr(goal_expr);

    let Some(start_text) = args.start_term.as_ref() else {
        return result_to_leg(verify_reachability(
            &read_guides(),
            &goal,
            rules,
            &args.eqsat,
            args.full_union,
        ));
    };

    let start_term = start_text
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse start term '{start_text}': {e}"));
    result_to_leg(verify_unguided(&start_term, &goal, rules, &args.eqsat))
}

/// Read this leg's guide subset from stdin: a JSON array of serialized
/// [`GuideExpr`] node lists.
fn read_guides<L: MyLanguage>() -> Vec<RecExpr<OriginLang<L>>> {
    let mut json = String::new();
    std::io::stdin()
        .read_to_string(&mut json)
        .expect("read guide subset from stdin");
    let guide_exprs: Vec<GuideExpr<L>> =
        serde_json::from_str(&json).expect("parse guide subset node lists");
    assert!(!guide_exprs.is_empty(), "leg needs at least one guide");
    guide_exprs
        .into_iter()
        .map(GuideExpr::into_recexpr)
        .collect()
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
