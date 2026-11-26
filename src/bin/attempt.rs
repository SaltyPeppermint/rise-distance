//! Run one attempt: union a set of guides, saturate, and report goal
//! reachability.
//!
//! Stateless — no guide egraph replay or samples construction.
//! `guided_search.py` spawns this once per attempt, passing `--goal` on argv
//! and the start input on stdin: with `--is-guide`, a JSON array of
//! origin-annotated [`OriginLang`] s-expressions; without `--is-guide`, a plain
//! s-expression for the unguided baseline.
//! Prints the run's [`AttemptSummary`] as the `payload` of a [`Measured`] envelope.
//!
//! One attempt per process is deliberate: the peak RSS reported in the
//! [`Measured`] envelope is a per-process lifetime high-water mark, so batching
//! several attempts into one invocation would report the max over all of them
//! while the unguided baseline — a single eqsat run in its own process —
//! reports just one. Keeping the work per process identical across both arms is
//! what makes the two peaks comparable. The driver owns the search loop and its
//! early stop.

use std::borrow::Cow;
use std::fmt::Display;
use std::io::{self, BufReader};

use clap::Parser;
use egg::{Analysis, FromOp, Iteration, Language, RecExpr, Rewrite};
use serde::Serialize;

use rise_distance::cli::Measured;
use rise_distance::eqsat::{self, AttemptError, EqsatConfig, ReachedRun};
use rise_distance::langs::{Lang, math, prop};
use rise_distance::origin::OriginLang;

#[derive(Parser)]
#[command(
    about = "Run one attempt: union a set of guides, saturate, report reachability",
    after_help = "\
Reads the start input from stdin: with `--is-guide`, a JSON array of
origin-annotated s-expressions; otherwise a plain s-expression. The result is
printed as a JSON object.

Example:
  echo 'x' | attempt --language math --goal '(+ x 0)' \\
    --max-iters 200 --max-nodes 1000000 --max-time 10
"
)]
struct Args {
    /// Which language's rules to run under (from the folder's `problem_args.json`).
    #[arg(long)]
    language: Lang,

    /// The goal as a lowered s-expression string.
    #[arg(long)]
    goal: String,

    /// Read stdin as a JSON array of guides rather than a baseline s-expression.
    #[arg(long, default_value_t = false)]
    is_guide: bool,

    /// Use the full-union add for the leg egraph.
    #[arg(long)]
    full_union: bool,

    #[command(flatten)]
    eqsat: EqsatConfig,
}

/// One attempt result, printed to stdout as JSON
fn main() {
    let args = Args::parse();

    match args.language {
        Lang::Math => run(&args, &math::rules()),
        Lang::Prop => run(&args, &prop::rules()),
    }
    println!();
}

/// Run this process's single eqsat
fn run<L, N>(args: &Args, rules: &[Rewrite<L, N>])
where
    L: Language + FromOp<Error: Display> + 'static,
    N: Analysis<L> + Default,
{
    let goal = args
        .goal
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse goal term '{}': {e}", args.goal));

    let result = if args.is_guide {
        let guide_strs: Vec<String> = serde_json::from_reader(BufReader::new(io::stdin().lock()))
            .unwrap_or_else(|e| panic!("Failed to parse start guides from stdin: {e}"));
        let guides: Vec<RecExpr<OriginLang<L>>> = guide_strs
            .iter()
            .map(|s| {
                s.parse()
                    .unwrap_or_else(|e| panic!("Failed to parse guide '{s}': {e}"))
            })
            .collect();
        eqsat::guided(&guides, &goal, rules, &args.eqsat, args.full_union)
    } else {
        let start_str =
            io::read_to_string(io::stdin()).expect("Failed to read start term from stdin");
        let start_expr = start_str
            .trim()
            .parse::<RecExpr<L>>()
            .unwrap_or_else(|e| panic!("Failed to parse start term '{start_str}': {e}"));
        eqsat::unguided(&start_expr, &goal, rules, &args.eqsat)
    };

    serde_json::to_writer(
        std::io::stdout(),
        &Measured::now(AttemptSummary::from(result)),
    )
    .expect("write attempt result JSON");
}

/// One attempt, flattened into the driver's attempt-row columns.
///
/// A panicked run leaves every measurement at `None`.
#[derive(Serialize)]
struct AttemptSummary {
    reached: bool,
    panic: bool,
    /// `goal_found`, `panic`, or egg's `StopReason` as `{:?}` (`NodeLimit(1000)`).
    stop_reason: Cow<'static, str>,
    iters: Option<usize>,
    nodes: Option<usize>,
    classes: Option<usize>,
    total_applied: Option<usize>,
    /// Sum of the iterations' `total_time`, in seconds.
    total_time: Option<f64>,
    /// Final process live heap.
    final_live_heap: Option<u64>,
    /// Peak process live heap.
    peak_live_heap: Option<u64>,
}

impl AttemptSummary {
    const PANIC: Self = Self {
        reached: false,
        panic: true,
        stop_reason: Cow::Borrowed("panic"),
        iters: None,
        nodes: None,
        classes: None,
        total_applied: None,
        total_time: None,
        final_live_heap: None,
        peak_live_heap: None,
    };

    fn measured(
        reached: bool,
        stop_reason: Cow<'static, str>,
        iterations: &[Iteration<()>],
        nodes: usize,
        classes: usize,
        final_live_heap: u64,
        peak_live_heap: u64,
    ) -> Self {
        Self {
            reached,
            panic: false,
            stop_reason,
            iters: Some(iterations.len()),
            nodes: Some(nodes),
            classes: Some(classes),
            total_applied: Some(iterations.iter().flat_map(|i| i.applied.values()).sum()),
            total_time: Some(iterations.iter().map(|i| i.total_time).sum()),
            final_live_heap: Some(final_live_heap),
            peak_live_heap: Some(peak_live_heap),
        }
    }
}

impl From<Result<ReachedRun, AttemptError>> for AttemptSummary {
    fn from(result: Result<ReachedRun, AttemptError>) -> Self {
        match result {
            Ok(run) => Self::measured(
                true,
                Cow::Borrowed("goal_found"),
                &run.iterations,
                run.nodes,
                run.classes,
                run.allocated,
                run.peak_allocated,
            ),
            Err(AttemptError::Unreached {
                stop_reason,
                iterations,
                nodes,
                classes,
                final_allocated,
                peak_allocated,
            }) => Self::measured(
                false,
                Cow::Owned(format!("{stop_reason:?}")),
                &iterations,
                nodes,
                classes,
                final_allocated,
                peak_allocated,
            ),
            Err(AttemptError::PanicWhileAttempt) => Self::PANIC,
        }
    }
}
