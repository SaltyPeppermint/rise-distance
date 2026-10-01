//! Run one attempt: union a single guide subset, saturate, and report goal
//! reachability.
//!
//! Stateless — no guide egraph replay or samples construction.
//! `guided_search.py` spawns this once per attempt, passing everything on argv:
//! the goal via `--goal` and, with `--is-guide`, that attempt's guide as a JSON
//! array of [`OriginLang`] nodes via `--start`. Without `--is-guide` the same
//! flag takes a plain s-expression and the run is the unguided baseline. Prints
//! the run's `Result<ReachedRun, GuideError>` as the `payload` of a
//! [`Measured`] envelope.
//!
//! One attempt per process is deliberate: the peak RSS reported in the
//! [`Measured`] envelope is a per-process lifetime high-water mark, so batching
//! several attempts into one invocation would report the max over all of them
//! while the unguided baseline — a single eqsat run in its own process —
//! reports just one. Keeping the work per process identical across both arms is
//! what makes the two peaks comparable. The driver owns the search loop and its
//! early stop.

use clap::Parser;
use egg::{RecExpr, Rewrite};

use rise_distance::cli::Measured;
use rise_distance::eqsat::{self, EqsatConfig, Goal};
use rise_distance::langs::{AvailableLanguages, MyAnalysis, MyLanguage, diospyros, math, prop};
use rise_distance::origin::OriginLang;

#[derive(Parser)]
#[command(
    about = "Run one attempt: union a single guide subset, saturate, report reachability",
    after_help = "\
With `--is-guide`, `--start` is one attempt's guide as a JSON array of guide
nodes. Otherwise `--start` is a plain s-expression. The result is printed as a
JSON object.

Example:
  attempt --language math --goal '(+ x 0)' --start 'x' \\
    --max-iters 200 --max-nodes 1000000 --max-time 10
"
)]
struct Args {
    /// Which language's rules to run under (from the folder's `problem_args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    /// The goal as a lowered s-expression string.
    #[arg(long)]
    goal: String,

    /// The start of the run: a JSON array of guide nodes with `--is-guide`,
    /// otherwise a plain s-expression for the unguided baseline.
    #[arg(long)]
    start: String,

    /// Read `--start` as a guide rather than as a baseline s-expression.
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
        AvailableLanguages::Diospyros => run(&args, &diospyros::rules(false, false)),
        AvailableLanguages::Math => run(&args, &math::rules()),
        AvailableLanguages::Prop => run(&args, &prop::rules()),
    }
    println!();
}

/// Run this process's single eqsat
fn run<L: MyLanguage, N: MyAnalysis<L>>(args: &Args, rules: &[Rewrite<L, N>]) {
    let goal = args
        .goal
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse goal term '{}': {e}", args.goal));
    let goal = Goal::Expr(goal);

    let result = if args.is_guide {
        let guide: Vec<OriginLang<L>> = serde_json::from_str(&args.start)
            .unwrap_or_else(|e| panic!("Failed to parse start term '{}': {e}", args.start));
        eqsat::guided_eqsat(
            &[RecExpr::from(guide)],
            &goal,
            rules,
            &args.eqsat,
            args.full_union,
        )
    } else {
        let start_expr = args
            .start
            .parse::<RecExpr<L>>()
            .unwrap_or_else(|e| panic!("Failed to parse start term '{}': {e}", args.start));
        eqsat::unguided_eqsat(&start_expr, &goal, rules, &args.eqsat)
    };

    serde_json::to_writer(std::io::stdout(), &Measured::now(result))
        .expect("write attempt result JSON");
}
