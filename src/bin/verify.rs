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

use clap::Parser;
use egg::{RecExpr, Rewrite};

use rise_distance::eqsat::{EqsatConfig, Goal, guided_eqsat, unguided_eqsat};
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
    start_term: String,

    #[arg(long, default_value_t = false)]
    is_guide: bool,

    /// Use the full-union add for the leg egraph.
    #[arg(long)]
    full_union: bool,

    #[command(flatten)]
    eqsat: EqsatConfig,
}

/// One leg result, printed to stdout as JSON
fn main() {
    let args = Args::parse();

    match args.language {
        AvailableLanguages::Diospyros => run::<_, ()>(&args, &diospyros::rules(false, false)),
        AvailableLanguages::Math => run::<_, math::ConstantFold>(&args, &math::rules()),
        AvailableLanguages::Prop => run::<_, prop::ConstantFold>(&args, &prop::rules()),
    }
    println!();
}

/// Run this process's single eqsat
fn run<L: MyLanguage, N: MyAnalysis<L>>(args: &Args, rules: &[Rewrite<L, N>]) {
    let goal_expr = args
        .goal_term
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse goal term '{}': {e}", args.goal_term));
    let goal = Goal::Expr(goal_expr);

    let result = if args.is_guide {
        let start_term: Vec<OriginLang<L>> = serde_json::from_str(&args.start_term)
            .unwrap_or_else(|e| panic!("Failed to parse start term '{}': {e}", args.start_term));
        guided_eqsat(
            &[RecExpr::from(start_term)],
            &goal,
            rules,
            &args.eqsat,
            args.full_union,
        )
    } else {
        let start_term = args
            .start_term
            .parse::<RecExpr<L>>()
            .unwrap_or_else(|e| panic!("Failed to parse start term '{}': {e}", args.start_term));
        unguided_eqsat(&start_term, &goal, rules, &args.eqsat)
    };

    serde_json::to_writer(std::io::stdout(), &result).expect("write leg result JSON");
}
