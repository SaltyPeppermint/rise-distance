//! Generate goal terms for a single start term expression.
//!
//! A stdout filter that touches no files: one start term per invocation, passed as
//! `--start-term <expr>` along with `--language` and the eqsat limits. The start term's
//! [`GoalGenMetadata`] (or the error string), as a `Result`-shaped `{"Ok": ..}`
//! / `{"Err": ..}` payload, is printed as JSON to stdout; all human-readable
//! logging goes to stderr. `scripts/generate_goals.py` owns all file I/O: it reads
//! `generation_args.json` (for `--language` and the eqsat flags),
//! flattens/filters `terms.json`, fans these invocations out one per start term,
//! keys each payload by its start term, and writes the enriched copy to
//! `goal_terms.json`.

use std::fmt::Write as _;
use std::time::Instant;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use num::BigUint;

use rise_distance::candidates::ExactCandidatePackage;
use rise_distance::cli::{GoalGenMetadata, Policy};
use rise_distance::eqsat::EqsatConfig;
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::lower;
use rise_distance::{MyAnalysis, MyLanguage, eqsat};

#[derive(Parser)]
#[command(
    about = "Generate goal terms for one start term (feeds scripts/generate_goals.py)",
    after_help = "\
Prints one start term's `{\"Ok\":..}`/`{\"Err\":..}` payload as JSON to stdout; logs
go to stderr. `scripts/generate_goals.py` fans these out and writes goal_terms.json.
Example:
  goal --start-term '(+ x 0)' --language math --max-iters 200 --max-nodes 1000000 \\
    --max-time 10   # -> payload JSON on stdout
"
)]
struct Args {
    /// The start term s-expression to generate goals from. `scripts/generate_goals.py`
    /// reads and flattens `terms.json` and passes one start term per invocation.
    #[arg(long)]
    start_term: String,

    /// Which language's rules to run under (from the folder's `generation_args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    #[command(flatten)]
    eqsat: EqsatConfig,

    /// Number of goal candidates to draw per start term.
    #[arg(long, default_value_t = 10)]
    n: usize,

    /// Policy used to draw goal candidates.
    #[arg(long, default_value_t = Policy::Count)]
    policy: Policy,

    /// How much to grow `max_size` on each size-search retry.
    #[arg(long, default_value_t = 5)]
    retry_step: usize,

    /// How many times to retry size discovery with a larger `max_size` before
    /// giving up on a start term.
    #[arg(long, default_value_t = 20)]
    max_retries: usize,

    /// How many novel sizes construction must find.
    #[arg(long, default_value_t = 5)]
    size_goal: usize,
}

fn main() {
    let args = Args::parse();

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
}

fn main_inner<L: MyLanguage, N: MyAnalysis<L>>(args: &Args, rules: &[Rewrite<L, N>]) {
    let eqsat = &args.eqsat;
    let start = args
        .start_term
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse start term '{}': {e}", args.start_term));

    let mut log = format!("[start term] {}\n", args.start_term);
    let enriched = process_start_term(args, eqsat, &start, rules, &mut log);
    match &enriched {
        Ok(g) => {
            writeln!(log, "Successfully generated {} goals!", g.goals.len()).unwrap();
        }
        Err(e) => {
            writeln!(log, "Failed to generate any goals due to: {e}").unwrap();
        }
    }
    // Logs to stderr; only the payload JSON goes to stdout so the driver can
    // capture it cleanly (a `Result`-shaped {"Ok":..}/{"Err":..} object).
    eprint!("{log}");
    serde_json::to_writer(std::io::stdout(), &enriched).expect("write goal payload JSON");
    println!();
}

fn process_start_term<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    eqsat: &EqsatConfig,
    start: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    log: &mut String,
) -> Result<GoalGenMetadata<BigUint>, String> {
    let Some(result) = eqsat::run_eqsat(start, rules.iter(), eqsat) else {
        return Err("big eqsat failed".to_owned());
    };

    // Absolute process live heap before exact candidate construction.
    let base_memory = result.allocated();
    let base_peak_live_heap = result.peak_allocated();

    let stop_reason = format!("{:?}", result.stop_reason());
    let goal = result.metadata();

    writeln!(log, "goal_iters={} stop={stop_reason}", goal.iters).unwrap();
    writeln!(
        log,
        "goal egraph:  {} nodes, {} classes in {:.2}s",
        goal.nodes, goal.classes, goal.time
    )
    .unwrap();

    let now = Instant::now();

    let start_size = AstSize.cost_rec(start);
    let (used_max_size, package) =
        ExactCandidatePackage::<BigUint, L, _>::build_through_novel_sizes(
            result,
            start_size,
            args.max_retries,
            args.retry_step,
            args.size_goal,
            log,
        )
        .map_err(|tried_max_size| {
            format!(
                "exact goal-candidate construction found too few novel sizes after {} retries \
             (goal_iters={}, max_size={})",
                args.max_retries, goal.iters, tried_max_size
            )
        })?;
    writeln!(
        log,
        "Exact candidate package built in {:.2}s",
        now.elapsed().as_secs_f64()
    )
    .unwrap();
    package.log_root_counts(log);

    let goals = package
        .draw_frontier_candidates(args.n, args.policy, [0, 0])
        .ok_or_else(|| "exact frontier candidate drawing failed".to_owned())?;

    let goal_strings = goals
        .iter()
        .map(|g| lower(g.clone()).to_string())
        .collect::<Vec<_>>();

    let frontier_histogram = package
        .root_histogram()
        .iter()
        .map(|(s, c)| (s.to_string(), c.clone()))
        .collect();

    Ok(GoalGenMetadata {
        max_size: used_max_size,
        goal_egraph: goal,
        base_memory,
        base_peak_live_heap,
        goals: goal_strings,
        frontier_histogram,
        stop_reason,
    })
}
