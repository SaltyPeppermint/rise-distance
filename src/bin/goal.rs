//! Generate goal terms for a single seed expression.
//!
//! A stdout filter that touches no files: one seed per invocation, passed as
//! `--seed <expr>` along with `--language` and the eqsat limits. The seed's
//! [`GoalGenMetadata`] (or the error string), as a `Result`-shaped `{"Ok": ..}`
//! / `{"Err": ..}` payload, is printed as JSON to stdout; all human-readable
//! logging goes to stderr. `scripts/generate_goals.py` owns all file I/O: it reads
//! `generation_args.json` (for `--language` and the eqsat flags),
//! flattens/filters `terms.json`, fans these invocations out one per seed,
//! keys each payload by its seed, and writes the enriched copy to
//! `goal_terms.json`.

use std::fmt::Write as _;
use std::time::Instant;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use num::BigUint;

use rise_distance::candidates::{ExactCandidatePackage, SizeAllocation};
use rise_distance::cli::{GoalGenMetadata, Policy};
use rise_distance::eqsat::EqsatConfig;
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::lower;
use rise_distance::{MyAnalysis, MyLanguage, eqsat};

#[derive(Parser)]
#[command(
    about = "Generate goal terms for one seed (feeds scripts/generate_goals.py)",
    after_help = "\
Prints one seed's `{\"Ok\":..}`/`{\"Err\":..}` payload as JSON to stdout; logs
go to stderr. `scripts/generate_goals.py` fans these out and writes goal_terms.json.
Example:
  goal --seed '(+ x 0)' --language math --max-iters 200 --max-nodes 1000000 \\
    --max-time 10   # -> payload JSON on stdout
"
)]
struct Args {
    /// The seed s-expression to generate goals from. `scripts/generate_goals.py`
    /// reads and flattens `terms.json` and passes one seed per invocation.
    #[arg(long)]
    seed: String,

    /// Which language's rules to run under (from the folder's `generation_args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    #[command(flatten)]
    eqsat: EqsatConfig,

    /// Number of goal candidates to draw per seed.
    #[arg(long, default_value_t = 10)]
    goals: usize,

    /// How to allocate the goal-candidate budget across sizes.
    #[arg(long, default_value_t = SizeAllocation::Greedy)]
    size_allocation: SizeAllocation,

    /// Exact policy used to draw goal candidates.
    #[arg(long, default_value_t = Policy::Count)]
    selection_policy: Policy,

    /// How much to grow `max_size` on each exact-size-search retry.
    #[arg(long, default_value_t = 5)]
    retry_step: usize,

    /// How many times to retry exact size discovery with a larger `max_size` before
    /// giving up on a seed.
    #[arg(long, default_value_t = 20)]
    max_retries: usize,

    /// How many novel sizes exact construction must find.
    #[arg(long, default_value_t = 5)]
    novel_size_goal: usize,
}

fn main() {
    let args = Args::parse();

    eprintln!("Language: {:?}", args.language);
    eprintln!("Seed: {}", args.seed);

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
    let seed_expr = args
        .seed
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed '{}': {e}", args.seed));

    let mut log = format!("[seed] {}\n", args.seed);
    let enriched = process_seed(args, eqsat, &seed_expr, rules, &mut log);
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

fn process_seed<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    eqsat: &EqsatConfig,
    seed_expr: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    log: &mut String,
) -> Result<GoalGenMetadata<BigUint>, String> {
    let Some(result) = eqsat::run_eqsat(seed_expr, rules.iter(), eqsat) else {
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

    let start_size = AstSize.cost_rec(seed_expr);
    let (used_max_size, package) =
        ExactCandidatePackage::<BigUint, L, _>::build_through_novel_sizes(
            result,
            start_size,
            args.max_retries,
            args.retry_step,
            args.novel_size_goal,
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
        .draw_frontier_candidates(
            args.goals,
            args.size_allocation,
            args.selection_policy,
            [0, 0],
        )
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
