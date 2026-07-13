//! Generate goal terms for a single seed expression.
//!
//! A stdout filter that touches no files: one seed per invocation, passed as
//! `--seed <expr>` along with `--language` and the eqsat limits. The seed's
//! [`GoalGenMetadata`] (or the error string), as a `Result`-shaped `{"Ok": ..}`
//! / `{"Err": ..}` payload, is printed as JSON to stdout; all human-readable
//! logging goes to stderr. `scripts/goal_driver.py` owns all file I/O: it reads
//! `args.json` (for `--language` and the eqsat flags), flattens/filters
//! `terms.json`, fans these invocations out one per seed, keys each payload by
//! its seed, and merges them back into `terms.json`.

use std::fmt::Write as _;
use std::time::Instant;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use num::BigUint;

use rise_distance::cli::{EqsatArgs, GoalGenMetadata};
use rise_distance::eqsat::{EqsatConfig, SplitMetadata};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::lower;
use rise_distance::sampling::{PrecomputePackage, SampleStrategy, TermSampleDist};
use rise_distance::{MyAnalysis, MyLanguage, eqsat};

#[derive(Parser)]
#[command(
    about = "Generate goal terms for one seed (feeds scripts/goal_driver.py)",
    after_help = "\
Prints one seed's `{\"Ok\":..}`/`{\"Err\":..}` payload as JSON to stdout; logs
go to stderr. `scripts/goal_driver.py` fans these out and merges into terms.json.
Example:
  goal --seed '(+ x 0)' --language math --max-iters 200 --max-nodes 1000000 \\
    --max-time 10 --backoff-scheduler   # -> payload JSON on stdout
"
)]
struct Args {
    /// The seed s-expression to generate goals from. `scripts/goal_driver.py`
    /// reads and flattens `terms.json` and passes one seed per invocation.
    #[arg(long)]
    seed: String,

    /// Which language's rules to run under (from the folder's `args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    #[command(flatten)]
    eqsat: EqsatArgs,

    /// Number of goal terms to sample per seed.
    #[arg(long, default_value_t = 10)]
    goals: usize,

    /// How to distribute the sample budget across sizes.
    #[arg(long, default_value_t = TermSampleDist::GREEDY)]
    size_distribution: TermSampleDist,

    /// How to sample the GOAL terms.
    #[arg(long, default_value_t = SampleStrategy::Count)]
    goal_sample_strategy: SampleStrategy,

    /// How much to grow `max_size` on each precompute retry.
    #[arg(long, default_value_t = 5)]
    retry_step: usize,

    /// How many times to retry precompute with a larger `max_size` before
    /// giving up on a seed.
    #[arg(long, default_value_t = 20)]
    max_retries: usize,

    /// How many sizes need to be present in the precomputed histogram of root
    #[arg(long, default_value_t = 5)]
    sample_sizes: usize,
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
    let eqsat: EqsatConfig = args.eqsat.into();
    let seed_expr = args
        .seed
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed '{}': {e}", args.seed));

    let mut log = format!("[seed] {}\n", args.seed);
    let enriched = process_seed(args, &eqsat, &seed_expr, rules, &mut log);
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

    let stop_reason = format!("{:?}", result.stop_reason());
    let SplitMetadata { guide, goal } = result.split_metadata();

    writeln!(
        log,
        "goal_iters={} guide_iters={} stop={stop_reason}",
        goal.iters, guide.iters
    )
    .unwrap();
    writeln!(
        log,
        "guide egraph: {} nodes, {} classes in {:.2}s",
        guide.nodes, guide.classes, guide.time
    )
    .unwrap();
    writeln!(
        log,
        "goal egraph:  {} nodes, {} classes in {:.2}s",
        goal.nodes, goal.classes, goal.time
    )
    .unwrap();

    let now = Instant::now();

    let start_size = AstSize.cost_rec(seed_expr);
    let (used_max_size, pp) = PrecomputePackage::<BigUint, L, _>::backoff_precompute(
        &result,
        start_size,
        args.max_retries,
        args.retry_step,
        args.sample_sizes,
        log,
    )
    .map_err(|tried_max_size| {
        format!(
            "goal precompute returned None after {} retries (goal_iters={}, max_size={})",
            args.max_retries, goal.iters, tried_max_size
        )
    })?;
    writeln!(
        log,
        "Precompute built in {:.2}s",
        now.elapsed().as_secs_f64()
    )
    .unwrap();
    pp.log_root(log);

    let goals = pp
        .sample_frontier_terms(
            args.goals,
            args.size_distribution,
            args.goal_sample_strategy,
            [0, 0],
            true,
        )
        .ok_or_else(|| "sample_frontier failed".to_owned())?;

    let goal_strings = goals
        .iter()
        .map(|g| lower(g.clone()).to_string())
        .collect::<Vec<_>>();

    let frontier_histogram = pp
        .root_histogram()
        .iter()
        .map(|(s, c)| (s.to_string(), c.clone()))
        .collect();

    Ok(GoalGenMetadata {
        max_size: used_max_size,
        goal_egraph: goal,
        guide_egraph: guide,
        goals: goal_strings,
        frontier_histogram,
        stop_reason,
    })
}
