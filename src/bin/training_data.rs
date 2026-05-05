use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use egg::RecExpr;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use num::BigUint;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;

use rise_distance::cli::argparse::{
    EqsatConfig, SampleStrategy, SeedInput, TermSampleDist, parse_seeds,
};
use rise_distance::cli::parquet::dump_full_eval_parquet;
use rise_distance::cli::{
    ExperimentError, GuideEval, PrecomputePackage, get_run_folder, init_log, measure_guide,
};
use rise_distance::cli::{write_config, write_stats};
use rise_distance::count::Counter;
use rise_distance::egg::math::{ConstantFold, Math, RULES};
use rise_distance::egg::{ToEgg, big_eqsat, verify_reachability};
use rise_distance::{OriginTree, TreeShaped, tee_println};

#[derive(Parser, Serialize)]
#[command(
    about = "Evaluate sampled guides against goals to generate training data",
    after_help = "\
Examples:
  # Basic run with 100 random guides, single seed
  training_data --seed '(d x (+ (* x x) 1))' --max-size 150 -g 100

  # Sample 5 goals and 100 guides
  training_data --seed '(d x (+ (* x x) 1))' --max-size 150 -g 100 --goals 5

  # Loop over many seeds from a JSON file (objects with size and term fields; size drives max-size)
  training_data --seed-json seeds.json -g 100

  # Write output to a specific folder
  training_data --seed '(d x (+ (* x x) 1))' --max-size 150 -g 100 -o results/

  # Use the experimental full-union egraph
  training_data --seed '(d x (+ (* x x) 1))' --max-size 150 -g 100 --full-union
"
)]
struct Args {
    /// Seed term as an s-expression (Math language). Requires --max-size.
    #[arg(long, group = "seed_input", requires = "max_size")]
    seed: Option<String>,

    /// Path to a JSON file with objects containing `size` and `term` fields. The `size` field drives --max-size.
    /// Mutually exclusive with --seed / --max-size.
    #[arg(long, group = "seed_input", conflicts_with = "max_size")]
    seed_json: Option<PathBuf>,

    /// Node limit for the baseline egraph
    #[arg(long, default_value_t = 1_000_000_000)]
    max_nodes: usize,

    /// Time limit for the baseline egraph in seconds
    #[arg(long, default_value_t = 0.2)]
    max_time: f64,

    /// Iteration limit for the baseline egraph
    #[arg(long, default_value_t = usize::MAX)]
    max_iters: usize,

    /// Number of guide candidates to sample from the n-1 frontier
    #[arg(short, long)]
    guides: usize,

    /// Number of goal terms to sample from the n frontier
    #[arg(long, default_value_t = 1)]
    goals: usize,

    /// Max term size for counting/sampling. Required with --seed; forbidden with --seed-json.
    #[arg(long)]
    max_size: Option<usize>,

    /// How to distribute the sample budget across sizes.
    /// Options: uniform, proportional:<`min_per_size`>, normal:<sigma>
    #[arg(long, default_value_t = TermSampleDist::UNIFORM)]
    size_distribution: TermSampleDist,

    /// How to sample the individual terms.
    #[arg(long, default_value_t = SampleStrategy::CountBased)]
    guide_sample_strategy: SampleStrategy,

    /// How to sample the GOAL terms.
    #[arg(long, default_value_t = SampleStrategy::CountBased)]
    goal_sample_strategy: SampleStrategy,

    /// Output folder (generated if omitted)
    #[arg(short, long)]
    output: Option<String>,

    /// Use the experimental `add_with_full_union` for the new egraph
    #[arg(long)]
    full_union: bool,

    /// Use egg's `BackoffScheduler` instead of the `SimpleScheduler`
    #[arg(long, default_value_t = false)]
    backoff_scheduler: bool,
}

fn main() {
    let args = Args::parse();
    let eqsat = EqsatConfig {
        max_iters: args.max_iters,
        max_nodes: args.max_nodes,
        max_time: args.max_time,
        backoff_scheduler: args.backoff_scheduler,
    };
    let prefix = format!(
        "nodes-{}-timems-{}-strategy-{}-fullunion-{}",
        eqsat.max_nodes,
        Duration::from_secs_f64(eqsat.max_time).as_millis(),
        args.guide_sample_strategy,
        args.full_union
    );
    let run_folder = get_run_folder(args.output.as_deref(), "training_data", &prefix);
    init_log(&run_folder);
    tee_println!("Run Folder: {}", run_folder.to_string_lossy());

    let seed_input = match (&args.seed, &args.seed_json) {
        (Some(s), None) => SeedInput::Single {
            term: s.clone(),
            max_size: args.max_size.expect("--max-size required with --seed"),
        },
        (None, Some(p)) => SeedInput::JSON(p.clone()),
        _ => panic!("clap group enforces exactly one of --seed / --seed-json"),
    };
    let seeds = parse_seeds(seed_input);

    tee_println!("Distribution: {}", args.size_distribution);
    tee_println!("Seeds to process: {}", seeds.len());

    let mut all_stats = Vec::new();

    for (i, (seed_str, seed_expr, max_size)) in seeds.iter().enumerate() {
        tee_println!("\n=== Seed {i}: {seed_str} (max_size={max_size}) ===");
        if let Some(stats) =
            process_seed(&args, &eqsat, seed_str, seed_expr, *max_size, &run_folder)
        {
            all_stats.push(stats);
        }
    }

    write_config(&run_folder, &args);
    write_stats(&run_folder, &all_stats);
}

/// Run eqsat for one seed, sample goals, evaluate each goal, and return the
/// collected results and per-goal stats. Returns `None` if the goal frontier
/// is empty (seed is skipped with a warning).
fn process_seed(
    args: &Args,
    eqsat: &EqsatConfig,
    seed_str: &str,
    seed_expr: &RecExpr<Math>,
    max_size: usize,
    run_folder: &Path,
) -> Option<serde_json::Value> {
    let result = big_eqsat(seed_expr, RULES.iter(), eqsat)?;
    tee_println!("Goal Iterations: {}", result.goal_iters());
    tee_println!("Guide Iterations: {}", result.guide_iters());
    tee_println!("Stop Reason: {:?}", result.stop_reason());

    let guide_secs = result
        .guide_data()
        .iter()
        .map(|i| i.total_time)
        .sum::<f64>();
    let goal_secs = result.goal_data().iter().map(|i| i.total_time).sum::<f64>();

    let guide_nodes = result.guide().total_number_of_nodes();
    let guide_classes = result.guide().classes().len();
    let goal_nodes = result.goal().total_number_of_nodes();
    let goal_classes = result.goal().classes().len();
    let goal_iters = result.goal_iters();
    tee_println!(
        "Guide egraph had {guide_nodes} nodes, {guide_classes} classes in {guide_secs:.2}s"
    );
    tee_println!("Final egraph had {goal_nodes} nodes, {goal_classes} classes in {goal_secs:.2}s");

    tee_println!("\nSampling goals from iteration-{goal_iters} frontier...",);
    let pp = PrecomputePackage::<BigUint, Math, _>::precompute(
        result.goal(),
        result.prev_goal().to_owned(),
        result.root(),
        max_size,
    )?;
    pp.log_root();
    let Ok(goals) = pp.sample_frontier_terms::<true>(
        args.goals,
        args.size_distribution,
        args.goal_sample_strategy,
        [0, 0],
    ) else {
        tee_println!("WARNING: Not enough goals in the frontier for seed '{seed_str}'. Skipping.");
        return None;
    };

    tee_println!("Sampled {} goal(s)", goals.len());
    let stats = json!({
        "seed": seed_str,
        "max_size": max_size,
        "guide_egraph_iters": result.guide_iters(),
        "guide_egraph_nodes": guide_nodes,
        "guide_egraph_classes": guide_classes,
        "guide_eqsat_time": guide_secs,
        "goal_egraph_iters": result.goal_iters(),
        "goal_egraph_nodes": goal_nodes,
        "goal_egraph_classes": goal_classes,
        "goal_eqsat_time": goal_secs,
    });

    let pc = PrecomputePackage::<BigUint, _, _>::precompute(
        result.guide(),
        result.prev_guide().clone(),
        result.root(),
        max_size,
    )?;

    tee_println!("\nRunning top_k experiments NO REPLACEMENT...");
    for goal in goals {
        tee_println!("Current goal: {}", goal.to_string());
        match try_all(args, eqsat, &goal, &pc) {
            Err(e) => tee_println!("ERROR TRYING TO SAMPLE FOR {goal}: {e}"),
            Ok(results) => dump_full_eval_parquet(run_folder, seed_str, &goal, &results),
        }
    }

    Some(stats)
}

fn try_all<C: Counter + Display + Ord>(
    args: &Args,
    eqsat: &EqsatConfig,
    goal: &OriginTree<Math>,
    pc: &PrecomputePackage<C, Math, ConstantFold>,
) -> Result<Vec<GuideEval<Math>>, ExperimentError> {
    let goal_recexpr = goal.to_rec_expr();
    let samples = pc.sample_frontier_terms::<true>(
        args.guides,
        args.size_distribution,
        args.guide_sample_strategy,
        [0, 0],
    )?;

    let goal_flat = goal.flatten(false);

    let style = ProgressStyle::with_template("{msg:>6} [{bar:40}] {pos}/{len}")
        .unwrap()
        .progress_chars("=> ");
    samples
        .into_par_iter()
        .map(|guide| {
            let v = verify_reachability(
                std::slice::from_ref(&guide),
                &goal_recexpr,
                &RULES,
                eqsat,
                args.full_union,
            );
            let measurements = measure_guide(&guide, &goal_flat);
            Ok(GuideEval {
                guide,
                measurements,
                iterations: v,
            })
        })
        .progress_with_style(style)
        .collect()
}
