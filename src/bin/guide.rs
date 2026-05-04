use std::fmt::Display;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use egg::RecExpr;
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::BigUint;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;

use rise_distance::cli::argparse::{SampleStrategy, SeedInput, TermSampleDist, parse_seeds};
use rise_distance::cli::parquet::dump_summary_parquet;
use rise_distance::cli::types::{GoalSummary, TrialsPerK};
use rise_distance::cli::{PrecomputePackage, get_run_folder, init_log};
use rise_distance::cli::{write_config, write_stats};
use rise_distance::count::Counter;
use rise_distance::egg::math::{ConstantFold, Math, RULES};
use rise_distance::egg::{ToEgg, big_eqsat, verify_reachability};
use rise_distance::tee_println;

#[derive(Parser, Serialize)]
#[command(
    about = "Evaluate distance metrics as guide predictors for equality saturation",
    after_help = "\
Examples:
  # Basic evaluation with 100 random guides, single seed
  guide --seed '(d x (+ (* x x) 1))' --max-size 150 -g 100

  # Sample 5 goals and 100 guides
  guide --seed '(d x (+ (* x x) 1))' --max-size 150 -g 100 --goals 5

  # Loop over many seeds from a CSV (size,term columns; size drives max-size)
  guide --seed-csv seeds.csv -g 100

  # Write JSON output to a folder
  guide --seed '(d x (+ (* x x) 1))' --max-size 150 -g 100 -o results/

  # Use the experimental full-union egraph
  guide --seed '(d x (+ (* x x) 1))' --max-size 150 -g 100 --full-union

  # Print top 20 guides in the summary table (default: 10)
  guide --seed '(d x (+ (* x x) 1))' --max-size 150 -g 100 --top 20
"
)]
struct Cli {
    /// Seed term as an s-expression (Math language). Requires --max-size.
    #[arg(long, group = "seed_input", requires = "max_size")]
    seed: Option<String>,

    /// Path to a CSV file with `size,term` columns. The `size` column drives --max-size.
    /// Mutually exclusive with --seed / --max-size.
    #[arg(long, group = "seed_input", conflicts_with = "max_size")]
    seed_csv: Option<PathBuf>,

    /// Node limit for the baseline egraph in seconds
    #[arg(long, default_value_t = 1_000_000_000)]
    node_limit: usize,

    /// Time limit for the baseline egraph in seconds
    #[arg(long, default_value_t = 0.2)]
    time_limit: f64,

    /// Number of guide candidates to sample from the n-1 frontier
    #[arg(short, long)]
    guides: usize,

    /// Number of goal terms to sample from the n frontier
    #[arg(long, default_value_t = 1)]
    goals: usize,

    /// Max term size for counting/sampling. Required with --seed; forbidden with --seed-csv.
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

const TRIAL_SIZE: [usize; 6] = [1, 2, 5, 10, 50, 100];
const NUM_TRIALS: usize = 100;

fn main() {
    let cli = Cli::parse();
    let prefix = format!(
        "nodes-{}-timems-{}-strategy-{}-fullunion-{}",
        cli.node_limit,
        Duration::from_secs_f64(cli.time_limit).as_millis(),
        cli.guide_sample_strategy,
        cli.full_union
    );
    let run_folder = get_run_folder(cli.output.as_deref(), "guide_eval", &prefix);
    init_log(&run_folder);
    tee_println!("Run Folder: {}", run_folder.to_string_lossy());

    let seed_input = match (&cli.seed, &cli.seed_csv) {
        (Some(s), None) => SeedInput::Single {
            term: s.clone(),
            max_size: cli.max_size.expect("--max-size required with --seed"),
        },
        (None, Some(p)) => SeedInput::Csv(p.clone()),
        _ => panic!("clap group enforces exactly one of --seed / --seed-csv"),
    };
    let seeds = parse_seeds(seed_input);

    tee_println!("Distribution: {}", cli.size_distribution);
    tee_println!("Seeds to process: {}", seeds.len());

    let mut all_with_replacement = Vec::new();
    let mut all_no_replacement = Vec::new();
    let mut all_stats = Vec::new();

    for (i, (seed_str, seed_expr, max_size)) in seeds.iter().enumerate() {
        tee_println!("\n=== Seed {i}: {seed_str} (max_size={max_size}) ===");
        if let Some((with_replacement, no_replacement, stats)) =
            process_seed(&cli, seed_str, seed_expr, *max_size)
        {
            all_stats.push(stats);
            all_with_replacement.extend(with_replacement);
            all_no_replacement.extend(no_replacement);
        }
    }

    write_top_k_outputs(&run_folder, &all_with_replacement, "with_replacement");
    write_top_k_outputs(&run_folder, &all_no_replacement, "no_replacement");
    write_config(&run_folder, &cli);
    write_stats(&run_folder, &all_stats);
}

/// Run eqsat for one seed, sample goals, evaluate each goal, and return the
/// collected results and per-goal stats. Returns `None` if the goal frontier
/// is empty (seed is skipped with a warning).
fn process_seed(
    cli: &Cli,
    seed_str: &str,
    seed_expr: &RecExpr<Math>,
    max_size: usize,
) -> Option<(Vec<GoalResults>, Vec<GoalResults>, serde_json::Value)> {
    let result = big_eqsat(
        seed_expr,
        RULES.iter(),
        Duration::from_secs_f64(cli.time_limit),
        cli.node_limit,
        cli.backoff_scheduler,
    )?;
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
        cli.goals,
        cli.size_distribution,
        cli.goal_sample_strategy,
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
    let no_replacement = goals
        .iter()
        .map(|goal| {
            tee_println!("Current goal: {}", goal.to_string());
            GoalResults {
                seed: seed_str.to_owned(),
                goal: goal.to_string(),
                runs: run_guide_set_trials_no_replacement(cli, &goal.to_rec_expr(), &pc),
            }
        })
        .collect();

    tee_println!("\nRunning top_k experiments WITH REPLACEMENT...");
    let with_replacement = goals
        .iter()
        .map(|goal| {
            tee_println!("Current goal: {}", goal.to_string());
            GoalResults {
                seed: seed_str.to_owned(),
                goal: goal.to_string(),
                runs: run_guide_set_trials_with_replacement(cli, &goal.to_rec_expr(), &pc),
            }
        })
        .collect();

    Some((with_replacement, no_replacement, stats))
}

fn write_top_k_outputs(run_folder: &Path, results: &[GoalResults], suffix: &str) {
    let output_path = run_folder.join(format!("{suffix}_top_k.json"));
    let output_file = File::create(output_path).expect("Failed to create output json file");
    let mut output_writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut output_writer, &results).expect("write top-k json");

    let summaries = results
        .iter()
        .map(|r| GoalSummary::from_entries(&r.seed, &r.goal, &r.runs))
        .collect::<Vec<_>>();
    let summary_path = run_folder.join(format!("{suffix}_top_k_summary.json"));
    let summary_file = File::create(summary_path).expect("Failed to create summary json file");
    let summary_writer = BufWriter::new(summary_file);
    serde_json::to_writer(summary_writer, &summaries).expect("write top-k summary json");

    let parquet_path = run_folder.join(format!("{suffix}_top_k_summary.parquet"));
    dump_summary_parquet(&parquet_path, &summaries);
}

fn run_guide_set_trials_with_replacement<C>(
    cli: &Cli,
    goal_recexpr: &RecExpr<Math>,
    pc: &PrecomputePackage<C, Math, ConstantFold>,
) -> TrialsPerK
where
    C: Counter + Display + Ord,
{
    let bars = progress_bars();
    bars.into_par_iter()
        .map(|(k, pb)| {
            let trials = (0..NUM_TRIALS)
                .into_par_iter()
                .map(|s| {
                    let subset = pc.sample_frontier_terms::<false>(
                        k,
                        cli.size_distribution,
                        cli.guide_sample_strategy,
                        [k as u64, s as u64],
                    )?;
                    verify_reachability(
                        &subset,
                        goal_recexpr,
                        &RULES,
                        Duration::from_secs_f64(cli.time_limit),
                        cli.node_limit,
                        cli.full_union,
                        cli.backoff_scheduler,
                    )
                    .map_err(|e| e.into())
                })
                .progress_with(pb)
                .collect::<Vec<_>>();

            // log_trials(k, &trials);
            (k, trials)
        })
        .collect()
}

fn run_guide_set_trials_no_replacement<C>(
    cli: &Cli,
    goal_recexpr: &RecExpr<Math>,
    pc: &PrecomputePackage<C, Math, ConstantFold>,
) -> TrialsPerK
where
    C: Counter + Display + Ord,
{
    let bars = progress_bars();
    bars.into_par_iter()
        .map(|(k, pb)| {
            let samples = pc.sample_frontier_terms::<true>(
                k * NUM_TRIALS,
                cli.size_distribution,
                cli.guide_sample_strategy,
                [k as u64, 0],
            );
            let trials = match samples {
                Err(e) => vec![Err(e); NUM_TRIALS],
                Ok(samples) => samples
                    .par_chunks(k)
                    .map(|subset| {
                        verify_reachability(
                            subset,
                            goal_recexpr,
                            &RULES,
                            Duration::from_secs_f64(cli.time_limit),
                            cli.node_limit,
                            cli.full_union,
                            cli.backoff_scheduler,
                        )
                        .map_err(Into::into)
                    })
                    .progress_with(pb)
                    .collect(),
            };
            (k, trials)
        })
        .collect()
}

fn progress_bars() -> Vec<(usize, ProgressBar)> {
    let mp = MultiProgress::new();
    let style = ProgressStyle::with_template("{msg:>6} [{bar:40}] {pos}/{len}")
        .unwrap()
        .progress_chars("=> ");

    TRIAL_SIZE
        .iter()
        .map(|&k| {
            let pb = mp.add(ProgressBar::new(NUM_TRIALS as u64).with_style(style.clone()));
            pb.set_message(format!("k={k}"));
            (k, pb)
        })
        .collect::<Vec<_>>()
}

#[derive(Serialize)]
struct GoalResults {
    seed: String,
    goal: String,
    runs: TrialsPerK,
}
