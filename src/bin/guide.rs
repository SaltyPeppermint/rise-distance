use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use egg::{Id, RecExpr};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;

use rise_distance::cli::argtypes::{SampleStrategy, SeedInput, TermSampleDist};
use rise_distance::cli::parquet::{dump_goal_summary_parquet, dump_to_parquet};
use rise_distance::cli::types::{GoalSummary, GuideEval, TrialsPerK};
use rise_distance::cli::{
    RULES, TRIAL_SIZE, get_run_folder, init_log, measure_guides, min_med_max,
    sample_frontier_terms, trial_avg,
};
use rise_distance::egg::math::{self, ConstantFold, Math, MathLabel};
use rise_distance::egg::{
    GuideGoalResult, ToEgg, convert, run_guide_goal, stop_reason_str, verify_reachability,
};
use rise_distance::{OriginTree, TreeShaped, tee_println};

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

    /// Number of top guides to print in summary table (default: 10)
    #[arg(long, default_value_t = 11)]
    top: usize,

    /// Sample Strategy
    #[arg(long)]
    eval_all: bool,

    /// Use the experimental `add_with_full_union` for the new egraph
    #[arg(long)]
    full_union: bool,
}

const MAX_TRIAL_SIZE: usize = const { TRIAL_SIZE[TRIAL_SIZE.len() - 1] };

#[allow(clippy::too_many_lines)]
fn main() {
    let cli = Cli::parse();
    let prefix = format!(
        "{}-{}-{}-fullunion-{}",
        cli.node_limit, cli.time_limit, cli.guide_sample_strategy, cli.full_union
    );
    let run_folder = get_run_folder(cli.output.as_deref(), "guide_eval", &prefix);
    init_log(&run_folder);

    // let verify_iters = cli.verify_iters.unwrap_or(cli.goal_iters);

    // Build the list of (seed_str, parsed_expr, max_size) to process.
    let seed_input = match (&cli.seed, &cli.seed_csv) {
        (Some(s), None) => SeedInput::Single {
            term: s.clone(),
            max_size: cli.max_size.expect("--max-size required with --seed"),
        },
        (None, Some(p)) => SeedInput::Csv(p.clone()),
        _ => panic!("clap group enforces exactly one of --seed / --seed-csv"),
    };

    let seeds = match seed_input {
        SeedInput::Single { term, max_size } => {
            let expr = term
                .parse::<RecExpr<Math>>()
                .unwrap_or_else(|e| panic!("Failed to parse seed: {e}"));
            vec![(term, expr, max_size)]
        }
        SeedInput::Csv(path) => {
            let mut rdr = csv::Reader::from_path(&path)
                .unwrap_or_else(|e| panic!("Failed to open CSV {}: {e}", path.display()));
            rdr.records()
                .map(|rec| {
                    let rec = rec.expect("CSV read error");
                    let max_size: usize = rec[0].parse().expect("CSV size column must be usize");
                    let term = rec[1].to_owned();
                    let expr = term
                        .parse::<RecExpr<Math>>()
                        .unwrap_or_else(|e| panic!("Failed to parse term '{term}': {e}"));
                    (term, expr, max_size * 2)
                })
                .collect()
        }
    };

    tee_println!("Distribution: {}", cli.size_distribution);
    tee_println!("Seeds to process: {}", seeds.len());

    let mut all_results: Vec<GoalResults> = Vec::new();
    let mut all_goal_stats: Vec<serde_json::Value> = Vec::new();

    for (seed_str, seed_expr, max_size) in &seeds {
        if let Some((results, stats)) =
            process_seed(&cli, seed_str, seed_expr, *max_size, &run_folder)
        {
            all_results.extend(results);
            all_goal_stats.push(stats);
        }
    }

    write_outputs(&run_folder, &all_results, &cli, &all_goal_stats);
}

/// Run eqsat for one seed, sample goals, evaluate each goal, and return the
/// collected results and per-goal stats. Returns `None` if the goal frontier
/// is empty (seed is skipped with a warning).
fn process_seed(
    cli: &Cli,
    seed_str: &str,
    seed_expr: &RecExpr<Math>,
    max_size: usize,
    run_folder: &Path,
) -> Option<(Vec<GoalResults>, serde_json::Value)> {
    tee_println!("\n=== Seed: {seed_str} (max_size={max_size}) ===");

    let result = run_guide_goal(
        seed_expr,
        RULES.get_or_init(math::rules),
        Duration::from_secs_f64(cli.time_limit),
        cli.node_limit,
    )?;

    tee_println!("Goal Iterations: {}", result.goal_iters());
    tee_println!("Guide Iterations: {}", result.guide_iters());
    tee_println!("Stop Reason: {}", stop_reason_str(result.stop_reason()));

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
    tee_println!(
        "Guide egraph had {guide_nodes} nodes, {guide_classes} classes in {guide_secs:.2}s"
    );
    tee_println!("Final egraph had {goal_nodes} nodes, {goal_classes} classes in {goal_secs:.2}s");

    tee_println!(
        "\nSampling goals from iteration-{} frontier...",
        result.goal_iters()
    );

    let root = result.root();
    let goals = sample_frontier_terms(
        &convert(result.goal(), root),
        result.prev_goal(),
        cli.goals,
        max_size,
        cli.size_distribution,
        cli.goal_sample_strategy,
    )?;
    if goals.is_empty() {
        tee_println!(
            "WARNING: Frontier empty for seed '{seed_str}'. Skipping. Try more iterations or a larger max-size."
        );
        return None;
    }
    tee_println!("Sampled {} goal(s)", goals.len());

    tee_println!(
        "\nGetting guides from iteration-{} frontier...",
        result.guide_iters()
    );

    let run_stats = json!({
        "seed": seed_str,
        "max_size": max_size,
        "guide_egraph_iters": result.goal_iters(),
        "guide_egraph_nodes": guide_nodes,
        "guide_egraph_classes": guide_classes,
        "guide_eqsat_time": guide_secs,
        "goal_egraph_iters": result.guide_iters(),
        "goal_egraph_nodes": goal_nodes,
        "goal_egraph_classes": goal_classes,
        "goal_eqsat_time": goal_secs,
    });

    let mut goal_results = Vec::new();
    let mut goal_stats = Vec::new();
    for goal in &goals {
        let (goal_result, goal_stat) =
            evaluate_goal(cli, &result, root, goal, seed_str, max_size, run_folder)?;
        goal_results.push(goal_result);
        goal_stats.push(goal_stat);
    }
    let full_stats = json!({"run_stats": run_stats, "goal_stats": goal_stats});
    Some((goal_results, full_stats))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_goal(
    cli: &Cli,
    result: &GuideGoalResult<Math, ConstantFold>,
    root: Id,
    goal: &OriginTree<MathLabel>,
    seed_str: &str,
    max_size: usize,
    run_folder: &Path,
) -> Option<(GoalResults, serde_json::Value)> {
    let goal_recexpr = goal.to_rec_expr();

    let sampled_guides = sample_frontier_terms(
        &convert(result.guide(), root),
        result.prev_guide(),
        cli.guides,
        max_size,
        cli.size_distribution,
        cli.guide_sample_strategy,
    )?;

    let runs = run_guide_set_trials(cli, &goal_recexpr, &sampled_guides);
    let goal_results = GoalResults {
        seed: seed_str.to_owned(),
        goal: goal.to_string(),
        runs,
    };

    let results = measure_guides(&sampled_guides, goal)
        .map(|measured_guide| {
            let iterations = verify_reachability(
                std::slice::from_ref(&measured_guide.guide),
                &goal_recexpr,
                RULES.get_or_init(math::rules),
                Duration::from_secs_f64(cli.time_limit),
                cli.node_limit,
                cli.full_union,
            );
            GuideEval {
                guide: measured_guide,
                iterations,
            }
        })
        .collect::<Vec<_>>();

    let stat = print_summary(&results, goal);
    if cli.eval_all {
        dump_to_parquet(run_folder, seed_str, goal, &results);
    }

    Some((goal_results, stat))
}

fn write_outputs(
    run_folder: &Path,
    all_results: &[GoalResults],
    cli: &Cli,
    stats: &[serde_json::Value],
) {
    let output_path = run_folder.join("top_k.json");
    let output_file = File::create(output_path).expect("Failed to create output json file");
    let mut output_writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut output_writer, &all_results).expect("write top-k json");

    let summaries = all_results
        .iter()
        .map(|r| GoalSummary::from_entries(&r.seed, &r.goal, &r.runs))
        .collect::<Vec<_>>();
    let summary_path = run_folder.join("top_k_summary.json");
    let summary_file = File::create(summary_path).expect("Failed to create summary json file");
    let summary_writer = BufWriter::new(summary_file);
    serde_json::to_writer(summary_writer, &summaries).expect("write top-k summary json");

    let parquet_path = run_folder.join("top_k_summary.parquet");
    dump_goal_summary_parquet(&parquet_path, &summaries);

    let config_path = run_folder.join("config.json");
    let config_file = File::create(config_path).expect("Failed to create output config.json file");
    let config_writer = BufWriter::new(config_file);
    serde_json::to_writer_pretty(config_writer, &cli).unwrap();

    let stats_path = run_folder.join("stats.json");
    let stats_file = File::create(&stats_path).expect("Failed to create stats.json");
    let stats_writer = BufWriter::new(stats_file);
    serde_json::to_writer_pretty(stats_writer, stats).expect("write stats json");
}

fn run_guide_set_trials(
    cli: &Cli,
    goal_recexpr: &RecExpr<Math>,
    sampled_guides: &[OriginTree<MathLabel>],
) -> TrialsPerK {
    assert!(sampled_guides.len() >= MAX_TRIAL_SIZE);
    TRIAL_SIZE
        .into_iter()
        .map(|k| {
            let trials = sampled_guides
                .par_chunks_exact(MAX_TRIAL_SIZE)
                .take(100)
                .map(|subset| {
                    verify_reachability(
                        &subset[..k],
                        goal_recexpr,
                        RULES.get_or_init(math::rules),
                        Duration::from_secs_f64(cli.time_limit),
                        cli.node_limit,
                        cli.full_union,
                    )
                })
                .collect::<Vec<_>>();

            let reached = trials.iter().filter(|v| v.is_some()).count();
            let combined_iters = trial_avg(trials.as_slice(), |t| Some(t.len()));
            let combined_nodes = trial_avg(&trials, |t| t.last().map(|i| i.egraph_nodes));
            let combined_time = trial_avg(&trials, |t| t.last().map(|i| i.total_time));
            if let (Some(avg_i), Some(avg_n), Some(avg_t)) =
                (combined_iters, combined_nodes, combined_time)
            {
                tee_println!(
                    "--- k = {k} guides ---\n\
                      Reached goal : {reached} / {}\n\
                      Avg iters    : {avg_i:.1}\n\
                      Avg nodes    : {avg_n:.0}\n\
                      Avg time     : {avg_t:.1}s",
                    trials.len(),
                );
            } else {
                tee_println!(
                    "--- k = {k} guides ---\n\
                      Reached goal : {reached} / {}\n\
                      Could NOT reach goal",
                    trials.len()
                );
            }
            (k, trials)
        })
        .collect()
}

#[derive(Serialize)]
struct GoalResults {
    seed: String,
    goal: String,
    runs: TrialsPerK,
}

#[expect(clippy::cast_precision_loss, clippy::shadow_unrelated)]
fn print_summary(
    results: &[GuideEval<MathLabel>],
    goal: &OriginTree<MathLabel>,
) -> serde_json::Value {
    let successful = results
        .iter()
        .filter(|r| r.iterations.is_some())
        .collect::<Vec<_>>();
    let total = results.len();
    let n_reached = successful.len();
    let goal_size = goal.size_without_types();
    let reach_pct = 100.0 * n_reached as f64 / total.max(1) as f64;

    tee_println!("\nSummary for goal: {goal}");
    tee_println!("  Goal size: {goal_size}");
    tee_println!("  Total guides evaluated: {total}");
    tee_println!("  Guides that reached goal: {n_reached}/{total} ({reach_pct:.1}%)");

    let mut stat = json!({
        "goal": goal.to_string(),
        "goal_size": goal_size,
        "total_guides": total,
        "reached": n_reached,
        "reach_pct": reach_pct,
    });

    if successful.is_empty() {
        return stat;
    }

    let (min, med, max) = min_med_max(&successful, |v| v.guide.zs_distance);
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Successful guide zs_dists:"
    );
    stat["zs_distance"] = json!({"min": min, "median": med, "max": max});

    let (min, med, max) = min_med_max(&successful, |v| v.guide.structural_distance);
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Successful guide struct_dist:"
    );
    stat["structural_distance"] = json!({"min": min, "median": med, "max": max});

    let (min, med, max) = min_med_max(&successful, |v| v.iterations.as_ref().unwrap().len());
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Iterations to reach:"
    );
    stat["iterations_to_reach"] = json!({"min": min, "median": med, "max": max});

    stat
}
