use std::fmt::Display;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use egg::RecExpr;
use num::BigUint;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use rise_distance::count::Counter;
use serde::Serialize;
use serde_json::json;

use rise_distance::cli::argtypes::{SampleStrategy, SeedInput, TermSampleDist};
use rise_distance::cli::parquet::dump_summary_parquet;
use rise_distance::cli::types::{GoalSummary, GuideError, TrialsPerK};
use rise_distance::cli::{PrecomputePackage, RULES, TRIAL_SIZE, get_run_folder, init_log};
use rise_distance::egg::math::{self, ConstantFold, Math, MathLabel};
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

    /// Number of top guides to print in summary table (default: 10)
    #[arg(long, default_value_t = 11)]
    top: usize,

    /// Use the experimental `add_with_full_union` for the new egraph
    #[arg(long)]
    full_union: bool,

    /// Measure the distance?
    #[arg(long)]
    measure: bool,
}

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
    let mut all_stats: Vec<serde_json::Value> = Vec::new();

    for (seed_str, seed_expr, max_size) in &seeds {
        if let Some((results, stats)) = process_seed(&cli, seed_str, seed_expr, *max_size) {
            all_stats.push(stats);
            all_results.extend(results);
        }
    }

    write_top_k_outputs(&run_folder, &all_results);
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
) -> Option<(Vec<GoalResults>, serde_json::Value)> {
    tee_println!("\n=== Seed: {seed_str} (max_size={max_size}) ===");

    let result = big_eqsat(
        seed_expr,
        RULES.get_or_init(math::rules),
        Duration::from_secs_f64(cli.time_limit),
        cli.node_limit,
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
    let Some(goals) = PrecomputePackage::<BigUint, MathLabel, _, _>::precompute(
        result.goal(),
        result.prev_goal().to_owned(),
        result.root(),
        max_size,
    )?
    .sample_frontier_terms(cli.goals, cli.size_distribution, cli.goal_sample_strategy) else {
        tee_println!(
            "WARNING: Not enough goals in the frontier for seed '{seed_str}'. Skipping. Try more iterations or a larger max-size."
        );
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

    let pc = PrecomputePackage::<BigUint, _, _, _>::precompute(
        result.guide(),
        result.prev_guide().clone(),
        result.root(),
        max_size,
    )?;

    tee_println!("\nRunning top_k experiments...");
    let goal_results = goals
        .iter()
        .map(|goal| {
            tee_println!("Current goal: {}", goal.to_string());
            GoalResults {
                seed: seed_str.to_owned(),
                goal: goal.to_string(),
                runs: run_guide_set_trials(cli, &goal.to_rec_expr(), &pc),
            }
        })
        .collect();
    // if cli.eval_all {
    //     tee_println!("\nRunning eval_all...");
    //     let big_stats = goals
    //         .iter()
    //         .map(|goal| {
    //             tee_println!("Current goal: {}", goal.to_string());
    //             eval_all_fn(
    //                 cli,
    //                 goal,
    //                 seed_str,
    //                 run_folder,
    //                 goal.to_rec_expr(),
    //                 &sampled_guides,
    //             )
    //         })
    //         .collect::<Vec<_>>();
    //     stats["all_eval_stats"] = big_stats.into();
    // }
    Some((goal_results, stats))
}

// fn eval_all_fn(
//     cli: &Cli,
//     goal: &OriginTree<MathLabel>,
//     seed_str: &str,
//     run_folder: &Path,
//     goal_recexpr: RecExpr<Math>,
//     sampled_guides: &[OriginTree<MathLabel>],
// ) -> serde_json::Value {
//     let pb_style = ProgressStyle::with_template(
//         "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}] ranking guides",
//     )
//     .unwrap();
//     let goal_flat = goal.flatten(false);
//     let results = sampled_guides
//         .into_par_iter()
//         .progress_with_style(pb_style)
//         .map(move |guide| {
//             let measurements = measure_guide(guide, &goal_flat);
//             let iterations = verify_reachability(
//                 std::slice::from_ref(guide).iter(),
//                 &goal_recexpr,
//                 RULES.get_or_init(math::rules),
//                 Duration::from_secs_f64(cli.time_limit),
//                 cli.node_limit,
//                 cli.full_union,
//             );
//             GuideEval {
//                 guide: guide.clone(),
//                 measurements,
//                 iterations,
//             }
//         })
//         .collect::<Vec<_>>();
//     let stat = print_summary(&results, goal);
//     dump_to_parquet(run_folder, seed_str, goal, &results);
//     stat
// }

fn write_top_k_outputs(run_folder: &Path, all_results: &[GoalResults]) {
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
    dump_summary_parquet(&parquet_path, &summaries);
}

fn write_config(run_folder: &Path, cli: &Cli) {
    let config_path = run_folder.join("config.json");
    let config_file = File::create(config_path).expect("Failed to create output config.json file");
    let config_writer = BufWriter::new(config_file);
    serde_json::to_writer_pretty(config_writer, &cli).unwrap();
}
fn write_stats(run_folder: &Path, stats: &[serde_json::Value]) {
    let stats_path = run_folder.join("stats.json");
    let stats_file = File::create(&stats_path).expect("Failed to create stats.json");
    let stats_writer = BufWriter::new(stats_file);
    serde_json::to_writer_pretty(stats_writer, stats).expect("write stats json");
}

fn run_guide_set_trials<C>(
    cli: &Cli,
    goal_recexpr: &RecExpr<Math>,
    pc: &PrecomputePackage<C, MathLabel, Math, ConstantFold>,
) -> TrialsPerK
where
    C: Counter + Display + Ord,
{
    // assert!(sampler.len() >= 10 * TRIAL_SIZE[TRIAL_SIZE.len() - 1]);
    TRIAL_SIZE
        .into_par_iter()
        .map(|k| {
            let outer_rng = ChaCha12Rng::seed_from_u64(k as u64);
            let trials = (0..100)
                .into_par_iter()
                .map(|s| {
                    let mut inner_rng = outer_rng.clone();
                    inner_rng.set_stream(s);
                    let subset = pc
                        .sample_frontier_terms(k, cli.size_distribution, cli.guide_sample_strategy)
                        .ok_or(GuideError::InsufficientSamples)?;
                    verify_reachability(
                        &subset,
                        goal_recexpr,
                        RULES.get_or_init(math::rules),
                        Duration::from_secs_f64(cli.time_limit),
                        cli.node_limit,
                        cli.full_union,
                    )
                })
                .collect::<Vec<_>>();

            // log_trials(k, &trials);
            (k, trials)
        })
        .collect()
}

// fn log_trials(k: usize, trials: &[Result<Vec<egg::Iteration<()>>, GuideError>]) {
//     let reached = trials.iter().filter(|v| v.is_some()).count();
//     let combined_iters = trial_avg(trials, |t| Ok(t.len()));
//     let combined_nodes = trial_avg(trials, |t| t.last().map(|i| i.egraph_nodes));
//     let combined_time = trial_avg(trials, |t| t.last().map(|i| i.total_time));
//     if let (Some(avg_i), Some(avg_n), Some(avg_t)) = (combined_iters, combined_nodes, combined_time)
//     {
//         tee_println!(
//             "--- k = {k} guides ---\n\
//                       Reached goal : {reached} / {}\n\
//                       Avg iters    : {avg_i:.1}\n\
//                       Avg nodes    : {avg_n:.0}\n\
//                       Avg time     : {avg_t:.1}s",
//             trials.len(),
//         );
//     } else {
//         tee_println!(
//             "--- k = {k} guides ---\n\
//                       Reached goal : {reached} / {}\n\
//                       Could NOT reach goal",
//             trials.len()
//         );
//     }
// }

#[derive(Serialize)]
struct GoalResults {
    seed: String,
    goal: String,
    runs: TrialsPerK,
}

// #[expect(clippy::cast_precision_loss, clippy::shadow_unrelated)]
// fn print_summary(
//     results: &[GuideEval<MathLabel>],
//     goal: &OriginTree<MathLabel>,
// ) -> serde_json::Value {
//     let successful = results
//         .iter()
//         .filter(|r| r.iterations.is_some())
//         .collect::<Vec<_>>();
//     let total = results.len();
//     let n_reached = successful.len();
//     let goal_size = goal.size_without_types();
//     let reach_pct = 100.0 * n_reached as f64 / total.max(1) as f64;

//     tee_println!("\nSummary for goal: {goal}");
//     tee_println!("  Goal size: {goal_size}");
//     tee_println!("  Total guides evaluated: {total}");
//     tee_println!("  Guides that reached goal: {n_reached}/{total} ({reach_pct:.1}%)");

//     let mut stat = json!({
//         "goal": goal.to_string(),
//         "goal_size": goal_size,
//         "total_guides": total,
//         "reached": n_reached,
//         "reach_pct": reach_pct,
//     });
//     if successful.is_empty() {
//         return stat;
//     }

//     let (min, med, max) = min_med_max(&successful, |r| r.measurements.zs_distance);
//     tee_println!(
//         "  {:<30} min={min}, median={med}, max={max}",
//         "Successful guide zs_dists:"
//     );
//     stat["zs_distance"] = json!({"min": min, "median": med, "max": max});

//     let (min, med, max) = min_med_max(&successful, |r| r.measurements.structural_distance);
//     tee_println!(
//         "  {:<30} min={min}, median={med}, max={max}",
//         "Successful guide struct_dist:"
//     );
//     stat["structural_distance"] = json!({"min": min, "median": med, "max": max});

//     let (min, med, max) = min_med_max(&successful, |v| v.iterations.as_ref().unwrap().len());
//     tee_println!(
//         "  {:<30} min={min}, median={med}, max={max}",
//         "Iterations to reach:"
//     );
//     stat["iterations_to_reach"] = json!({"min": min, "median": med, "max": max});

//     stat
// }
