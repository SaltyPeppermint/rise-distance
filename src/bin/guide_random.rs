use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use clap::Parser;
use egg::RecExpr;
use hashbrown::HashSet;
use rayon::prelude::*;
use rise_distance::cli::parquet::{dump_to_parquet, dump_top_k_summary_parquet};
use serde::Serialize;
use serde_json::json;

use rise_distance::cli::{
    EvalResult, RULES, RandomEntry, SizeDistribution, TRIAL_SIZE, TopKSummary, get_run_folder,
    init_log, measure_guides, min_med_max, sample_frontier_terms, trial_avg,
};
use rise_distance::egg::math::{self, Math, MathLabel};
use rise_distance::egg::{ToEgg, convert, run_guide_goal, verify_reachability};
use rise_distance::{TreeNodeWithOrigin, TreeShaped, tee_println};

#[derive(Parser, Serialize)]
#[command(
    about = "Evaluate distance metrics as guide predictors for equality saturation",
    after_help = "\
Examples:
  # Basic evaluation with 100 random guides
  guide-random -s '(d x (+ (* x x) 1))' -n 5 -i 4 -g 100 --max-size 150

  # Sample 5 goals and 100 guides
  guide-random -s '(d x (+ (* x x) 1))' -n 8 -i 3 -g 100 --goals 5 --max-size 150

  # Write JSON output to a file
  guide-random -s '(d x (+ (* x x) 1))' -n 5 -i 4 -g 100 --max-size 150 -o results.json

  # Limit verification iterations separately from goal iterations
  guide-random -s '(d x (+ (* x x) 1))' -n 8 -i 4 -g 100 --max-size 150 --verify-iters 5

  # Print top 20 guides in the summary table (default: 10)
  guide-random -s '(d x (+ (* x x) 1))' -n 5 -i 4 -g 100 --max-size 150 --top 20
"
)]
struct Cli {
    /// Seed term as an s-expression (Math language)
    #[arg(short, long)]
    seed: String,

    /// Number of eqsat iterations to grow the egraph
    #[arg(short = 'n', long)]
    goal_iters: usize,

    /// Number of eqsat iterations to grow the egraph to reach the guide
    #[arg(short = 'i', long)]
    guide_iters: usize,

    /// Number of guide candidates to sample from the n-1 frontier
    #[arg(short, long)]
    guides: usize,

    /// Number of goal terms to sample from the n frontier
    #[arg(long, default_value_t = 1)]
    goals: usize,

    /// Max term size for counting/sampling
    #[arg(long)]
    max_size: usize,

    /// How to distribute the sample budget across sizes.
    /// Options: uniform, proportional:<`min_per_size`>, normal:<sigma>
    #[arg(long, default_value_t = SizeDistribution::Uniform)]
    size_distribution: SizeDistribution,

    /// Output folder (generated if omitted)
    #[arg(short, long)]
    output: Option<String>,

    /// Max iterations when verifying guide reachability (defaults to -n value)
    #[arg(long)]
    verify_iters: Option<usize>,

    /// Number of top guides to print in summary table (default: 10)
    #[arg(long, default_value_t = 10)]
    top: usize,

    /// Sample Strategy
    #[arg(long)]
    eval_all: bool,

    /// Use the experimental `add_with_full_union` for the new egraph
    #[arg(long)]
    full_union: bool,
}

const MAX_TRIAL_SIZE: usize = const { TRIAL_SIZE[TRIAL_SIZE.len() - 1] };

fn main() {
    let cli = Cli::parse();
    let prefix = format!(
        "run-{}-{}-random-fullunion-{}",
        cli.guide_iters, cli.goal_iters, cli.full_union
    );
    let run_folder = get_run_folder(cli.output.as_deref(), "guide_eval", &prefix);
    init_log(&run_folder);

    let verify_iters = cli.verify_iters.unwrap_or(cli.goal_iters);

    let seed = cli
        .seed
        .parse::<RecExpr<Math>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed: {e}"));

    tee_println!("Seed: {seed}");
    tee_println!("Goal Iterations: {}", cli.goal_iters);
    tee_println!("Guide Iterations: {}", cli.guide_iters);
    tee_println!("Distribution: {}", cli.size_distribution);

    tee_println!("Running eqsat for {} iterations...", cli.goal_iters);
    let start = Instant::now();
    let result = run_guide_goal(
        &seed,
        RULES.get_or_init(math::rules),
        cli.guide_iters,
        cli.goal_iters,
    );

    let eqsat_secs = start.elapsed().as_secs_f64();
    tee_println!("Eqsat completed in {eqsat_secs:.2}s");
    let guide_nodes = result.guide().total_number_of_nodes();
    let guide_classes = result.guide().classes().len();
    let goal_nodes = result.goal().total_number_of_nodes();
    let goal_classes = result.goal().classes().len();
    tee_println!("Guide egraph had {guide_nodes} nodes, {guide_classes} classes");
    tee_println!("Final egraph had {goal_nodes} nodes, {goal_classes} classes");

    let mut stats = json!({
        "eqsat_time_secs": eqsat_secs,
        "guide_egraph_nodes": guide_nodes,
        "guide_egraph_classes": guide_classes,
        "goal_egraph_nodes": goal_nodes,
        "goal_egraph_classes": goal_classes,
    });

    tee_println!(
        "\nSampling goals from iteration-{} frontier...",
        cli.goal_iters
    );

    let root = result.root();
    let goals = sample_frontier_terms(
        &convert(result.goal(), root),
        result.prev_goal(),
        cli.goals,
        cli.max_size,
        cli.size_distribution,
    );
    assert!(
        !goals.is_empty(),
        "Frontier empty. Try more iterations or a larger max-size.",
    );
    tee_println!("Sampled {} goal(s)", goals.len());

    tee_println!(
        "\nGetting guides from iteration-{} frontier...",
        cli.guide_iters
    );

    let mut all_top_k = Vec::new();
    let mut goal_stats = Vec::new();
    for goal in &goals {
        let goal_recexpr = goal.to_rec_expr();

        let sampled_guides = sample_frontier_terms(
            &convert(result.guide(), root),
            result.prev_guide(),
            cli.guides,
            cli.max_size,
            cli.size_distribution,
        );

        let entries = take_n_trials(&cli, cli.guides, &goal_recexpr, &sampled_guides);
        all_top_k.push(TopKResults {
            goal: goal.to_string(),
            entries,
        });
        let measured = measure_guides(&sampled_guides, goal)
            .into_iter()
            .collect::<HashSet<_>>();
        let results = measured
            .par_iter()
            .map(|measured| EvalResult {
                guide: measured,
                iterations: verify_reachability(
                    std::slice::from_ref(&measured.guide),
                    &goal_recexpr,
                    RULES.get_or_init(math::rules),
                    verify_iters,
                    cli.full_union,
                ),
            })
            .collect::<Vec<_>>();
        goal_stats.push(print_summary(&results, goal, verify_iters));
        if cli.eval_all {
            dump_to_parquet(&run_folder, goal, &results);
        }
    }

    stats["goals"] = serde_json::Value::Array(goal_stats);
    write_outputs(&run_folder, &all_top_k, &cli, &stats);
}

fn write_outputs(
    run_folder: &std::path::Path,
    all_top_k: &[TopKResults],
    cli: &Cli,
    stats: &serde_json::Value,
) {
    let output_path = run_folder.join("top_k.json");
    let output_file = File::create(output_path).expect("Failed to create output json file");
    let mut output_writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut output_writer, &all_top_k).expect("write top-k json");

    let summaries: Vec<TopKSummary> = all_top_k
        .iter()
        .map(|tk| TopKSummary::from_entries(&tk.goal, &tk.entries))
        .collect();
    let summary_path = run_folder.join("top_k_summary.json");
    let summary_file = File::create(summary_path).expect("Failed to create summary json file");
    let summary_writer = BufWriter::new(summary_file);
    serde_json::to_writer(summary_writer, &summaries).expect("write top-k summary json");

    let parquet_path = run_folder.join("top_k_summary.parquet");
    dump_top_k_summary_parquet(&parquet_path, &summaries);

    let config_path = run_folder.join("config.json");
    let config_file = File::create(config_path).expect("Failed to create output config.json file");
    let config_writer = BufWriter::new(config_file);
    serde_json::to_writer_pretty(config_writer, &cli).unwrap();

    let stats_path = run_folder.join("stats.json");
    let stats_file = File::create(&stats_path).expect("Failed to create stats.json");
    let stats_writer = BufWriter::new(stats_file);
    serde_json::to_writer_pretty(stats_writer, stats).expect("write stats json");
}

fn take_n_trials(
    cli: &Cli,
    guide_count: usize,
    goal_recexpr: &RecExpr<Math>,
    sampled_guides: &[TreeNodeWithOrigin<MathLabel>],
) -> Vec<RandomEntry> {
    let mut entries = Vec::new();
    for k in TRIAL_SIZE {
        let trials = sampled_guides
            .par_windows(guide_count / MAX_TRIAL_SIZE)
            .map(|guides_here| {
                verify_reachability(
                    &guides_here[..k],
                    goal_recexpr,
                    RULES.get_or_init(math::rules),
                    cli.goal_iters,
                    cli.full_union,
                )
            })
            .collect::<Vec<_>>();

        let reached = trials.iter().filter(|v| v.is_some()).count();
        tee_println!("{reached} out of {} reached the goal", trials.len());
        let combined_iters = trial_avg(trials.as_slice(), |t| Some(t.len()));
        let combined_nodes = trial_avg(&trials, |t| t.last().map(|i| i.egraph_nodes));
        if let (Some(avg_i), Some(avg_n)) = (combined_iters, combined_nodes) {
            tee_println!(
                "Could reach with {k} guides: {avg_i:.1} ({avg_n:.0} nodes) (avg over {reached} in {MAX_TRIAL_SIZE} trials)"
            );
        } else {
            tee_println!("Could NOT reach with {k} guides");
        }

        entries.push(RandomEntry { k, trials });
    }
    entries
}

#[derive(Serialize)]
struct TopKResults {
    goal: String,
    entries: Vec<RandomEntry>,
}

#[expect(clippy::cast_precision_loss, clippy::shadow_unrelated)]
fn print_summary(
    results: &[EvalResult<MathLabel>],
    goal: &TreeNodeWithOrigin<MathLabel>,
    max_iters: usize,
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
    tee_println!("  Max verify iterations: {max_iters}");
    tee_println!("  Guides that reached goal: {n_reached}/{total} ({reach_pct:.1}%)");

    let mut stat = json!({
        "goal": goal.to_string(),
        "goal_size": goal_size,
        "total_guides": total,
        "max_verify_iters": max_iters,
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
