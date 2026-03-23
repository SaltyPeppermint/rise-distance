use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use clap::Parser;
use egg::RecExpr;
use hashbrown::HashSet;
use rayon::prelude::*;
use serde::Serialize;

use rise_distance::TreeNode;
use rise_distance::cli::{
    EvalResult, N_RANDOM, RULES, RandomEntry, SizeDistribution, dump_to_parquet, get_run_folder,
    init_log, measure_guides, min_med_max, sample_frontier_terms, trial_avg,
};
use rise_distance::egg::math::{self, Math, MathLabel};
use rise_distance::egg::{convert, run_guide_goal, verify_reachability};
use rise_distance::tee_println;

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
}

const N_TRIALS: usize = const { N_RANDOM[N_RANDOM.len() - 1] };

fn main() {
    let cli = Cli::parse();
    let prefix = format!("run-{}-{}-random", cli.guide_iters, cli.goal_iters);
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

    tee_println!(
        "Running equality saturation for {} iterations...",
        cli.goal_iters
    );
    let start = Instant::now();
    let result = run_guide_goal(
        &seed,
        RULES.get_or_init(math::rules),
        cli.guide_iters,
        cli.goal_iters,
    );

    tee_println!("Eqsat completed in {:.2?}", start.elapsed());
    tee_println!(
        "Guide egraph had {} nodes",
        result.guide().total_number_of_nodes()
    );
    tee_println!(
        "Final egraph had {} nodes",
        result.goal().total_number_of_nodes()
    );

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
        "No frontier terms found. Try more iterations or a larger max-size.",
    );
    tee_println!("Sampled {} goal(s)", goals.len());

    tee_println!(
        "\nGetting guides from iteration-{} frontier...",
        cli.guide_iters
    );

    let mut all_top_k = Vec::new();
    for goal in &goals {
        let goal_recexpr = goal.into();

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
            .map(|measured| {
                let values = verify_reachability(
                    std::slice::from_ref(&measured.guide),
                    &goal_recexpr,
                    RULES.get_or_init(math::rules),
                    verify_iters,
                );
                EvalResult {
                    guide: measured,
                    iterations: values,
                }
            })
            .collect::<Vec<_>>();
        print_summary(&results, goal, verify_iters);
        if cli.eval_all {
            dump_to_parquet(&run_folder, goal, &results);
        }
    }

    let output_path = run_folder.join("top_k.json");
    let output_file = File::create(output_path).expect("Failed to create output json file");
    let mut output_writer = BufWriter::new(output_file);
    serde_json::to_writer(&mut output_writer, &all_top_k).expect("write top-k json");

    let config_path = run_folder.join("config.json");
    let config_file = File::create(config_path).expect("Failed to create output config.json file");
    let config_writer = BufWriter::new(config_file);
    serde_json::to_writer_pretty(config_writer, &cli).unwrap();
}

fn take_n_trials(
    cli: &Cli,
    guide_count: usize,
    goal_recexpr: &RecExpr<Math>,
    sampled_guides: &[TreeNode<MathLabel>],
) -> Vec<RandomEntry> {
    let mut entries = Vec::new();
    for k in N_RANDOM {
        let trials = sampled_guides
            .par_windows(guide_count / N_TRIALS)
            .map(|guides_here| {
                let subset = &guides_here[..k];
                verify_reachability(
                    subset,
                    goal_recexpr,
                    RULES.get_or_init(math::rules),
                    cli.goal_iters,
                )
            })
            .collect::<Vec<_>>();

        let reached = trials.iter().filter(|v| v.is_some()).count();
        tee_println!("{reached} out of {} reached the goal", trials.len());
        let combined_iters = trial_avg(trials.as_slice(), |t| Some(t.len()));
        let combined_nodes = trial_avg(&trials, |t| t.last().map(|i| i.egraph_nodes));
        if let (Some(avg_i), Some(avg_n)) = (combined_iters, combined_nodes) {
            tee_println!(
                "Could reach with {k} guides: {avg_i:.1} ({avg_n:.0} nodes) (avg over {reached} in {N_TRIALS} trials)"
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
fn print_summary(results: &[EvalResult<MathLabel>], goal: &TreeNode<MathLabel>, max_iters: usize) {
    let successful = results
        .iter()
        .filter(|r| r.iterations.is_some())
        .collect::<Vec<_>>();
    let total = results.len();
    let n_reached = successful.len();

    tee_println!("\nSummary for goal: {goal}");
    tee_println!("  Goal size: {}", goal.size_without_types());
    tee_println!("  Total guides evaluated: {total}");
    tee_println!("  Max verify iterations: {max_iters}");
    tee_println!(
        "  Guides that reached goal: {n_reached}/{total} ({:.1}%)",
        100.0 * n_reached as f64 / total.max(1) as f64
    );

    if successful.is_empty() {
        return;
    }

    let (min, med, max) = min_med_max(&successful, |v| v.guide.zs_distance);
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Successful guide zs_dists:"
    );

    let (min, med, max) = min_med_max(&successful, |v| v.iterations.as_ref().unwrap().len());
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Iterations to reach:"
    );

    let (min, med, max) = min_med_max(&successful, |v| v.guide.structural_distance);
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Successful guide struct_dist:"
    );

    let (min, med, max) = min_med_max(&successful, |v| v.iterations.as_ref().unwrap().len());
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Iterations to reach:"
    );
}
