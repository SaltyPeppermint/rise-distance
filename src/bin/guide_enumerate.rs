use std::fs::File;
use std::path::Path;
use std::time::Instant;

use clap::Parser;
use egg::RecExpr;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::Serialize;

use rise_distance::TreeNode;
use rise_distance::cli::{
    EvalResult, MeasuredGuide, N_RANDOM, RULES, RandomEntry, RandomTrial, SampleDistribution,
    dump_to_csv, enumerate_frontier_terms, get_run_folder, init_log, measure_guides, min_med_max,
    sample_frontier_terms, trial_avg,
};
use rise_distance::egg::math::{self, Math, MathLabel};
use rise_distance::egg::{VerifyResult, convert, run_guide_goal, verify_reachability};
use rise_distance::tee_println;

#[derive(Parser, Serialize)]
#[command(
    about = "Evaluate distance metrics as guide predictors for equality saturation",
    after_help = "\
Examples:
  # Enumerate all guides up to size 150
  guide-enumerate -s '(d x (+ (* x x) 1))' -n 5 -i 4 --max-size 150

  # Write JSON output to a file
  guide-enumerate -s '(d x (+ (* x x) 1))' -n 5 -i 4 --max-size 150 -o results.json

  # Limit verification iterations separately from goal iterations
  guide-enumerate -s '(d x (+ (* x x) 1))' -n 8 -i 4 --max-size 150 --verify-iters 5

  # Evaluate all guides and write CSV, print top 20 in summary
  guide-enumerate -s '(d x (+ (* x x) 1))' -n 5 -i 4 --max-size 150 --eval-all --top 20
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

    /// Number of goal terms to sample from the n frontier
    #[arg(long, default_value_t = 1)]
    goals: usize,

    /// Max term size for counting/sampling
    #[arg(long)]
    max_size: usize,

    /// How to distribute the sample budget across sizes for goals.
    /// Options: uniform, proportional:<`min_per_size`>, normal:<sigma>
    #[arg(long, default_value_t = SampleDistribution::Uniform)]
    distribution: SampleDistribution,

    /// Output folder (generated if omitted)
    #[arg(short, long)]
    output: Option<String>,

    /// Max iterations when verifying guide reachability (defaults to -n value)
    #[arg(long)]
    verify_iters: Option<usize>,

    /// Number of top guides to print in summary table (default: 10)
    #[arg(long, default_value_t = 10)]
    top: usize,

    #[arg(long)]
    eval_all: bool,

    /// How often to evaluate the trial
    #[arg(long, default_value_t = 100)]
    n_trials: usize,
}

fn main() {
    let cli = Cli::parse();
    let prefix = format!("run-{}-{}-enumerate", cli.guide_iters, cli.goal_iters);
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
    tee_println!("Distribution: {}", cli.distribution);

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
        cli.distribution,
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

    let guides = enumerate_frontier_terms(
        &convert(result.guide(), root),
        result.prev_guide(),
        cli.max_size,
    );
    assert!(!guides.is_empty(), "No frontier terms found");
    tee_println!("Found {} guide(s)", guides.len());

    let mut all_top_k = Vec::new();
    let n_goals = goals.len();
    for (goal_idx, goal) in goals.iter().enumerate() {
        tee_println!(
            "\n=== Goal {}/{} (size {})===",
            goal_idx + 1,
            n_goals,
            goal.size_without_types()
        );
        tee_println!("{goal}\n");

        let mut ranked = measure_guides(&guides, goal);

        let timer = Instant::now();

        if cli.eval_all {
            eval_all(verify_iters, &ranked, goal, &run_folder);
        }
        tee_println!("Verification completed in {:.2?}", timer.elapsed());

        let top_k = eval_top_k(&mut ranked, goal, verify_iters, cli.n_trials);
        all_top_k.push(top_k);
    }
    let json_output =
        File::create(run_folder.join("top_k.json")).expect("Failed to create JSON output file");
    serde_json::to_writer_pretty(json_output, &all_top_k).expect("write top-k JSON");
}

fn eval_all(
    verify_iters: usize,
    ranked: &[MeasuredGuide<MathLabel>],
    goal: &TreeNode<MathLabel>,
    run_folder: &Path,
) {
    let goal_recexpr = goal.into();
    let pb_style = ProgressStyle::with_template(
        "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}] verifying guides",
    )
    .unwrap();
    tee_println!(
        "Verifying {} guides (max {verify_iters} iters each)...",
        ranked.len(),
    );
    let results = ranked
        .par_iter()
        .progress_with_style(pb_style)
        .map(|guide| EvalResult {
            guide,
            values: verify_reachability(
                std::slice::from_ref(&guide.guide),
                &goal_recexpr,
                RULES.get_or_init(math::rules),
                verify_iters,
            ),
        })
        .collect::<Vec<_>>();
    print_summary(&results, goal, verify_iters);
    dump_to_csv(run_folder, goal, &results);
}

#[derive(Serialize)]
struct TopKEntry {
    k: usize,
    best_single_iters: Option<usize>,
    combined_iters: Option<usize>,
    combined_nodes: Option<usize>,
}

#[derive(Serialize)]
struct TopKResults {
    goal: String,
    zs: Vec<TopKEntry>,
    structural: Vec<TopKEntry>,
    random: Vec<RandomEntry>,
    random_with_best_zs: Vec<RandomEntry>,
    random_with_best_structural: Vec<RandomEntry>,
}

fn eval_top_k(
    results: &mut [MeasuredGuide<MathLabel>],
    goal: &TreeNode<MathLabel>,
    max_iters: usize,
    n_trials: usize,
) -> TopKResults {
    tee_println!("Testing out top-k to see if that improves things");
    let go = goal.into();

    let mut top_k_results = TopKResults {
        goal: goal.to_string(),
        zs: Vec::new(),
        structural: Vec::new(),
        random: Vec::new(),
        random_with_best_zs: Vec::new(),
        random_with_best_structural: Vec::new(),
    };

    if results.is_empty() {
        return top_k_results;
    }

    results.sort_unstable_by_key(|v| v.zs_distance);
    tee_println!("ZS DISTANCE:");
    top_k_results.zs = top_k(max_iters, results, &go);

    results.sort_unstable_by_key(|v| v.structural_distance);
    tee_println!("STRUCTURAL DISTANCE:");
    top_k_results.structural = top_k(max_iters, results, &go);

    tee_println!("RANDOM ({n_trials} trials averaged):");
    top_k_results.random = random_k(max_iters, results, &go, n_trials);

    top_k_results
}

fn top_k(
    max_iters: usize,
    ranked: &[MeasuredGuide<MathLabel>],
    go: &RecExpr<Math>,
) -> Vec<TopKEntry> {
    let mut entries = Vec::new();
    for k in N_RANDOM {
        if k > ranked.len() {
            continue;
        }
        let top_guides = ranked[0..k]
            .iter()
            .map(|k| k.guide.clone())
            .collect::<Vec<_>>();
        let could_reach =
            verify_reachability(&top_guides, go, RULES.get_or_init(math::rules), max_iters);
        let best_in_k = ranked[0..k]
            .par_iter()
            .filter_map(|v| {
                verify_reachability(
                    std::slice::from_ref(&v.guide),
                    go,
                    RULES.get_or_init(math::rules),
                    max_iters,
                )
            })
            .map(|a| a.iters)
            .min();
        if let Some(i) = best_in_k {
            tee_println!("Best single guide in top {k} guides: {i}");
        } else {
            tee_println!("No single guide in top {k} could reach it");
        }
        if let Some(VerifyResult { iters, nodes }) = could_reach {
            tee_println!("Could reach with top {k} guides: {iters} ({nodes} nodes)");
        } else {
            tee_println!("Could NOT reach with top {k} guides");
        }
        entries.push(TopKEntry {
            k,
            best_single_iters: best_in_k,
            combined_iters: could_reach.map(|v| v.iters),
            combined_nodes: could_reach.map(|v| v.nodes),
        });
    }
    entries
}

fn random_k(
    max_iters: usize,
    successful: &[MeasuredGuide<MathLabel>],
    go: &RecExpr<Math>,
    n_trials: usize,
) -> Vec<RandomEntry> {
    let mut entries = Vec::new();
    for k in N_RANDOM {
        if k > successful.len() {
            continue;
        }
        let trials = (0..n_trials)
            .into_par_iter()
            .map(|trial_idx| {
                let mut rng = ChaCha12Rng::seed_from_u64(trial_idx as u64);
                let subset = successful
                    .choose_multiple(&mut rng, k)
                    .map(|v| v.guide.clone())
                    .collect::<Vec<_>>();
                let single_iters = subset
                    .iter()
                    .map(|guide| {
                        verify_reachability(
                            std::slice::from_ref(guide),
                            go,
                            RULES.get_or_init(math::rules),
                            max_iters,
                        )
                    })
                    .collect();
                let combined =
                    verify_reachability(&subset, go, RULES.get_or_init(math::rules), max_iters);
                RandomTrial {
                    single_iters,
                    combined_iters: combined.map(|v| v.iters),
                    combined_nodes: combined.map(|v| v.nodes),
                }
            })
            .collect::<Vec<_>>();

        let best_single_iters = trial_avg(&trials, |t| {
            t.single_iters.iter().flatten().map(|v| v.iters).min()
        });
        let best_single_nodes = trial_avg(&trials, |t| {
            t.single_iters.iter().flatten().map(|v| v.nodes).min()
        });
        if let (Some((avg_i, n_i)), Some((avg_n, _))) = (best_single_iters, best_single_nodes) {
            tee_println!(
                "Best single ITERS guide in top {k} guides: {avg_i:.1} (avg over {n_i}/{n_trials} trials)"
            );
            tee_println!(
                "Best single NODES guide in top {k} guides: {avg_n:.1} (avg over {n_i}/{n_trials} trials)"
            );
        } else {
            tee_println!("No single guide in top {k} could reach it");
        }
        let combined_iters = trial_avg(&trials, |t| t.combined_iters);
        let combined_nodes = trial_avg(&trials, |t| t.combined_nodes);
        if let (Some((avg_i, n)), Some((avg_n, _))) = (combined_iters, combined_nodes) {
            tee_println!(
                "Could reach with top {k} guides: {avg_i:.1} ({avg_n:.0} nodes) (avg over {n}/{n_trials} trials)"
            );
        } else {
            tee_println!("Could NOT reach with top {k} guides");
        }

        entries.push(RandomEntry { k, trials });
    }
    entries
}

#[expect(clippy::cast_precision_loss, clippy::shadow_unrelated)]
fn print_summary(results: &[EvalResult<MathLabel>], goal: &TreeNode<MathLabel>, max_iters: usize) {
    let mut successful = results
        .iter()
        .filter(|r| r.values.is_some())
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

    successful.sort_unstable_by_key(|v| v.guide.zs_distance);
    tee_println!("ZHANG SHASHA");

    let (min, med, max) = min_med_max(&successful, |v| v.guide.zs_distance);
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Successful guide zs_dists:"
    );

    let (min, med, max) = min_med_max(&successful, |v| v.values.unwrap().iters);
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Iterations to reach:"
    );

    tee_println!(
        "  First viable guide zs_distance: {}",
        successful[0].guide.zs_distance,
    );

    successful.sort_unstable_by_key(|v| v.guide.structural_distance);
    tee_println!("STRUCTURAL");

    let (min, med, max) = min_med_max(&successful, |v| v.guide.structural_distance);
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Successful guide struct_dist:"
    );

    let (min, med, max) = min_med_max(&successful, |v| v.values.unwrap().iters);
    tee_println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Iterations to reach:"
    );

    tee_println!(
        "  First viable guide structural_distance: {}",
        successful[0].guide.structural_distance,
    );
}
