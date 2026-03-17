use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use clap::Parser;
use egg::{RecExpr, Rewrite};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::Serialize;

use rise_distance::TreeNode;
use rise_distance::cli::{
    CsvRow, N_RANDOM, RandomEntry, RandomTrial, RankedGuide, SampleDistribution,
    enumerate_frontier_terms, get_run_folder, measure_guides, min_med_max, sample_frontier_terms,
    trial_avg,
};
use rise_distance::egg::math::{self, ConstantFold, Math, MathLabel};
use rise_distance::egg::{VerifyResult, run_guide_goal, verify_reachability};

#[derive(Parser)]
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

    /// Sample Strategy
    #[arg(long)]
    eval_all: bool,
}

type MathRankedGuide = RankedGuide<MathLabel>;

#[derive(Serialize, Debug)]
struct EvalResult<'a> {
    guide: &'a MathRankedGuide,
    values: Option<VerifyResult>,
}

static RULES: OnceLock<Vec<Rewrite<Math, ConstantFold>>> = OnceLock::new();

fn main() {
    let cli = Cli::parse();
    let prefix = format!("run-{}-{}-enumerate", cli.guide_iters, cli.goal_iters);
    let run_folder = get_run_folder(cli.output.as_deref(), "guide_eval", &prefix);

    let verify_iters = cli.verify_iters.unwrap_or(cli.goal_iters);

    let seed = cli
        .seed
        .parse::<RecExpr<Math>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed: {e}"));

    println!("Seed: {seed}");
    println!("Goal Iterations: {}", cli.goal_iters);
    println!("Guide Iterations: {}", cli.guide_iters);
    println!("Distribution: {}", cli.distribution);

    println!(
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

    println!("Eqsat completed in {:.2?}", start.elapsed());
    println!("Final egraph had {} nodes", result.goal_eg_size);

    println!(
        "\nSampling goals from iteration-{} frontier...",
        cli.goal_iters
    );
    let goals = sample_frontier_terms(
        &result.goal,
        &result.prev_goal,
        cli.goals,
        cli.max_size,
        cli.distribution,
    );
    assert!(
        !goals.is_empty(),
        "No frontier terms found. Try more iterations or a larger max-size.",
    );

    println!("Sampled {} goal(s)", goals.len());

    println!(
        "\nGetting guides from iteration-{} frontier...",
        cli.guide_iters
    );

    let guides = enumerate_frontier_terms(&result.guide, &result.prev_guide, cli.max_size);
    assert!(!guides.is_empty(), "No frontier terms found");
    println!("Found {} guide(s)", guides.len());

    let mut all_top_k = Vec::new();
    let n_goals = goals.len();
    for (goal_idx, goal) in goals.iter().enumerate() {
        println!(
            "\n=== Goal {}/{} (size {})===",
            goal_idx + 1,
            n_goals,
            goal.size_without_types()
        );
        println!("{goal}\n");

        let mut ranked = measure_guides(&guides, goal);

        let timer = Instant::now();

        if cli.eval_all {
            eval_all(verify_iters, &ranked, goal, &run_folder);
        }
        println!("Verification completed in {:.2?}", timer.elapsed());

        let top_k = eval_top_k(&mut ranked, goal, verify_iters);
        all_top_k.push(top_k);
    }
    let json_output =
        File::create(run_folder.join("top_k.json")).expect("Failed to create JSON output file");
    serde_json::to_writer_pretty(json_output, &all_top_k).expect("write top-k JSON");
}

fn eval_all(
    verify_iters: usize,
    ranked: &[MathRankedGuide],
    goal: &TreeNode<MathLabel>,
    run_folder: &Path,
) {
    let csv_output =
        File::create(run_folder.join("out.csv")).expect("Failed to create output file");
    let mut csv_writer = csv::Writer::from_writer(csv_output);
    let goal_recexpr = goal.into();
    let pb_style = ProgressStyle::with_template(
        "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}] verifying guides",
    )
    .unwrap();
    println!(
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
    for r in &results {
        csv_writer
            .serialize(CsvRow::new(goal, r.guide, r.values))
            .expect("write CSV row");
    }
    println!("Wrote goal to CSV");
    csv_writer.flush().expect("flush output");
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
    results: &mut [MathRankedGuide],
    goal: &TreeNode<MathLabel>,
    max_iters: usize,
) -> TopKResults {
    println!("Testing out top-k to see if that improves things");
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
    let best_zs_guide = results[0].guide.clone();
    println!("ZS DISTANCE:");
    top_k_results.zs = top_k(max_iters, results, &go);

    results.sort_unstable_by_key(|v| v.structural_distance);
    let best_structural_guide = results[0].guide.clone();
    println!("STRUCTURAL DISTANCE:");
    top_k_results.structural = top_k(max_iters, results, &go);

    println!("RANDOM (10 trials averaged):");
    top_k_results.random = random_k(max_iters, results, &go, 10, None);

    println!("RANDOM + best ZS helper (10 trials):");
    top_k_results.random_with_best_zs = random_k(
        max_iters,
        results,
        &go,
        10,
        Some(("best_zs", &best_zs_guide)),
    );

    println!("RANDOM + best structural helper (10 trials):");
    top_k_results.random_with_best_structural = random_k(
        max_iters,
        results,
        &go,
        10,
        Some(("best_structural", &best_structural_guide)),
    );

    top_k_results
}

fn top_k(max_iters: usize, ranked: &[MathRankedGuide], go: &RecExpr<Math>) -> Vec<TopKEntry> {
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
            println!("Best single guide in top {k} guides: {i}");
        } else {
            println!("No single guide in top {k} could reach it");
        }
        if let Some(VerifyResult { iters, nodes }) = could_reach {
            println!("Could reach with top {k} guides: {iters} ({nodes} nodes)");
        } else {
            println!("Could NOT reach with top {k} guides");
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
    successful: &[MathRankedGuide],
    go: &RecExpr<Math>,
    n_trials: usize,
    helper: Option<(&str, &TreeNode<MathLabel>)>,
) -> Vec<RandomEntry> {
    if let Some((name, h)) = &helper {
        println!("  Helper guide ({name}): {h}");
    }
    let mut entries = Vec::new();
    let helper_label = helper.as_ref().map_or("none", |(name, _)| name);
    for k in N_RANDOM {
        if k > successful.len() {
            continue;
        }
        let trials = (0..n_trials)
            .into_par_iter()
            .map(|trial_idx| {
                let mut rng = ChaCha12Rng::seed_from_u64(trial_idx as u64);
                let mut subset = successful
                    .choose_multiple(&mut rng, k)
                    .map(|v| v.guide.clone())
                    .collect::<Vec<_>>();
                if let Some((_, h)) = &helper {
                    subset.push((*h).clone());
                }
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
            println!(
                "Best single ITERS guide in top {k} guides (helper={helper_label}): {avg_i:.1} (avg over {n_i}/{n_trials} trials)"
            );
            println!(
                "Best single NODES guide in top {k} guides (helper={helper_label}): {avg_n:.1} (avg over {n_i}/{n_trials} trials)"
            );
        } else {
            println!("No single guide in top {k} could reach it (helper={helper_label})");
        }
        let combined_iters = trial_avg(&trials, |t| t.combined_iters);
        let combined_nodes = trial_avg(&trials, |t| t.combined_nodes);
        if let (Some((avg_i, n)), Some((avg_n, _))) = (combined_iters, combined_nodes) {
            println!(
                "Could reach with top {k} guides (helper={helper_label}): {avg_i:.1} ({avg_n:.0} nodes) (avg over {n}/{n_trials} trials)"
            );
        } else {
            println!("Could NOT reach with top {k} guides (helper={helper_label})");
        }

        entries.push(RandomEntry { k, trials });
    }
    entries
}

#[expect(clippy::cast_precision_loss, clippy::shadow_unrelated)]
fn print_summary(results: &[EvalResult], goal: &TreeNode<MathLabel>, max_iters: usize) {
    let mut successful = results
        .iter()
        .filter(|r| r.values.is_some())
        .collect::<Vec<_>>();
    let total = results.len();
    let n_reached = successful.len();

    println!("\nSummary for goal: {goal}");
    println!("  Goal size: {}", goal.size_without_types());
    println!("  Total guides evaluated: {total}");
    println!("  Max verify iterations: {max_iters}");
    println!(
        "  Guides that reached goal: {n_reached}/{total} ({:.1}%)",
        100.0 * n_reached as f64 / total.max(1) as f64
    );

    if successful.is_empty() {
        return;
    }

    successful.sort_unstable_by_key(|v| v.guide.zs_distance);
    println!("RAW ZS");

    let (min, med, max) = min_med_max(&successful, |v| v.guide.zs_distance);
    println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Successful guide zs_dists:"
    );

    let (min, med, max) = min_med_max(&successful, |v| v.values.unwrap().iters);
    println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Iterations to reach:"
    );

    println!(
        "  First viable guide zs_distance: {}",
        successful[0].guide.zs_distance,
    );

    successful.sort_unstable_by_key(|v| v.guide.structural_distance);
    println!("STRUCTURAL");

    let (min, med, max) = min_med_max(&successful, |v| v.guide.structural_distance);
    println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Successful guide struct_dist:"
    );

    let (min, med, max) = min_med_max(&successful, |v| v.values.unwrap().iters);
    println!(
        "  {:<30} min={min}, median={med}, max={max}",
        "Iterations to reach:"
    );

    println!(
        "  First viable guide structural_distance: {}",
        successful[0].guide.structural_distance,
    );
}
