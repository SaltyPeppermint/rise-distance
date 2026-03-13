use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use hashbrown::HashSet;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::{BigUint, ToPrimitive};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use rise_distance::cli::{SampleDistribution, SampleStrategy, get_run_folder, is_frontier, log};
use rise_distance::egg::math::{self, ConstantFold, Math, MathLabel};
use rise_distance::egg::{VerifyResult, run_guide_goal, verify_reachability};
use rise_distance::{
    EGraph, StructuralDistance, TermCount, TreeNode, UnitCost, structural_diff, tree_distance_unit,
};
use serde::Serialize;

#[derive(Parser)]
#[command(
    about = "Evaluate distance metrics as guide predictors for equality saturation",
    after_help = "\
Examples:
  # Basic evaluation with Zhang-Shasha distance
  guide-eval -s '(d x (+ (* x x) 1))' -n 5 -i 4 -g 100 --max-size 150 -d zhang-shasha

  # Guide from an earlier iteration
  guide-eval -s '(d x (+ (* x x) 1))' -n 8 -i 3 -g 100 --goals 5 -d structural

  # Write JSON output to a file
  guide-eval -s '(d x (+ (* x x) 1))' -n 5 -i 4 -g 100 --max-size 150 -o results.json

  # Limit verification iterations separately from goal iterations
  guide-eval -s '(d x (+ (* x x) 1))' -n 8 -i 4 -g 100 --max-size 150 --verify-iters 5

  # Print top 20 guides in the summary table (default: 10)
  guide-eval -s '(d x (+ (* x x) 1))' -n 5 -i 4 -g 100 --max-size 150 -d zhang-shasha --top 20
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
    #[arg(short, long, conflicts_with = "enumerate")]
    guides: Option<usize>,

    /// Number of goal terms to sample from the n frontier
    #[arg(long, default_value_t = 1)]
    goals: usize,

    /// Max term size for counting/sampling (derived from egraph if omitted)
    #[arg(long)]
    max_size: Option<usize>,

    /// How to distribute the sample budget across sizes.
    /// Options: uniform, proportional:<`min_per_size`>, normal:<sigma>
    #[arg(long, default_value_t = SampleDistribution::Uniform)]
    distribution: SampleDistribution,

    /// JSON output file (generated if omitted)
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
    strategy: SampleStrategy,

    /// Sample Strategy
    #[arg(long)]
    eval_all: bool,

    /// Sample Strategy
    #[arg(long)]
    top_k: bool,
}

#[derive(Serialize, Debug)]
struct EvalResult<'a> {
    guide: &'a RankedGuide,
    values: Option<VerifyResult>,
}

static RULES: OnceLock<Vec<Rewrite<Math, ConstantFold>>> = OnceLock::new();

fn main() {
    let cli = Cli::parse();
    let prefix = format!(
        "run-{}-{}-{}",
        cli.guide_iters, cli.goal_iters, cli.strategy
    );
    let run_folder = get_run_folder(cli.output.as_deref(), "guide_eval", &prefix);
    let mut log = File::create(run_folder.join("out.log")).expect("Failed to create output file");

    let verify_iters = cli.verify_iters.unwrap_or(cli.goal_iters);

    let seed = cli
        .seed
        .parse::<RecExpr<Math>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed: {e}"));

    log!(log, "Seed: {seed}");
    log!(log, "Goal Iterations: {}", cli.goal_iters);
    log!(log, "Guide Iterations: {}", cli.guide_iters);
    // log!(log, "Distance metric: {}", cli.distance);
    log!(log, "Distribution: {}", cli.distribution);

    // Step 1: Grow egraph and capture snapshots at guide and target iterations
    log!(
        log,
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

    log!(log, "Eqsat completed in {:.2?}", start.elapsed());
    log!(log, "Final egraph had {} nodes", result.goal_eg_size);

    // Step 2: Sample goals from the iteration-n frontier
    log!(
        log,
        "\nSampling goals from iteration-{} frontier...",
        cli.goal_iters
    );
    let max_size = cli.max_size.unwrap_or_else(|| {
        let m = AstSize.cost_rec(&seed);
        log!(log, "No max_size was given, using 1 goal size (={m})");
        m
    });
    let goals = get_goal_term(
        &result.goal,
        &result.prev_goal,
        cli.goals,
        max_size,
        cli.distribution,
        &mut log,
    );
    if goals.is_empty() {
        log!(
            log,
            "No frontier terms found at iteration {}. Try more iterations or a larger max-size.",
            cli.goal_iters
        );
        return;
    }
    log!(log, "Sampled {} goal(s)", goals.len());

    // Step 3: Get guides from the iteration-(n-1) frontier (shared across all goals)
    log!(
        log,
        "\nGetting guides from iteration-{} frontier...",
        cli.guide_iters
    );
    let guide_count = if let SampleStrategy::Enumerate = cli.strategy {
        usize::MAX
    } else {
        cli.guides
            .expect("-g/--guides is required when not using --strategy enumerate")
    };
    let guides = get_guide_terms(
        &result.guide,
        &result.prev_guide,
        guide_count,
        max_size,
        cli.distribution,
        cli.strategy,
        &mut log,
    );
    if guides.is_empty() {
        log!(
            log,
            "No frontier terms found at iteration {}.",
            cli.guide_iters
        );
    } else {
        log!(log, "Found {} guide(s)", guides.len());
    }

    run_eval(
        &goals,
        &guides,
        verify_iters,
        cli.eval_all,
        cli.top_k,
        &run_folder,
        &mut log,
    );
    log.flush().unwrap();
}

fn run_eval(
    goals: &[TreeNode<MathLabel>],
    guides: &[TreeNode<MathLabel>],
    verify_iters: usize,
    eval_all_guides: bool,
    eval_top: bool,
    run_folder: &Path,
    log: &mut File,
) {
    let mut all_top_k = Vec::new();
    let n_goals = goals.len();
    for (goal_idx, goal) in goals.iter().enumerate() {
        log!(
            log,
            "\n=== Goal {}/{} (size {})===",
            goal_idx + 1,
            n_goals,
            goal.size_without_types()
        );
        log!(log, "{goal}\n");

        let mut ranked = rank_guides(guides, goal);

        let timer = Instant::now();

        if eval_all_guides {
            eval_all(verify_iters, &ranked, goal, run_folder, log);
        }
        log!(log, "Verification completed in {:.2?}", timer.elapsed());

        if eval_top {
            let top_k = eval_top_k(&mut ranked, goal, verify_iters, log);
            all_top_k.push(top_k);
        }
    }
    let json_output =
        File::create(run_folder.join("top_k.json")).expect("Failed to create JSON output file");
    serde_json::to_writer_pretty(json_output, &all_top_k).expect("write top-k JSON");
}

fn eval_all(
    verify_iters: usize,
    ranked: &[RankedGuide],
    goal: &TreeNode<MathLabel>,
    run_folder: &Path,
    log: &mut File,
) {
    let csv_output =
        File::create(run_folder.join("out.csv")).expect("Failed to create output file");
    let mut csv_writer = csv::Writer::from_writer(csv_output);
    let goal_recexpr = goal.into();
    let pb_style = ProgressStyle::with_template(
        "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}] verifying guides",
    )
    .unwrap();
    log!(
        log,
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
    print_summary(&results, goal, verify_iters, log);
    for r in &results {
        csv_writer
            .serialize(CsvRow::new(goal, r))
            .expect("write CSV row");
    }
    log!(log, "Wrote goal to CSV");
    csv_writer.flush().expect("flush output");
    // results
}

#[derive(Serialize)]
struct CsvRow {
    goal: String,
    guide: String,
    zs_distance: usize,
    structural_overlap: usize,
    structural_zs_sum: usize,
    zs_rank: usize,
    structural_rank: usize,
    iterations_to_reach: Option<usize>,
    nodes_to_reach: Option<usize>,
}

impl CsvRow {
    fn new(goal: &TreeNode<MathLabel>, r: &EvalResult) -> CsvRow {
        CsvRow {
            goal: goal.to_string(),
            guide: r.guide.guide.to_string(),
            zs_distance: r.guide.zs_distance,
            structural_overlap: r.guide.structural_distance.overlap(),
            structural_zs_sum: r.guide.structural_distance.zs_sum(),
            zs_rank: r.guide.zs_rank,
            structural_rank: r.guide.structural_rank,
            iterations_to_reach: r.values.map(|v| v.iters),
            nodes_to_reach: r.values.map(|v| v.nodes),
        }
    }
}

/// Get frontier terms from `egraph` that are NOT present in `prev_raw_egg`.
///
/// When `enumerate` is true, enumerates all terms exhaustively.
/// Otherwise, samples terms using the given distribution.
fn get_guide_terms(
    egraph: &EGraph<MathLabel>,
    prev_raw_egg: &egg::EGraph<Math, ConstantFold>,
    count: usize,
    max_size: usize,
    distribution: SampleDistribution,
    strategy: SampleStrategy,
    log: &mut File,
) -> Vec<TreeNode<MathLabel>> {
    let tc = TermCount::<BigUint, _>::new(max_size, false, egraph);

    let Some(histogram) = tc.of_root() else {
        return Vec::new();
    };

    let mut sorted_hist = histogram.iter().collect::<Vec<_>>();
    sorted_hist.sort_unstable();
    log!(log, "Terms in guide frontier:");
    for (k, v) in &sorted_hist {
        log!(log, "{v} terms of size {k}");
    }

    let start = Instant::now();
    let result = match strategy {
        SampleStrategy::Enumerate => {
            let total_terms = histogram.values().cloned().sum::<BigUint>();
            log!(
                log,
                "Enumerating all {total_terms} terms up to size {max_size}"
            );
            assert!(
                total_terms.to_usize().is_some(),
                "Cannot enumerate more than usize!"
            );

            tc.enumerate_root(max_size, Some(ProgressBar::new(max_size as u64)))
                .into_iter()
                .filter(|t| is_frontier(t, prev_raw_egg))
                .collect::<Vec<_>>()
        }
        SampleStrategy::Random => {
            let min_size = histogram.keys().min().copied().unwrap_or(1);
            #[expect(clippy::cast_precision_loss)]
            let normal_center = (min_size + max_size) as f64 / 2.0;

            let mut result = HashSet::new();
            let mut oversample = 5;
            loop {
                let samples_per_size = distribution.samples_per_size(
                    histogram,
                    min_size,
                    max_size,
                    count * oversample,
                    normal_center,
                );
                let batch = tc.sample_unique_root(min_size, max_size, &samples_per_size);
                let prev_len = result.len();
                result.extend(batch.into_iter().filter(|t| is_frontier(t, prev_raw_egg)));
                if result.len() >= count || result.len() == prev_len {
                    break;
                }
                oversample *= 2;
                log!(
                    log,
                    "Have {}/{count} frontier guides, retrying with {oversample}x oversample...",
                    result.len()
                );
            }
            result.into_iter().take(count).collect()
        }
    };
    log!(
        log,
        "Spent {} seconds enumerating/sampling the terms",
        start.elapsed().as_secs()
    );
    result
}

/// Get frontier terms from `egraph` that are NOT present in `prev_raw_egg`.
///
/// When `enumerate` is true, enumerates all terms exhaustively.
/// Otherwise, samples terms using the given distribution.
fn get_goal_term(
    egraph: &EGraph<MathLabel>,
    prev_raw_egg: &egg::EGraph<Math, ConstantFold>,
    count: usize,
    max_size: usize,
    distribution: SampleDistribution,
    log: &mut File,
) -> Vec<TreeNode<MathLabel>> {
    let tc = TermCount::<BigUint, _>::new(max_size, false, egraph);

    let Some(histogram) = tc.of_root() else {
        return Vec::new();
    };

    let mut sorted_hist = histogram.iter().collect::<Vec<_>>();
    sorted_hist.sort_unstable();
    log!(log, "Terms in guide frontier:");
    for (k, v) in &sorted_hist {
        log!(log, "{v} terms of size {k}");
    }

    let min_size = histogram.keys().min().copied().unwrap_or(1);
    #[expect(clippy::cast_precision_loss)]
    let normal_center = (min_size + max_size) as f64 / 2.0;

    let mut result = HashSet::new();
    let mut oversample = 5;
    loop {
        let samples_per_size = distribution.samples_per_size(
            histogram,
            min_size,
            max_size,
            count * oversample,
            normal_center,
        );
        let batch = tc.sample_unique_root(min_size, max_size, &samples_per_size);
        let prev_len = result.len();
        result.extend(batch.into_iter().filter(|t| is_frontier(t, prev_raw_egg)));
        if result.len() >= count || result.len() == prev_len {
            break;
        }
        oversample *= 2;
        log!(
            log,
            "Have {}/{count} frontier goals, retrying with {oversample}x oversample...",
            result.len()
        );
    }
    result.into_iter().take(count).collect()
}

#[derive(Serialize, Debug)]
struct RankedGuide {
    guide: TreeNode<MathLabel>,
    zs_distance: usize,
    #[serde(flatten)]
    structural_distance: StructuralDistance,
    zs_rank: usize,
    structural_rank: usize,
}

/// Rank guides by distance to the goal. Returns `(distance, guide)` pairs sorted ascending.
fn rank_guides(guides: &[TreeNode<MathLabel>], goal: &TreeNode<MathLabel>) -> Vec<RankedGuide> {
    let goal_flat = goal.flatten(false);
    let pb_style = ProgressStyle::with_template(
        "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}] ranking guides",
    )
    .unwrap();
    let r = guides
        .par_iter()
        .progress_with_style(pb_style)
        .map(|guide| {
            let guide_flat = guide.flatten(false);

            let zs_dist = tree_distance_unit(&guide_flat, &goal_flat);
            let structural_dist = structural_diff(&goal_flat, &guide_flat, &UnitCost);
            (zs_dist, structural_dist, guide.clone())
        })
        .collect::<Vec<_>>();
    let mut zs_ranks = (0..r.len()).collect::<Vec<_>>();
    zs_ranks.sort_by(|&a, &b| r[a].0.cmp(&r[b].0));
    let mut structural_ranks = (0..r.len()).collect::<Vec<_>>();
    structural_ranks.sort_by(|&a, &b| r[a].1.cmp(&r[b].1));
    r.into_iter()
        .zip(zs_ranks)
        .zip(structural_ranks)
        .map(|((a, zs_rank), structural_rank)| RankedGuide {
            guide: a.2,
            zs_distance: a.0,
            structural_distance: a.1,
            zs_rank,
            structural_rank,
        })
        .collect()
}

const TOP_K: [usize; 6] = [1, 2, 5, 10, 50, 100];

#[derive(Serialize)]
struct TopKEntry {
    k: usize,
    best_single_iters: Option<usize>,
    combined_iters: Option<usize>,
    combined_nodes: Option<usize>,
}

#[derive(Serialize)]
struct TopKRandomTrial {
    single_iters: Vec<Option<VerifyResult>>,
    combined_iters: Option<usize>,
    combined_nodes: Option<usize>,
}

#[derive(Serialize)]
struct TopKRandomEntry {
    k: usize,
    trials: Vec<TopKRandomTrial>,
}

#[derive(Serialize)]
struct TopKResults {
    goal: String,
    zs: Vec<TopKEntry>,
    structural: Vec<TopKEntry>,
    random: Vec<TopKRandomEntry>,
    random_with_best_zs: Vec<TopKRandomEntry>,
    random_with_best_structural: Vec<TopKRandomEntry>,
    known_iters: Vec<TopKEntry>,
}

fn eval_top_k(
    results: &mut [RankedGuide],
    goal: &TreeNode<MathLabel>,
    max_iters: usize,
    log: &mut File,
) -> TopKResults {
    log!(log, "Testing out top-k to see if that improves things");
    let go = goal.into();

    let mut top_k_results = TopKResults {
        goal: goal.to_string(),
        zs: Vec::new(),
        structural: Vec::new(),
        random: Vec::new(),
        random_with_best_zs: Vec::new(),
        random_with_best_structural: Vec::new(),
        known_iters: Vec::new(),
    };

    if !results.is_empty() {
        results.sort_unstable_by_key(|v| v.zs_rank);
        let best_zs_guide = results[0].guide.clone();
        log!(log, "ZS DISTANCE:");
        top_k_results.zs = top_k(max_iters, log, results, &go);

        results.sort_unstable_by_key(|v| v.structural_rank);
        let best_structural_guide = results[0].guide.clone();
        log!(log, "STRUCTURAL DISTANCE:");
        top_k_results.structural = top_k(max_iters, log, results, &go);

        log!(log, "RANDOM (10 trials averaged):");
        top_k_results.random = top_k_random(max_iters, log, results, &go, 10, None);

        log!(log, "RANDOM + best ZS helper (10 trials):");
        top_k_results.random_with_best_zs = top_k_random(
            max_iters,
            log,
            results,
            &go,
            10,
            Some(("best_zs", &best_zs_guide)),
        );

        log!(log, "RANDOM + best structural helper (10 trials):");
        top_k_results.random_with_best_structural = top_k_random(
            max_iters,
            log,
            results,
            &go,
            10,
            Some(("best_structural", &best_structural_guide)),
        );
    }

    top_k_results
}

fn top_k(
    max_iters: usize,
    log: &mut File,
    ranked: &[RankedGuide],
    go: &RecExpr<Math>,
) -> Vec<TopKEntry> {
    let mut entries = Vec::new();
    for k in TOP_K {
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
            .iter()
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
            log!(log, "Best single guide in top {k} guides: {i}");
        } else {
            log!(log, "No single guide in top {k} could reach it");
        }
        if let Some(VerifyResult { iters, nodes }) = could_reach {
            log!(
                log,
                "Could reach with top {k} guides: {iters} ({nodes} nodes)"
            );
        } else {
            log!(log, "Could NOT reach with top {k} guides");
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

#[expect(clippy::cast_precision_loss)]
fn top_k_random(
    max_iters: usize,
    log: &mut File,
    successful: &mut [RankedGuide],
    go: &RecExpr<Math>,
    n_trials: usize,
    helper: Option<(&str, &TreeNode<MathLabel>)>,
) -> Vec<TopKRandomEntry> {
    if let Some((name, h)) = &helper {
        log!(log, "  Helper guide ({name}): {h}");
    }
    let mut rng = ChaCha12Rng::seed_from_u64(0);
    let mut entries = Vec::new();
    let helper_label = helper.as_ref().map_or("none", |(name, _)| name);
    for k in TOP_K {
        if k > successful.len() {
            continue;
        }
        let trials = (0..n_trials)
            .map(|_| {
                successful.shuffle(&mut rng);
                let mut top_guides: Vec<_> =
                    successful[0..k].iter().map(|v| v.guide.clone()).collect();
                if let Some((_, h)) = &helper {
                    top_guides.push((*h).clone());
                }
                let single_iters = successful[0..k]
                    .iter()
                    .map(|v| {
                        verify_reachability(
                            std::slice::from_ref(&v.guide),
                            go,
                            RULES.get_or_init(math::rules),
                            max_iters,
                        )
                    })
                    .collect();
                let combined =
                    verify_reachability(&top_guides, go, RULES.get_or_init(math::rules), max_iters);
                TopKRandomTrial {
                    single_iters,
                    combined_iters: combined.map(|v| v.iters),
                    combined_nodes: combined.map(|v| v.nodes),
                }
            })
            .collect::<Vec<_>>();

        let n_best = trials
            .iter()
            .filter(|t| t.single_iters.iter().any(|i| i.is_some()))
            .count();
        let n_combined = trials.iter().filter(|t| t.combined_iters.is_some()).count();
        if n_best > 0 {
            let avg_best_iters = trials
                .iter()
                .map(|t| {
                    t.single_iters
                        .iter()
                        .flatten()
                        .map(|v| v.iters)
                        .min()
                        .unwrap()
                })
                .sum::<usize>() as f64
                / n_best as f64;
            log!(
                log,
                "Best single ITERS guide in top {k} guides (helper={helper_label}): {avg_best_iters:.1} (avg over {n_best}/{n_trials} trials)"
            );
            let avg_best_nodes = trials
                .iter()
                .map(|t| {
                    t.single_iters
                        .iter()
                        .flatten()
                        .map(|v| v.nodes)
                        .min()
                        .unwrap()
                })
                .sum::<usize>() as f64
                / n_best as f64;
            log!(
                log,
                "Best single NODES guide in top {k} guides (helper={helper_label}): {avg_best_nodes:.1} (avg over {n_best}/{n_trials} trials)"
            );
        } else {
            log!(
                log,
                "No single guide in top {k} could reach it (helper={helper_label})"
            );
        }
        if n_combined > 0 {
            let avg_iters = trials
                .iter()
                .filter_map(|t| t.combined_iters)
                .sum::<usize>() as f64
                / n_combined as f64;
            let avg_nodes = trials
                .iter()
                .filter_map(|t| t.combined_nodes)
                .sum::<usize>() as f64
                / n_combined as f64;
            log!(
                log,
                "Could reach with top {k} guides (helper={helper_label}): {avg_iters:.1} ({avg_nodes:.0} nodes) (avg over {n_combined}/{n_trials} trials)"
            );
        } else {
            log!(
                log,
                "Could NOT reach with top {k} guides (helper={helper_label})"
            );
        }

        entries.push(TopKRandomEntry { k, trials });
    }
    entries
}

#[expect(clippy::cast_precision_loss, clippy::too_many_lines)]
fn print_summary(
    results: &[EvalResult],
    goal: &TreeNode<MathLabel>,
    max_iters: usize,
    // top: usize,
    log: &mut File,
) {
    let mut successful = results
        .iter()
        .filter(|r| r.values.is_some())
        .collect::<Vec<_>>();
    let total = results.len();
    let n_reached = successful.len();

    log!(log, "\nSummary for goal: {goal}");
    log!(log, "  Goal size: {}", goal.size_without_types());
    log!(log, "  Total guides evaluated: {total}");
    log!(log, "  Max verify iterations: {max_iters}");
    log!(
        log,
        "  Guides that reached goal: {n_reached}/{total} ({:.1}%)",
        100.0 * n_reached as f64 / total.max(1) as f64
    );

    if !successful.is_empty() {
        successful.sort_unstable_by_key(|v| v.guide.zs_rank);
        log!(log, "RAW ZS");
        log!(
            log,
            "  Successful guide zs_ranks:      min={}, median={}, max={}",
            successful.first().unwrap().guide.zs_rank,
            successful[successful.len() / 2].guide.zs_rank,
            successful.last().unwrap().guide.zs_rank,
        );
        log!(
            log,
            "  Successful guide zs_dists:  min={}, median={}, max={}",
            successful
                .iter()
                .map(|v| v.guide.zs_distance)
                .min()
                .unwrap(),
            successful[successful.len() / 2].guide.zs_distance,
            successful
                .iter()
                .map(|v| v.guide.zs_distance)
                .max()
                .unwrap(),
        );
        if !successful.is_empty() {
            log!(
                log,
                "  Iterations to reach:         min={}, median={}, max={}",
                successful
                    .iter()
                    .filter_map(|v| v.values.map(|v| v.iters))
                    .min()
                    .unwrap(),
                successful[successful.len() / 2].values.unwrap().iters,
                successful
                    .iter()
                    .filter_map(|v| v.values.map(|v| v.iters))
                    .max()
                    .unwrap(),
            );
            // First successful rank (how far down the ranking before finding a viable guide)
            log!(
                log,
                "  First viable guide at zs_rank: {} (zs_distance {})",
                successful[0].guide.zs_rank,
                successful[0].guide.zs_distance,
            );
        }

        successful.sort_unstable_by_key(|v| v.guide.structural_rank);
        log!(log, "STRUCTURAL");
        log!(
            log,
            "  Successful guide structural_rank:      min={}, median={}, max={}",
            successful.first().unwrap().guide.structural_rank,
            successful[successful.len() / 2].guide.structural_rank,
            successful.last().unwrap().guide.structural_rank,
        );
        log!(
            log,
            "  Successful guide structural_dist:  min={}, median={}, max={}",
            successful
                .iter()
                .map(|v| v.guide.structural_distance)
                .min()
                .unwrap(),
            successful[successful.len() / 2].guide.structural_distance,
            successful
                .iter()
                .map(|v| v.guide.structural_distance)
                .max()
                .unwrap(),
        );
        if !successful.is_empty() {
            log!(
                log,
                "  Iterations to reach:         min={}, median={}, max={}",
                successful
                    .iter()
                    .filter_map(|v| v.values.map(|v| v.iters))
                    .min()
                    .unwrap(),
                successful[successful.len() / 2].values.unwrap().iters,
                successful
                    .iter()
                    .filter_map(|v| v.values.map(|v| v.iters))
                    .max()
                    .unwrap(),
            );

            // First successful rank (how far down the ranking before finding a viable guide)
            log!(
                log,
                "  First viable guide at structural_rank: {} (structural_distance {})",
                successful[0].guide.structural_rank,
                successful[0].guide.structural_distance,
            );
        }
    }
}
