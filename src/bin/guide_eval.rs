use std::env::current_dir;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use csv::Writer;
use egg::{AstSize, CostFunction, RecExpr, Rewrite, Runner, SimpleScheduler};
use hashbrown::HashMap;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::BigUint;
use rayon::prelude::*;

use rise_distance::cli::{SampleDistribution, SampleStrategy, log};
use rise_distance::egg::math::{ConstantFold, Math, MathLabel};
use rise_distance::egg::run_guide_goal;
use rise_distance::{
    EGraph, StructuralDistance, TermCount, TreeNode, UnitCost, structural_diff, tree_distance_unit,
};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(
    about = "Evaluate distance metrics as guide predictors for equality saturation",
    after_help = "\
Examples:
  # Basic evaluation with Zhang-Shasha distance
  guide-eval -s '(d x (+ (* x x) 1))' -n 5 -i 4 -g 100 --max-size 150 -d zhang-shasha

  # Guide from an earlier iteration
  guide-eval -s '(d x (+ (* x x) 1))' -n 8 -i 3 -g 100 --goals 5 -d structural

  # Write CSV output to a file
  guide-eval -s '(d x (+ (* x x) 1))' -n 5 -i 4 -g 100 --max-size 150 -o results.csv

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

    // /// Distance metric to use for ranking
    // #[arg(long)] // default_value_t = DistanceMetric::ZhangShasha
    // distance: DistanceMetric,
    /// How to distribute the sample budget across sizes.
    /// Options: uniform, proportional:<`min_per_size`>, normal:<sigma>
    #[arg(long, default_value_t = SampleDistribution::Uniform)]
    distribution: SampleDistribution,

    /// CSV output file (generated if omitted)
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

    #[arg(long)]
    log_file: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct VerifyResult {
    #[serde(flatten)]
    guide: RankedGuide,
    reached: bool,
    iterations_to_reach: Option<usize>,
}

// struct PlotConfig {
//     strategy: SampleStrategy,
//     distance: DistanceMetric,
//     distribution: SampleDistribution,
//     num_guides: Option<usize>,
//     guide_iters: usize,
//     goal_iters: usize,
//     verify_iters: usize,
// }

fn main() {
    let cli = Cli::parse();
    let mut log =
        File::create(output_file_path(&cli, ".log")).expect("Failed to create output file");

    let rules = rise_distance::egg::math::rules();
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
    let [(prev_eg_guide, eg_guide), (prev_eg_goal, eg_goal)] =
        run_guide_goal(&seed, &rules, cli.guide_iters, cli.goal_iters);
    log!(log, "Eqsat completed in {:.2?}", start.elapsed());

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
        &eg_goal,
        &prev_eg_goal,
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

    let mut guides_per_goal = HashMap::new();

    for (id, goal) in goals.into_iter().enumerate() {
        // Step 3: Get guides from the iteration-(n-1) frontier
        log!(log, "Looking at goal {id}: {goal}");
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
            &eg_guide,
            &prev_eg_guide,
            guide_count,
            max_size,
            cli.distribution,
            cli.strategy,
            &goal,
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
        guides_per_goal.insert(goal, guides);
    }

    // Step 4 & 5: For each goal, rank guides by distance, then verify reachability
    // let plot_cfg = PlotConfig {
    //     strategy: cli.strategy,
    //     distance: cli.distance,
    //     distribution: cli.distribution,
    //     num_guides: cli.guides,
    //     guide_iters: cli.guide_iters,
    //     goal_iters: cli.goal_iters,
    //     verify_iters,
    // };
    run_eval(
        &guides_per_goal,
        &rules,
        verify_iters,
        // cli.top,
        File::create(output_file_path(&cli, ".csv")).expect("Failed to create output file"),
        &mut log,
    );
    log.flush().unwrap();
}

/// Create an output file for this run.
///
/// If `-o` was given, uses that path directly. Otherwise, auto-generates a
/// filename like `runs/run-{guide_iters}-{goal_iters}-sampling_{N}{ending}`
/// inside a `runs/` directory (created if needed), where `N` is one higher
/// than the largest existing run number.
/// `ending` should include the leading dot (e.g. `".csv"`).
fn output_file_path(cli: &Cli, ending: &str) -> PathBuf {
    cli.output.as_deref().map_or_else(
        || {
            let runs_dir = current_dir().unwrap().join("runs");
            std::fs::create_dir_all(&runs_dir).expect("Failed to create runs/ directory");

            let pat = format!("run-{}-{}-sampling", cli.guide_iters, cli.goal_iters);
            let max_existing = runs_dir
                .read_dir()
                .unwrap()
                .filter_map(|e| {
                    if let Ok(d) = e
                        && !d.file_type().ok()?.is_dir()
                        && let Some(s) = d.file_name().to_str()
                        && s.contains(&pat)
                        && s.ends_with(ending)
                    {
                        return s.strip_suffix(ending)?.rsplit_once('_')?.1.parse().ok();
                    }
                    None
                })
                .max()
                .unwrap_or(0);
            let pat = format!("{pat}_{}{ending}", max_existing + 1);
            runs_dir.join(pat)
        },
        PathBuf::from,
    )
}

fn run_eval(
    guides_for_goal: &HashMap<TreeNode<MathLabel>, Vec<TreeNode<MathLabel>>>,
    rules: &[Rewrite<Math, ConstantFold>],
    verify_iters: usize,
    // top: usize,
    output: File,
    log: &mut File,
) {
    let mut csv_writer = Writer::from_writer(output);
    csv_writer
        .write_record([
            "goal",
            "zs_rank",
            "zs_distance",
            "structural_rank",
            "structural_distance",
            "reached",
            "iterations_to_reach",
            "guide",
        ])
        .expect("write CSV header");

    for (goal_idx, (goal, guides)) in guides_for_goal.into_iter().enumerate() {
        log!(
            log,
            "\n=== Goal {}/{} (size {})===\n",
            goal_idx + 1,
            guides_for_goal.len(),
            goal.size_without_types()
        );

        let ranked = rank_guides(guides, goal);
        log!(
            log,
            "Verifying {} guides (max {verify_iters} iters each)...",
            ranked.len(),
        );

        let timer = Instant::now();
        let goal_recexpr = goal
            .to_string()
            .parse::<RecExpr<Math>>()
            .expect("goal round-trips to RecExpr");

        let pb_style = ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}] verifying guides",
        )
        .unwrap();

        let results = ranked
            .into_par_iter()
            .progress_with_style(pb_style)
            .map(|ranked_guide| {
                let iters =
                    verify_reachability(&ranked_guide.guide, &goal_recexpr, rules, verify_iters);
                VerifyResult {
                    guide: ranked_guide,
                    reached: iters.is_some(),
                    iterations_to_reach: iters,
                }
            })
            .collect::<Vec<_>>();

        log!(log, "Verification completed in {:.2?}", timer.elapsed());

        print_summary(&results, goal, verify_iters, log);
        let goal_str = goal.to_string();
        for r in results {
            csv_writer
                .serialize(Record {
                    goal: goal_str.clone(),
                    verify_results: r,
                })
                .expect("write CSV row");
        }

        // let corr = rank_iterations_correlation(&results);
        // log!(
        //     log,
        //     "\n  Spearman rank correlation (rank vs iterations_to_reach): {corr:.4}"
        // );
        // plot_rank_vs_iterations(&results, goal, plot_path, plot_cfg, log);
    }
    csv_writer.flush().expect("flush CSV");
}

#[derive(Debug, Serialize, Deserialize)]
struct Record {
    goal: String,
    verify_results: VerifyResult,
}

/// Get frontier terms from `egraph` that are NOT present in `prev_raw_egg`.
///
/// When `enumerate` is true, enumerates all terms exhaustively.
/// Otherwise, samples terms using the given distribution.
#[expect(clippy::too_many_arguments)]
fn get_guide_terms(
    egraph: &EGraph<MathLabel>,
    prev_raw_egg: &egg::EGraph<Math, ConstantFold>,
    count: usize,
    max_size: usize,
    distribution: SampleDistribution,
    strategy: SampleStrategy,
    goal: &TreeNode<MathLabel>,
    log: &mut File,
) -> Vec<TreeNode<MathLabel>> {
    let tc = TermCount::<BigUint, _>::new(max_size, false, egraph);

    let Some(histogram) = tc.of_root() else {
        return Vec::new();
    };

    let mut sorted_hist = histogram.iter().collect::<Vec<_>>();
    sorted_hist.sort_unstable();
    log!(log, "Terms in goal frontier:");
    for (k, v) in &sorted_hist {
        log!(log, "{v} terms of size {k}");
    }

    match strategy {
        SampleStrategy::Enumerate => {
            let total_terms = histogram.values().cloned().sum::<BigUint>();
            log!(
                log,
                "Enumerating all {total_terms} terms up to size {max_size}"
            );

            let pb = ProgressBar::new(max_size as u64);
            tc.enumerate_root(max_size, Some(pb))
                .into_iter()
                .filter(|t| is_frontier(t, prev_raw_egg))
                .take(count)
                .collect()
        }
        SampleStrategy::Random => {
            let min_size = histogram.keys().min().copied().unwrap_or(1);
            // Oversample 5x to account for rejection filtering
            #[expect(clippy::cast_precision_loss)]
            let normal_center = (min_size + max_size) as f64 / 2.0;
            let samples_per_size = distribution.samples_per_size(
                histogram,
                min_size,
                max_size,
                count * 5,
                normal_center,
            );

            tc.sample_unique_root(min_size, max_size, &samples_per_size)
                .into_iter()
                .filter(|t| is_frontier(t, prev_raw_egg))
                .take(count)
                .collect()
        }
        SampleStrategy::Overlap => {
            let min_size = histogram.keys().min().copied().unwrap_or(1);
            // Oversample 5x to account for rejection filtering
            #[expect(clippy::cast_precision_loss)]
            let normal_center = (min_size + max_size) as f64 / 2.0;
            let samples_per_size = distribution.samples_per_size(
                histogram,
                min_size,
                max_size,
                count * 5,
                normal_center,
            );
            tc.sample_unique_root_overlap(goal, min_size, max_size, &samples_per_size)
                .into_iter()
                .filter(|t| is_frontier(t, prev_raw_egg))
                .take(count)
                .collect()
        }
    }
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
    // Oversample 5x to account for rejection filtering
    let total_samples = count * 5;

    #[expect(clippy::cast_precision_loss)]
    let normal_center = (min_size + max_size) as f64 / 2.0;
    let samples_per_size =
        distribution.samples_per_size(histogram, min_size, max_size, total_samples, normal_center);

    tc.sample_unique_root(min_size, max_size, &samples_per_size)
        .into_iter()
        .filter(|t| is_frontier(t, prev_raw_egg))
        .take(count)
        .collect()
}

/// Helper function to check if something is in the frontier
fn is_frontier(tree: &TreeNode<MathLabel>, prev_raw_egg: &egg::EGraph<Math, ConstantFold>) -> bool {
    let Ok(recexpr) = tree.to_string().parse::<RecExpr<Math>>() else {
        return false;
    };
    prev_raw_egg.lookup_expr(&recexpr).is_none()
}

#[derive(Serialize, Deserialize, Debug)]
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

/// Run eqsat from `guide` and check if `goal` becomes reachable.
/// Returns `Some(iteration)` if reached, `None` otherwise.
fn verify_reachability(
    guide: &TreeNode<MathLabel>,
    goal: &RecExpr<Math>,
    rules: &[Rewrite<Math, ConstantFold>],
    max_iters: usize,
) -> Option<usize> {
    let guide_recexpr = guide
        .to_string()
        .parse::<RecExpr<Math>>()
        .expect("guide round-trips to RecExpr");

    let goal_clone = goal.clone();

    let runner = Runner::<Math, ConstantFold>::default()
        .with_scheduler(SimpleScheduler)
        .with_iter_limit(max_iters)
        .with_expr(&guide_recexpr)
        .with_hook(move |runner| {
            if runner.egraph.lookup_expr(&goal_clone).is_some() {
                return Err("goal found".to_owned());
            }
            Ok(())
        })
        .run(rules);

    if runner.egraph.lookup_expr(goal).is_some() {
        return Some(runner.iterations.len());
    }

    None
}

#[expect(clippy::cast_precision_loss, clippy::too_many_lines)]
fn print_summary(
    results: &[VerifyResult],
    goal: &TreeNode<MathLabel>,
    max_iters: usize,
    // top: usize,
    log: &mut File,
) {
    let mut successful = results.iter().filter(|r| r.reached).collect::<Vec<_>>();
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
                    .filter_map(|v| v.iterations_to_reach)
                    .min()
                    .unwrap(),
                successful[successful.len() / 2]
                    .iterations_to_reach
                    .unwrap(),
                successful
                    .iter()
                    .filter_map(|v| v.iterations_to_reach)
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
                    .filter_map(|v| v.iterations_to_reach)
                    .min()
                    .unwrap(),
                successful[successful.len() / 2]
                    .iterations_to_reach
                    .unwrap(),
                successful
                    .iter()
                    .filter_map(|v| v.iterations_to_reach)
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

    // // Breakdown by iterations-to-reach (None = unreached)
    // let by_iters = results.iter().fold(
    //     BTreeMap::<Option<usize>, (Vec<_>, Vec<_>)>::new(),
    //     |mut acc, r| {
    //         let (ranks, sizes) = acc.entry(r.iterations_to_reach).or_default();
    //         ranks.push(r.rank);
    //         sizes.push(r.guide.size_without_types());
    //         acc
    //     },
    // );
    // log!(log, "\n  Breakdown by iterations to reach:");
    // log!(
    //     log,
    //     "    {:>5}  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
    //     "iters",
    //     "count",
    //     "min_rank",
    //     "med_rank",
    //     "max_rank",
    //     "min_size",
    //     "med_size",
    //     "max_size"
    // );
    // for (iters_key, (mut ranks, mut sizes)) in by_iters {
    //     ranks.sort_unstable();
    //     sizes.sort_unstable();
    //     let label = iters_key.map_or_else(|| "none".to_owned(), |i| i.to_string());
    //     let count = ranks.len();
    //     let min_r = ranks[0];
    //     let med_r = ranks[count / 2];
    //     let max_r = ranks[count - 1];
    //     let min_s = sizes[0];
    //     let med_s = sizes[count / 2];
    //     let max_s = sizes[count - 1];
    //     log!(
    //         log,
    //         "    {label:>5}  {count:>5}  {min_r:>8}  {med_r:>8}  {max_r:>8}  {min_s:>8}  {med_s:>8}  {max_s:>8}"
    //     );
    // }

    // // Top-N guide table
    // log!(
    //     log,
    //     "\n  Top guides (rank / dist / reached / iters / term):"
    // );
    // for r in results.iter().take(top) {
    //     let reached_marker = if r.reached { "✓" } else { "✗" };
    //     let iters_str = r
    //         .iterations_to_reach
    //         .map_or_else(|| "-".to_owned(), |i| i.to_string());
    //     log!(
    //         log,
    //         "    {:>3}  dist={:<4}  {}  iters={:<4}  {}",
    //         r.rank,
    //         r.zs_distance,
    //         reached_marker,
    //         iters_str,
    //         r.guide
    //     );
    // }
}

// /// Compute the Spearman rank correlation between guide rank and iterations to reach the goal.
// ///
// /// Only guides that actually reached the goal are included. Returns 0.0 when
// /// fewer than 2 data points are available.
// #[expect(clippy::cast_precision_loss, clippy::float_cmp, clippy::similar_names)]
// fn rank_iterations_correlation(results: &[VerifyResult]) -> f64 {
//     fn to_ranks(vals: &[f64]) -> Vec<f64> {
//         let mut indexed = vals.iter().copied().enumerate().collect::<Vec<_>>();
//         indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
//         let mut ranks = vec![0.0; vals.len()];
//         let mut i = 0;
//         while i < indexed.len() {
//             let mut j = i;
//             while j < indexed.len() && indexed[j].1 == indexed[i].1 {
//                 j += 1;
//             }
//             let avg_rank = (i + j + 1) as f64 / 2.0;
//             for item in &indexed[i..j] {
//                 ranks[item.0] = avg_rank;
//             }
//             i = j;
//         }
//         ranks
//     }

//     let pairs = results
//         .iter()
//         .filter_map(|r| Some((r.rank as f64, r.iterations_to_reach? as f64)))
//         .collect::<Vec<_>>();

//     if pairs.len() < 2 {
//         return 0.0;
//     }

//     let xs = pairs.iter().map(|(x, _)| *x).collect::<Vec<_>>();
//     let ys = pairs.iter().map(|(_, y)| *y).collect::<Vec<_>>();
//     let rx = to_ranks(&xs);
//     let ry = to_ranks(&ys);

//     let n = rx.len() as f64;
//     let mean_rx = rx.iter().sum::<f64>() / n;
//     let mean_ry = ry.iter().sum::<f64>() / n;

//     let (cov, var_x, var_y) =
//         rx.iter()
//             .zip(&ry)
//             .fold((0.0, 0.0, 0.0), |(cov, var_x, var_y), (rx_i, ry_i)| {
//                 let dx = rx_i - mean_rx;
//                 let dy = ry_i - mean_ry;
//                 (cov + dx * dy, var_x + dx * dx, var_y + dy * dy)
//             });

//     let denom = (var_x * var_y).sqrt();
//     if denom < f64::EPSILON {
//         0.0
//     } else {
//         cov / denom
//     }
// }

// struct LinearRegression {
//     slope: f64,
//     intercept: f64,
//     r_sq: f64,
//     se_slope: f64,
// }

// /// Fit a simple linear regression (OLS) to `(x, y)` pairs.
// /// Returns `None` if fewer than 2 points or zero variance in x.
// #[expect(clippy::cast_precision_loss, clippy::similar_names)]
// fn linear_regression(points: &[(f64, f64)]) -> Option<LinearRegression> {
//     let n = points.len() as f64;
//     if n < 2.0 {
//         return None;
//     }

//     let sum_x = points.iter().map(|(x, _)| x).sum::<f64>();
//     let sum_y = points.iter().map(|(_, y)| y).sum::<f64>();
//     let sum_xy = points.iter().map(|(x, y)| x * y).sum::<f64>();
//     let sum_x2 = points.iter().map(|(x, _)| x * x).sum::<f64>();
//     let denom = n * sum_x2 - sum_x * sum_x;
//     if denom.abs() <= f64::EPSILON {
//         return None;
//     }

//     let slope = (n * sum_xy - sum_x * sum_y) / denom;
//     let intercept = (sum_y - slope * sum_x) / n;

//     let mean_y = sum_y / n;
//     let ss_res = points
//         .iter()
//         .map(|(x, y)| {
//             let pred = slope * x + intercept;
//             (y - pred).powi(2)
//         })
//         .sum::<f64>();
//     let ss_tot = points
//         .iter()
//         .map(|(_, y)| (y - mean_y).powi(2))
//         .sum::<f64>();
//     let r_sq = if ss_tot.abs() > f64::EPSILON {
//         1.0 - ss_res / ss_tot
//     } else {
//         0.0
//     };
//     let se_slope = if n > 2.0 {
//         let mse = ss_res / (n - 2.0);
//         let mean_x = sum_x / n;
//         let ss_xx = points
//             .iter()
//             .map(|(x, _)| (x - mean_x).powi(2))
//             .sum::<f64>();
//         if ss_xx.abs() > f64::EPSILON {
//             (mse / ss_xx).sqrt()
//         } else {
//             0.0
//         }
//     } else {
//         0.0
//     };

//     Some(LinearRegression {
//         slope,
//         intercept,
//         r_sq,
//         se_slope,
//     })
// }

// /// Plot rank vs iterations-to-reach as a scatter plot and save as PNG.
// #[expect(clippy::cast_precision_loss, clippy::too_many_lines)]
// fn plot_rank_vs_iterations(
//     results: &[VerifyResult],
//     goal: &TreeNode<MathLabel>,
//     path: &PathBuf,
//     cfg: &PlotConfig,
//     log: &mut File,
// ) {
//     let reached = results
//         .iter()
//         .filter_map(|r| Some((r.rank as f64, r.iterations_to_reach? as f64)))
//         .collect::<Vec<_>>();
//     let timed_out = results
//         .iter()
//         .filter(|r| !r.reached)
//         .map(|r| r.rank as f64)
//         .collect::<Vec<_>>();

//     if reached.len() + timed_out.len() < 2 {
//         log!(log, "  Not enough guides to plot.");
//         return;
//     }

//     let max_rank = results
//         .iter()
//         .map(|r| r.rank as f64)
//         .fold(0.0_f64, f64::max);
//     let verify_iters_f = cfg.verify_iters as f64;
//     let timeout_y = verify_iters_f + 1.0;
//     let max_iters = reached.iter().map(|(_, y)| *y).fold(timeout_y, f64::max);

//     let root = BitMapBackend::new(path, (1600, 1280)).into_drawing_area();
//     root.fill(&WHITE).expect("fill background");

//     let corr = rank_iterations_correlation(results);
//     let guides_str = cfg
//         .num_guides
//         .map_or_else(|| "all".to_owned(), |g| g.to_string());

//     // Compute regression early so we can show stats in the header
//     let regression = linear_regression(&reached);
//     let n_timeout = timed_out.len();
//     let stats_line = regression.as_ref().map_or_else(
//         || format!("r = {corr:.3} | n = {} reached, {n_timeout} timed out", reached.len()),
//         |reg| {
//             format!(
//                 "R² = {:.3} | slope = {:.3} ± {:.3} | r = {corr:.3} | n = {} reached, {n_timeout} timed out",
//                 reg.r_sq, reg.slope, reg.se_slope, reached.len()
//             )
//         },
//     );

//     // Build iteration breakdown line
//     let mut iter_counts = BTreeMap::<String, usize>::new();
//     for r in results {
//         let key = r
//             .iterations_to_reach
//             .map_or_else(|| "none".to_owned(), |i| i.to_string());
//         *iter_counts.entry(key).or_default() += 1;
//     }
//     let iter_line = iter_counts
//         .iter()
//         .map(|(k, v)| format!("i{k}={v}"))
//         .collect::<Vec<_>>()
//         .join(" | ");

//     // Split into title area + chart area
//     let (title_area, chart_area) = root.split_vertically(140);

//     // Draw title and subtitle
//     let meta_style = ("sans-serif", 24).into_font().color(&GREY);
//     title_area
//         .draw_text(
//             "Rank vs Iterations to Reach Goal",
//             &("sans-serif", 32).into_font().color(&BLACK),
//             (10, 2),
//         )
//         .expect("draw title");
//     let subtitle = format!(
//         "{} | {} | {} | guides={} | goal_i={} guide_i={} | goal_size={}",
//         cfg.strategy,
//         cfg.distance,
//         cfg.distribution,
//         guides_str,
//         cfg.goal_iters,
//         cfg.guide_iters,
//         goal.size_without_types(),
//     );
//     title_area
//         .draw_text(&subtitle, &meta_style, (10, 38))
//         .expect("draw subtitle");
//     title_area
//         .draw_text(&stats_line, &meta_style, (10, 68))
//         .expect("draw stats line");
//     title_area
//         .draw_text(&iter_line, &meta_style, (10, 98))
//         .expect("draw iter breakdown");

//     let mut chart = ChartBuilder::on(&chart_area)
//         .margin(10)
//         .x_label_area_size(35)
//         .y_label_area_size(45)
//         .build_cartesian_2d(0.0..(max_rank * 1.05), 0.0..(max_iters * 1.05))
//         .expect("build chart");

//     chart
//         .configure_mesh()
//         .x_desc("Rank (by distance)")
//         .y_desc("Iterations to reach goal")
//         .draw()
//         .expect("draw mesh");

//     // Draw reached guides as blue circles
//     chart
//         .draw_series(
//             reached
//                 .iter()
//                 .map(|&(x, y)| Circle::new((x, y), 4, BLUE.filled())),
//         )
//         .expect("draw reached points")
//         .label("reached")
//         .legend(|(x, y)| Circle::new((x, y), 4, BLUE.filled()));

//     // Draw timed-out guides as red crosses at y = verify_iters
//     if !timed_out.is_empty() {
//         chart
//             .draw_series(
//                 timed_out
//                     .iter()
//                     .map(|&x| Cross::new((x, timeout_y), 4, RED.filled())),
//             )
//             .expect("draw timed-out points")
//             .label(format!("timed out (>{} iters)", cfg.verify_iters))
//             .legend(|(x, y)| Cross::new((x, y), 4, RED.filled()));

//         // Draw a dashed line at verify_iters to show the cutoff
//         chart
//             .draw_series(DashedLineSeries::new(
//                 vec![(0.0, verify_iters_f), (max_rank * 1.05, verify_iters_f)],
//                 5,
//                 3,
//                 RED.mix(0.4).stroke_width(1),
//             ))
//             .expect("draw cutoff line");
//     }

//     chart
//         .configure_series_labels()
//         .position(SeriesLabelPosition::LowerRight)
//         .background_style(WHITE.mix(0.8))
//         .border_style(BLACK)
//         .label_font(("sans-serif", 24).into_font().color(&GREY))
//         .draw()
//         .expect("draw legend");

//     // Draw regression line
//     if let Some(reg) = &regression {
//         let x0 = 0.0_f64;
//         let x1 = max_rank * 1.05;
//         let y0 = (reg.intercept + reg.slope * x0).clamp(0.0, max_iters * 1.05);
//         let y1 = (reg.intercept + reg.slope * x1).clamp(0.0, max_iters * 1.05);
//         chart
//             .draw_series(LineSeries::new(
//                 vec![(x0, y0), (x1, y1)],
//                 GREEN.stroke_width(2),
//             ))
//             .expect("draw regression line");
//     }

//     root.present().expect("present plot");
//     log!(log, "  Plot saved to {}", path.display());
// }
