use std::env::current_dir;
use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite, Runner, SimpleScheduler};
use hashbrown::HashSet;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::{BigUint, ToPrimitive};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use rise_distance::cli::{SampleDistribution, SampleStrategy, log};
use rise_distance::egg::math::{ConstantFold, Math, MathLabel};
use rise_distance::egg::run_guide_goal;
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

    // /// Distance metric to use for ranking
    // #[arg(long)] // default_value_t = DistanceMetric::ZhangShasha
    // distance: DistanceMetric,
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
}

#[derive(Serialize, Debug)]
struct VerifyResult {
    guide: RankedGuide,
    iterations_to_reach: Option<usize>,
}

fn main() {
    let cli = Cli::parse();
    let out_folder = output_folder(&cli);
    let mut log = File::create(out_folder.join("out.log")).expect("Failed to create output file");

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
        &eg_guide,
        &prev_eg_guide,
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
        &rules,
        verify_iters,
        File::create(out_folder.join("out.csv")).expect("Failed to create output file"),
        &mut log,
    );
    log.flush().unwrap();
}

/// Create an output folder for this run.
///
/// If `-o` was given, uses that path directly. Otherwise, auto-generates a
/// filename like `runs/run-{goal_iters}-{strategy}-sampling_{N}`
/// inside a `output/` directory (created if needed), where `N` is one higher
/// than the largest existing run number.
fn output_folder(cli: &Cli) -> PathBuf {
    cli.output.as_deref().map_or_else(
        || {
            let runs_dir = current_dir().unwrap().join("data").join("guide_eval");
            std::fs::create_dir_all(&runs_dir).expect("Failed to create output directory");
            let pat: OsString = format!(
                "run-{}-{}-{}-sampling",
                cli.guide_iters, cli.goal_iters, cli.strategy
            )
            .into();
            let max_existing = runs_dir
                .read_dir()
                .unwrap()
                .filter_map(|e| {
                    let d = e.ok()?;
                    if d.file_type().ok()?.is_dir() && pat.as_os_str() == d.path().file_stem()? {
                        return d.path().extension()?.to_str()?.parse::<usize>().ok();
                    }
                    None
                })
                .max()
                .unwrap_or(0);
            let this_run_dir = runs_dir
                .join(pat)
                .with_extension((max_existing + 1).to_string());
            std::fs::create_dir_all(&this_run_dir).expect("Failed to create output directory");
            this_run_dir
        },
        |c| {
            let this_run_dir = PathBuf::from(c);
            std::fs::create_dir_all(&this_run_dir).expect("Failed to create output directory");
            this_run_dir
        },
    )
}

fn run_eval(
    goals: &[TreeNode<MathLabel>],
    guides: &[TreeNode<MathLabel>],
    rules: &[Rewrite<Math, ConstantFold>],
    verify_iters: usize,
    output: File,
    log: &mut File,
) {
    let mut csv_writer = csv::Writer::from_writer(output);
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

        let ranked = rank_guides(guides, goal);
        log!(
            log,
            "Verifying {} guides (max {verify_iters} iters each)...",
            ranked.len(),
        );

        let timer = Instant::now();
        let goal_recexpr = goal.into();
        let pb_style = ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}] verifying guides",
        )
        .unwrap();

        let results = ranked
            .into_par_iter()
            .progress_with_style(pb_style)
            .map(|ranked_guide| {
                let iters = verify_reachability(
                    std::slice::from_ref(&ranked_guide.guide),
                    &goal_recexpr,
                    rules,
                    verify_iters,
                );
                VerifyResult {
                    guide: ranked_guide,
                    iterations_to_reach: iters,
                }
            })
            .collect::<Vec<_>>();

        log!(log, "Verification completed in {:.2?}", timer.elapsed());

        print_summary(&results, goal, verify_iters, log);
        verify_top_k(&results, goal, rules, verify_iters, log);

        for r in &results {
            csv_writer
                .serialize(CsvRow::new(goal, r))
                .expect("write CSV row");
        }
        log!(log, "Wrote goal {}/{} to CSV", goal_idx + 1, n_goals);
    }
    csv_writer.flush().expect("flush output");
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
}

impl CsvRow {
    fn new(goal: &TreeNode<MathLabel>, r: &VerifyResult) -> CsvRow {
        CsvRow {
            goal: goal.to_string(),
            guide: r.guide.guide.to_string(),
            zs_distance: r.guide.zs_distance,
            structural_overlap: r.guide.structural_distance.overlap(),
            structural_zs_sum: r.guide.structural_distance.zs_sum(),
            zs_rank: r.guide.zs_rank,
            structural_rank: r.guide.structural_rank,
            iterations_to_reach: r.iterations_to_reach,
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
        // SampleStrategy::Overlap => {
        //     let min_size = histogram.keys().min().copied().unwrap_or(1);
        //     // Oversample 5x to account for rejection filtering
        //     #[expect(clippy::cast_precision_loss)]
        //     let normal_center = (min_size + max_size) as f64 / 2.0;
        //     let samples_per_size = distribution.samples_per_size(
        //         histogram,
        //         min_size,
        //         max_size,
        //         count * 5,
        //         normal_center,
        //     );
        //     tc.sample_unique_root_overlap(goal, min_size, max_size, &samples_per_size)
        //         .into_iter()
        //         .filter(|t| is_frontier(t, prev_raw_egg))
        //         .take(count)
        //         .collect()
        // }
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

/// Helper function to check if something is in the frontier
fn is_frontier(tree: &TreeNode<MathLabel>, prev_raw_egg: &egg::EGraph<Math, ConstantFold>) -> bool {
    prev_raw_egg.lookup_expr(&tree.into()).is_none()
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

/// Run eqsat from `guides` (all unioned together) and check if `goal` becomes reachable.
/// Returns `Some(iteration)` if reached, `None` otherwise.
fn verify_reachability(
    guides: &[TreeNode<MathLabel>],
    goal: &RecExpr<Math>,
    rules: &[Rewrite<Math, ConstantFold>],
    max_iters: usize,
) -> Option<usize> {
    assert!(!guides.is_empty(), "must have at least one guide");

    let goal_clone = goal.clone();

    let mut runner = Runner::default()
        .with_scheduler(SimpleScheduler)
        .with_iter_limit(max_iters)
        .with_hook(move |runner| {
            if runner.egraph.lookup_expr(&goal_clone).is_some() {
                return Err("goal found".to_owned());
            }
            Ok(())
        });

    for expr in guides.iter().map(|g| g.into()) {
        runner = runner.with_expr(&expr);
    }

    // Union all guide roots together before running
    for &root in &runner.roots[1..] {
        runner.egraph.union(runner.roots[0], root);
    }
    runner.egraph.rebuild();

    let runner = runner.run(rules);

    runner
        .egraph
        .lookup_expr(goal)
        .map(|_| runner.iterations.len())
}

const TOP_K: [usize; 6] = [1, 2, 5, 10, 50, 100];

fn verify_top_k(
    results: &[VerifyResult],
    goal: &TreeNode<MathLabel>,
    rules: &[Rewrite<Math, ConstantFold>],
    max_iters: usize,
    // top: usize,
    log: &mut File,
) {
    let mut successful = results
        .iter()
        .filter(|r| r.iterations_to_reach.is_some())
        .collect::<Vec<_>>();
    log!(log, "Testing out top-k to see if that improves things");
    let go = goal.into();

    if !successful.is_empty() {
        successful.sort_unstable_by_key(|v| v.guide.zs_rank);
        log!(log, "ZS DISTANCE:");
        top_k(rules, max_iters, log, &successful, &go);
        log!(log, "STRUCTURAL DISTANCE:");
        successful.sort_unstable_by_key(|v| v.guide.structural_rank);
        top_k(rules, max_iters, log, &successful, &go);
        log!(log, "RANDOM:");
        successful.shuffle(&mut ChaCha12Rng::seed_from_u64(0));
        top_k(rules, max_iters, log, &successful, &go);
        log!(log, "KNOWN ITERATIONS:");
        successful.sort_unstable_by_key(|v| v.iterations_to_reach);
        top_k(rules, max_iters, log, &successful, &go);
    }
}

fn top_k(
    rules: &[Rewrite<Math, ConstantFold>],
    max_iters: usize,
    log: &mut File,
    successful: &[&VerifyResult],
    go: &RecExpr<Math>,
) {
    for k in TOP_K {
        let top_guides = successful[0..k]
            .iter()
            .map(|k| k.guide.guide.clone())
            .collect::<Vec<_>>();
        let could_reach = verify_reachability(&top_guides, go, rules, max_iters);
        let best_in_k = successful[0..k]
            .iter()
            .filter_map(|v| v.iterations_to_reach)
            .min();
        if let Some(i) = best_in_k {
            log!(log, "Best single guide in top {k} guides: {i}");
        } else {
            log!(log, "No single guide in top {k} could reach it");
        }
        if let Some(i) = could_reach {
            log!(log, "Could reach with top {k} guides: {i}");
        } else {
            log!(log, "Could NOT reach with top {k} guides");
        }
    }
}

#[expect(clippy::cast_precision_loss, clippy::too_many_lines)]
fn print_summary(
    results: &[VerifyResult],
    goal: &TreeNode<MathLabel>,
    max_iters: usize,
    // top: usize,
    log: &mut File,
) {
    let mut successful = results
        .iter()
        .filter(|r| r.iterations_to_reach.is_some())
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
}
