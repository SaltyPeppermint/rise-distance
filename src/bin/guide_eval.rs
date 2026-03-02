use std::collections::BTreeMap;
use std::env::current_dir;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use clap::Parser;
use csv::Writer;
use egg::{AstSize, CostFunction, RecExpr, Rewrite, Runner, SimpleScheduler};
use hashbrown::HashMap;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::BigUint;
use rayon::prelude::*;

use rise_distance::cli::{DistanceMetric, SampleDistribution, SampleStrategy, log};
use rise_distance::egg::math::{ConstantFold, Math, MathLabel};
use rise_distance::egg::run_guide_goal;
use rise_distance::{EGraph, TermCount, TreeNode, UnitCost, structural_diff, tree_distance_unit};

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

    /// Distance metric to use for ranking
    #[arg(long)] // default_value_t = DistanceMetric::ZhangShasha
    distance: DistanceMetric,

    /// How to distribute the sample budget across sizes.
    /// Options: uniform, proportional:<`min_per_size`>, normal:<sigma>
    #[arg(long, default_value_t = SampleDistribution::Uniform)]
    distribution: SampleDistribution,

    /// CSV output file (stdout if omitted)
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

struct VerifyResult {
    rank: usize,
    distance: usize,
    reached: bool,
    iterations_to_reach: Option<usize>,
    guide: TreeNode<MathLabel>,
}

fn main() {
    let cli = Cli::parse();
    let mut log = output_file_name(&cli, ".log");

    let rules = rise_distance::egg::math::rules();
    let verify_iters = cli.verify_iters.unwrap_or(cli.goal_iters);

    let seed = cli
        .seed
        .parse::<RecExpr<Math>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed: {e}"));

    log!(log, "Seed: {seed}");
    log!(log, "Goal Iterations: {}", cli.goal_iters);
    log!(log, "Guide Iterations: {}", cli.guide_iters);
    log!(log, "Distance metric: {}", cli.distance);
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
    run_eval(
        &guides_per_goal,
        &rules,
        cli.distance,
        verify_iters,
        cli.top,
        output_file_name(&cli, ".csv"),
        &mut log,
    );
    log.flush().unwrap();
}

/// Create an output file for this run.
///
/// If `-o` was given, uses that path directly. Otherwise, auto-generates a
/// filename like `run-{guide_iters}-{goal_iters}-sampling_{N}{ending}` in the
/// current directory, where `N` is one higher than the largest existing run number.
/// `ending` should include the leading dot (e.g. `".csv"`).
fn output_file_name(cli: &Cli, ending: &str) -> File {
    cli.output.as_deref().map_or_else(
        || {
            let pat = format!("run-{}-{}-sampling", cli.guide_iters, cli.goal_iters);
            let max_existing = current_dir()
                .unwrap()
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
            let path = current_dir().unwrap().join(pat);
            File::create(path).unwrap()
        },
        |path| File::create(path).expect("Failed to create output csv"),
    )
}

fn run_eval(
    guides_for_goal: &HashMap<TreeNode<MathLabel>, Vec<TreeNode<MathLabel>>>,
    rules: &[Rewrite<Math, ConstantFold>],
    metric: DistanceMetric,
    verify_iters: usize,
    top: usize,
    output: File,
    log: &mut File,
) {
    let mut csv_writer = Writer::from_writer(output);
    csv_writer
        .write_record([
            "goal",
            "rank",
            "distance",
            "reached",
            "iterations_to_reach",
            "guide",
        ])
        .expect("write CSV header");

    for (goal_idx, (goal, guides)) in guides_for_goal.iter().enumerate() {
        log!(
            log,
            "\n=== Goal {}/{} (size {})===\n",
            goal_idx + 1,
            guides_for_goal.len(),
            goal.size_without_types()
        );

        let ranked = rank_guides(guides, goal, metric);
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
            .enumerate()
            .progress_with_style(pb_style)
            .map(|(rank, (dist_val, guide))| {
                let iters = verify_reachability(&guide, &goal_recexpr, rules, verify_iters);
                VerifyResult {
                    rank: rank + 1,
                    distance: dist_val,
                    reached: iters.is_some(),
                    iterations_to_reach: iters,
                    guide,
                }
            })
            .collect::<Vec<_>>();

        log!(log, "Verification completed in {:.2?}", timer.elapsed());

        let goal_str = goal.to_string();
        for r in &results {
            csv_writer
                .write_record([
                    &goal_str,
                    &r.rank.to_string(),
                    &r.distance.to_string(),
                    &r.reached.to_string(),
                    &r.iterations_to_reach
                        .map_or_else(String::new, |i| i.to_string()),
                    &r.guide.to_string(),
                ])
                .expect("write CSV row");
        }

        print_summary(&results, goal, verify_iters, top, log);
    }
    csv_writer.flush().expect("flush CSV");
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

/// Rank guides by distance to the goal. Returns `(distance, guide)` pairs sorted ascending.
fn rank_guides(
    guides: &[TreeNode<MathLabel>],
    goal: &TreeNode<MathLabel>,
    metric: DistanceMetric,
) -> Vec<(usize, TreeNode<MathLabel>)> {
    let goal_flat = goal.flatten(false);
    let pb_style = ProgressStyle::with_template(
        "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}] ranking guides",
    )
    .unwrap();
    let mut ranked = guides
        .par_iter()
        .progress_with_style(pb_style)
        .map(|guide| {
            let guide_flat = guide.flatten(false);
            let dist = match metric {
                DistanceMetric::ZhangShasha => tree_distance_unit(&guide_flat, &goal_flat),
                DistanceMetric::Structural => {
                    structural_diff(&goal_flat, &guide_flat, &UnitCost).zs_sum()
                }
            };
            (dist, guide.clone())
        })
        .collect::<Vec<_>>();
    ranked.sort_by_key(|(d, _)| *d);
    ranked
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
    top: usize,
    log: &mut File,
) {
    let successful = results.iter().filter(|r| r.reached).collect::<Vec<_>>();
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
        let ranks = successful.iter().map(|r| r.rank).collect::<Vec<_>>();
        let dists = successful.iter().map(|r| r.distance).collect::<Vec<_>>();
        let iters = successful
            .iter()
            .filter_map(|r| r.iterations_to_reach)
            .collect::<Vec<_>>();

        log!(
            log,
            "  Successful guide ranks:      min={}, median={}, max={}",
            ranks.first().unwrap(),
            ranks[ranks.len() / 2],
            ranks.last().unwrap()
        );
        log!(
            log,
            "  Successful guide distances:  min={}, median={}, max={}",
            dists.iter().min().unwrap(),
            dists[dists.len() / 2],
            dists.iter().max().unwrap()
        );
        if !iters.is_empty() {
            log!(
                log,
                "  Iterations to reach:         min={}, median={}, max={}",
                iters.iter().min().unwrap(),
                iters[iters.len() / 2],
                iters.iter().max().unwrap()
            );
        }

        // First successful rank (how far down the ranking before finding a viable guide)
        log!(
            log,
            "  First viable guide at rank: {} (distance {})",
            successful[0].rank,
            successful[0].distance
        );
    }

    // Breakdown by iterations-to-reach (None = unreached)
    let mut by_iters: BTreeMap<Option<usize>, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    for r in results {
        let (ranks, sizes) = by_iters.entry(r.iterations_to_reach).or_default();
        ranks.push(r.rank);
        sizes.push(r.guide.size_without_types());
    }
    log!(log, "\n  Breakdown by iterations to reach:");
    log!(
        log,
        "    {:>5}  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "iters",
        "count",
        "min_rank",
        "med_rank",
        "max_rank",
        "min_size",
        "med_size",
        "max_size"
    );
    for (iters_key, (mut ranks, mut sizes)) in by_iters {
        ranks.sort_unstable();
        sizes.sort_unstable();
        let label = iters_key.map_or_else(|| "none".to_owned(), |i| i.to_string());
        let count = ranks.len();
        let min_r = ranks[0];
        let med_r = ranks[count / 2];
        let max_r = ranks[count - 1];
        let min_s = sizes[0];
        let med_s = sizes[count / 2];
        let max_s = sizes[count - 1];
        log!(
            log,
            "    {label:>5}  {count:>5}  {min_r:>8}  {med_r:>8}  {max_r:>8}  {min_s:>8}  {med_s:>8}  {max_s:>8}"
        );
    }

    // Top-N guide table
    log!(
        log,
        "\n  Top guides (rank / dist / reached / iters / term):"
    );
    for r in results.iter().take(top) {
        let reached_marker = if r.reached { "✓" } else { "✗" };
        let iters_str = r
            .iterations_to_reach
            .map_or_else(|| "-".to_owned(), |i| i.to_string());
        log!(
            log,
            "    {:>3}  dist={:<4}  {}  iters={:<4}  {}",
            r.rank,
            r.distance,
            reached_marker,
            iters_str,
            r.guide
        );
    }
}
