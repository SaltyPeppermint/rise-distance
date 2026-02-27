use std::time::Instant;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite, Runner, SimpleScheduler};
use num::BigUint;

use rise_distance::cli::{DistanceMetric, SampleDistribution};
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
    goal_iteration: usize,

    /// Number of eqsat iterations to grow the egraph to reach the guide
    #[arg(short = 'i', long)]
    guide_iteration: usize,

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
    #[arg(short, long)] // default_value_t = DistanceMetric::ZhangShasha
    distance: DistanceMetric,

    /// How to distribute the sample budget across sizes.
    /// Options: uniform, proportional:<`min_per_size`>, normal:<sigma>
    #[arg(short = 'p', long, default_value_t = SampleDistribution::Uniform, conflicts_with = "enumerate")]
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

    /// Enumerate all terms exhaustively instead of sampling.
    /// Only feasible for small egraphs.
    #[arg(long)]
    enumerate: bool,
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
    let rules = rise_distance::egg::math::rules();
    let verify_iters = cli.verify_iters.unwrap_or(cli.goal_iteration);

    let seed = cli
        .seed
        .parse::<RecExpr<Math>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed: {e}"));

    eprintln!("Seed: {seed}");
    eprintln!("Goal Iterations: {}", cli.goal_iteration);
    eprintln!("Guide Iterations: {}", cli.guide_iteration);
    eprintln!("Distance metric: {}", cli.distance);
    eprintln!("Distribution: {}", cli.distribution);

    // Step 1: Grow egraph and capture snapshots at guide and target iterations
    eprintln!(
        "Running equality saturation for {} iterations...",
        cli.goal_iteration
    );
    let start = Instant::now();
    let [(prev_eg_guide, eg_guide), (prev_eg_goal, eg_goal)] =
        run_guide_goal::<Math, ConstantFold, MathLabel, _>(
            &seed,
            &rules,
            cli.guide_iteration,
            cli.goal_iteration,
        );
    eprintln!("Eqsat completed in {:.2?}", start.elapsed());

    // Step 2: Sample goals from the iteration-n frontier
    eprintln!(
        "Sampling goals from iteration-{} frontier...",
        cli.goal_iteration
    );
    let max_size = cli.max_size.unwrap_or_else(|| {
        let k = AstSize.cost_rec(&seed);
        let m = k + k / 2;
        eprintln!("No max_size was given, using 1.5 goal size (={m})");
        m
    });
    let goals = get_frontier_terms(
        &eg_goal,
        &prev_eg_goal,
        cli.goals,
        max_size,
        cli.distribution,
        false,
    );
    if goals.is_empty() {
        eprintln!(
            "No frontier terms found at iteration {}. Try more iterations or a larger max-size.",
            cli.goal_iteration
        );
        return;
    }
    eprintln!("Sampled {} goal(s)", goals.len());

    // Step 3: Get guides from the iteration-(n-1) frontier
    eprintln!(
        "Getting guides from iteration-{} frontier...",
        cli.guide_iteration
    );
    let guide_count = if cli.enumerate {
        usize::MAX
    } else {
        cli.guides.expect("-g/--guides is required when not using --enumerate")
    };
    let guides = get_frontier_terms(
        &eg_guide,
        &prev_eg_guide,
        guide_count,
        max_size,
        cli.distribution,
        cli.enumerate,
    );
    if guides.is_empty() {
        eprintln!(
            "No frontier terms found at iteration {}.",
            cli.guide_iteration
        );
        return;
    }
    eprintln!("Found {} guide(s)", guides.len());

    // Step 4 & 5: For each goal, rank guides by distance, then verify reachability
    let cfg = EvalConfig {
        guides: &guides,
        rules: &rules,
        metric: cli.distance,
        verify_iters,
    };

    let csv_writer = cli.output.as_deref().map(|path| {
        let file = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("Failed to create '{path}': {e}"));
        csv::Writer::from_writer(file)
    });

    run_eval(&goals, &cfg, csv_writer, cli.top);
}

fn run_eval(
    goals: &[TreeNode<MathLabel>],
    cfg: &EvalConfig<'_>,
    mut csv: Option<csv::Writer<std::fs::File>>,
    top: usize,
) {
    if let Some(w) = csv.as_mut() {
        w.write_record([
            "goal",
            "rank",
            "distance",
            "reached",
            "iterations_to_reach",
            "guide",
        ])
        .expect("write CSV header");
    }
    for (goal_idx, goal) in goals.iter().enumerate() {
        evaluate_goal(goal, goal_idx, goals.len(), cfg, csv.as_mut(), top);
    }
    if let Some(w) = csv.as_mut() {
        w.flush().expect("flush CSV");
    }
}

struct EvalConfig<'a> {
    guides: &'a [TreeNode<MathLabel>],
    rules: &'a [Rewrite<Math, ConstantFold>],
    metric: DistanceMetric,
    verify_iters: usize,
}

fn evaluate_goal(
    goal: &TreeNode<MathLabel>,
    goal_idx: usize,
    total_goals: usize,
    cfg: &EvalConfig<'_>,
    mut csv: Option<&mut csv::Writer<std::fs::File>>,
    top: usize,
) {
    eprintln!(
        "\n=== Goal {}/{}: {} (size {}) ===",
        goal_idx + 1,
        total_goals,
        goal,
        goal.size_without_types()
    );

    let ranked = rank_guides(cfg.guides, goal, cfg.metric);
    eprintln!(
        "Verifying {} guides (max {} iters each)...",
        ranked.len(),
        cfg.verify_iters
    );

    let timer = Instant::now();
    let goal_recexpr = goal
        .to_string()
        .parse::<RecExpr<Math>>()
        .expect("goal round-trips to RecExpr");

    let goal_str = goal.to_string();
    let mut results = Vec::with_capacity(ranked.len());
    for (rank, (dist_val, guide)) in ranked.into_iter().enumerate() {
        let iters = verify_reachability(&guide, &goal_recexpr, cfg.rules, cfg.verify_iters);
        let reached = iters.is_some();

        if let Some(w) = csv.as_deref_mut() {
            w.write_record([
                &goal_str,
                &(rank + 1).to_string(),
                &dist_val.to_string(),
                &reached.to_string(),
                &iters.map_or_else(String::new, |i| i.to_string()),
                &guide.to_string(),
            ])
            .expect("write CSV row");
        }

        results.push(VerifyResult {
            rank: rank + 1,
            distance: dist_val,
            reached,
            iterations_to_reach: iters,
            guide,
        });
    }
    eprintln!("Verification completed in {:.2?}", timer.elapsed());

    print_summary(&results, goal, cfg.verify_iters, top);
}

/// Get frontier terms from `egraph` that are NOT present in `prev_raw_egg`.
///
/// When `enumerate` is true, enumerates all terms exhaustively.
/// Otherwise, samples terms using the given distribution.
fn get_frontier_terms(
    egraph: &EGraph<MathLabel>,
    prev_raw_egg: &egg::EGraph<Math, ConstantFold>,
    count: usize,
    max_size: usize,
    distribution: SampleDistribution,
    enumerate: bool,
) -> Vec<TreeNode<MathLabel>> {
    let tc = TermCount::<BigUint, MathLabel>::new(max_size, false, egraph);

    let Some(histogram) = tc.of_root() else {
        return Vec::new();
    };

    let mut sorted_hist = histogram.iter().collect::<Vec<_>>();
    sorted_hist.sort_unstable();
    for (k, v) in &sorted_hist {
        eprintln!("{v} terms of size {k}");
    }

    let is_frontier = |tree: &TreeNode<MathLabel>| {
        let Ok(recexpr) = tree.to_string().parse::<RecExpr<Math>>() else {
            return false;
        };
        prev_raw_egg.lookup_expr(&recexpr).is_none()
    };

    if enumerate {
        let total_terms: BigUint = histogram.values().cloned().sum();
        eprintln!("Enumerating all {total_terms} terms up to size {max_size}");
        tc.enumerate_root(max_size)
            .filter(is_frontier)
            .take(count)
            .collect()
    } else {
        let min_size = histogram.keys().min().copied().unwrap_or(1);
        // Oversample 5x to account for rejection filtering
        let total_samples = count * 5;

        #[expect(clippy::cast_precision_loss)]
        let normal_center = (min_size + max_size) as f64 / 2.0;
        let samples_per_size = distribution.samples_per_size(
            histogram,
            min_size,
            max_size,
            total_samples,
            normal_center,
        );

        tc.sample_unique_root(min_size, max_size, &samples_per_size)
            .into_iter()
            .filter(is_frontier)
            .take(count)
            .collect()
    }
}

/// Rank guides by distance to the goal. Returns `(distance, guide)` pairs sorted ascending.
fn rank_guides(
    guides: &[TreeNode<MathLabel>],
    goal: &TreeNode<MathLabel>,
    metric: DistanceMetric,
) -> Vec<(usize, TreeNode<MathLabel>)> {
    let goal_flat = goal.flatten(false);
    let mut ranked = guides
        .iter()
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

#[expect(clippy::cast_precision_loss)]
fn print_summary(
    results: &[VerifyResult],
    goal: &TreeNode<MathLabel>,
    max_iters: usize,
    top: usize,
) {
    let successful = results.iter().filter(|r| r.reached).collect::<Vec<_>>();
    let total = results.len();
    let n_reached = successful.len();

    eprintln!("\nSummary for goal: {goal}");
    eprintln!("  Goal size: {}", goal.size_without_types());
    eprintln!("  Total guides evaluated: {total}");
    eprintln!("  Max verify iterations: {max_iters}");
    eprintln!(
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

        eprintln!(
            "  Successful guide ranks:      min={}, median={}, max={}",
            ranks.first().unwrap(),
            ranks[ranks.len() / 2],
            ranks.last().unwrap()
        );
        eprintln!(
            "  Successful guide distances:  min={}, median={}, max={}",
            dists.iter().min().unwrap(),
            dists[dists.len() / 2],
            dists.iter().max().unwrap()
        );
        if !iters.is_empty() {
            eprintln!(
                "  Iterations to reach:         min={}, median={}, max={}",
                iters.iter().min().unwrap(),
                iters[iters.len() / 2],
                iters.iter().max().unwrap()
            );
        }

        // First successful rank (how far down the ranking before finding a viable guide)
        eprintln!(
            "  First viable guide at rank: {} (distance {})",
            successful[0].rank, successful[0].distance
        );
    }

    // Top-N guide table
    eprintln!("\n  Top guides (rank / dist / reached / iters / term):");
    for r in results.iter().take(top) {
        let reached_marker = if r.reached { "✓" } else { "✗" };
        let iters_str = r
            .iterations_to_reach
            .map_or_else(|| "-".to_owned(), |i| i.to_string());
        eprintln!(
            "    {:>3}  dist={:<4}  {}  iters={:<4}  {}",
            r.rank, r.distance, reached_marker, iters_str, r.guide
        );
    }
}
