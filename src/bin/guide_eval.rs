use std::io::Write;
use std::time::Instant;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite, Runner, SimpleScheduler};
use hashbrown::HashMap;
use num::BigUint;

use rise_distance::egg::math::{ConstantFold, Math, MathLabel};
use rise_distance::egg::run_last_three_with_raw;
use rise_distance::{
    DistanceMetric, EGraph, TermCount, TreeNode, UnitCost, structural_diff, tree_distance_unit,
};

#[derive(Parser)]
#[command(
    about = "Evaluate distance metrics as guide predictors for equality saturation",
    after_help = "\
Examples:
  # Basic evaluation with Zhang-Shasha distance
  guide-eval -s '(d x (+ (* x x) 1))' -n 5 -g 10 --max-size 10 -d zhang-shasha

  # More guides and goals, structural distance metric
  guide-eval -s '(d x (+ (* x x) 1))' -n 5 -g 50 --goals 5 -d structural

  # Write CSV output to a file
  guide-eval -s '(d x (+ (* x x) 1))' -n 5 -g 10 -o results.csv

  # Limit verification iterations separately from growth iterations
  guide-eval -s '(d x (+ (* x x) 1))' -n 8 -g 20 --verify-iters 5
"
)]
struct Cli {
    /// Seed term as an s-expression (Math language)
    #[arg(short, long)]
    seed: String,

    /// Number of eqsat iterations to grow the egraph
    #[arg(short = 'n', long)]
    iterations: usize,

    /// Number of guide candidates to sample from the n-1 frontier
    #[arg(short, long)]
    guides: usize,

    /// Number of goal terms to sample from the n frontier
    #[arg(long, default_value_t = 1)]
    goals: usize,

    /// Max term size for counting/sampling (derived from egraph if omitted)
    #[arg(long)]
    max_size: Option<usize>,

    /// Distance metric to use for ranking
    #[arg(short, long)] // default_value_t = DistanceMetric::ZhangShasha
    distance: DistanceMetric,

    /// CSV output file (stdout if omitted)
    #[arg(short, long)]
    output: Option<String>,

    /// Max iterations when verifying guide reachability (defaults to -n value)
    #[arg(long)]
    verify_iters: Option<usize>,
}

struct VerifyResult {
    rank: usize,
    distance: usize,
    reached: bool,
    iterations_to_reach: Option<usize>,
}

fn main() {
    let cli = Cli::parse();
    let rules = rise_distance::egg::math::rules();
    let verify_iters = cli.verify_iters.unwrap_or(cli.iterations);

    let seed = cli
        .seed
        .parse::<RecExpr<Math>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed: {e}"));

    eprintln!("Seed: {seed}");
    eprintln!("Iterations: {}", cli.iterations);
    eprintln!("Distance metric: {}", cli.distance);

    // Step 1: Grow egraph and capture three consecutive snapshots + raw egg graphs
    eprintln!(
        "Running equality saturation for {} iterations...",
        cli.iterations
    );
    let start = Instant::now();
    let (converted_egs, raw_egs) =
        run_last_three_with_raw::<Math, ConstantFold, MathLabel, _>(&seed, &rules, cli.iterations);
    eprintln!("Eqsat completed in {:.2?}", start.elapsed());

    let [_, eg_n1, eg_n] = &converted_egs;
    let [raw_n2, raw_n1, _raw_n] = &raw_egs;

    // Step 2: Sample goals from the iteration-n frontier
    eprintln!(
        "Sampling goals from iteration-{} frontier...",
        cli.iterations
    );
    let max_size = cli.max_size.unwrap_or_else(|| {
        let k = AstSize.cost_rec(&seed);
        let m = k + k / 2;
        eprintln!("No max_size was given, using 1.5 goal size (={m})");
        m
    });
    let goals = sample_frontier_terms(eg_n, raw_n1, cli.goals, max_size);
    if goals.is_empty() {
        eprintln!(
            "No frontier terms found at iteration {}. Try more iterations or a larger max-size.",
            cli.iterations
        );
        return;
    }
    eprintln!("Sampled {} goal(s)", goals.len());

    // Step 3: Sample guides from the iteration-(n-1) frontier
    eprintln!(
        "Sampling guides from iteration-{} frontier...",
        cli.iterations - 1
    );
    let guides = sample_frontier_terms(eg_n1, raw_n2, cli.guides, max_size);
    if guides.is_empty() {
        eprintln!(
            "No frontier terms found at iteration {}.",
            cli.iterations - 1
        );
        return;
    }
    eprintln!("Sampled {} guide(s)", guides.len());

    // Step 4 & 5: For each goal, rank guides by distance, then verify reachability
    let cfg = EvalConfig {
        guides: &guides,
        rules: &rules,
        metric: cli.distance,
        verify_iters,
    };

    if let Some(path) = &cli.output {
        let file = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("Failed to create '{path}': {e}"));
        run_eval(&goals, &cfg, file);
    } else {
        run_eval(&goals, &cfg, std::io::stdout().lock());
    }
}

fn run_eval(goals: &[TreeNode<MathLabel>], cfg: &EvalConfig<'_>, mut writer: impl Write) {
    writeln!(
        writer,
        "goal,rank,distance,reached,iterations_to_reach,guide"
    )
    .expect("write header");
    for (goal_idx, goal) in goals.iter().enumerate() {
        evaluate_goal(goal, goal_idx, goals.len(), cfg, &mut writer);
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
    writer: &mut impl Write,
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

    let mut results = Vec::with_capacity(ranked.len());
    for (rank, (dist_val, guide)) in ranked.into_iter().enumerate() {
        let iters = verify_reachability(&guide, &goal_recexpr, cfg.rules, cfg.verify_iters);
        let reached = iters.is_some();
        let iters_str = iters.map_or_else(String::new, |i| i.to_string());

        writeln!(
            writer,
            "{goal},{rank},{dist_val},{reached},{iters_str},{guide}",
        )
        .expect("write row");

        results.push(VerifyResult {
            rank: rank + 1,
            distance: dist_val,
            reached,
            iterations_to_reach: iters,
        });
    }
    eprintln!("Verification completed in {:.2?}", timer.elapsed());

    print_summary(&results, goal, cfg.verify_iters);
}

/// Sample terms from `egraph` that are NOT present in `prev_raw_egg` (i.e. frontier terms).
fn sample_frontier_terms(
    egraph: &EGraph<MathLabel>,
    prev_raw_egg: &egg::EGraph<Math, ConstantFold>,
    count: usize,
    max_size: usize,
) -> Vec<TreeNode<MathLabel>> {
    let tc = TermCount::<BigUint, MathLabel>::new(max_size, false, egraph);

    let Some(histogram) = tc.of_root() else {
        return Vec::new();
    };

    let min_size = histogram.keys().min().copied().unwrap_or(1);

    // Uniform sampling across sizes, oversample to account for rejection
    let num_sizes = (max_size - min_size).max(1);
    let per_size = (count / num_sizes).max(1) as u64;
    let oversample_per_size = per_size.saturating_mul(5);
    let samples_per_size = (min_size..=max_size)
        .map(|s| (s, oversample_per_size))
        .collect::<HashMap<usize, u64>>();

    let candidates = tc.sample_unique_root(min_size, max_size, &samples_per_size);

    // Rejection filter: keep only terms NOT in the previous egraph
    candidates
        .into_iter()
        .filter(|tree| {
            let Ok(recexpr) = tree.to_string().parse::<RecExpr<Math>>() else {
                return false;
            };
            prev_raw_egg.lookup_expr(&recexpr).is_none()
        })
        .take(count)
        .collect()
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
fn print_summary(results: &[VerifyResult], goal: &TreeNode<MathLabel>, max_iters: usize) {
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
}
