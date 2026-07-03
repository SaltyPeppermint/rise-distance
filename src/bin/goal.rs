// #[global_allocator]
// static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::fmt::Write as _;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use hashbrown::HashMap;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::ToPrimitive;
use rayon::prelude::*;

use serde::Serialize;

use rise_distance::cli::sample::GoalGenMetadata;
use rise_distance::cli::{read_folder_args, read_folder_language};
use rise_distance::eqsat::{EqsatConfig, SplitMetadata};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::lower;
use rise_distance::sampling::{PrecomputePackage, SampleStrategy, TermSampleDist};
use rise_distance::{MyAnalysis, MyLanguage, eqsat};

#[derive(Parser, Serialize)]
#[command(
    about = "Generate goal terms from seeds and write them back into terms.json",
    after_help = "\
Example:
  goal data/seed_terms/dusky-cramp --goals 10
"
)]
struct Args {
    /// Folder containing `terms.json` and `args.json`. Output is written back
    /// to `terms.json` in this folder.
    path: PathBuf,

    /// Number of goal terms to sample per seed.
    #[arg(long, default_value_t = 10)]
    goals: usize,

    /// How to distribute the sample budget across sizes.
    #[arg(long, default_value_t = TermSampleDist::UNIFORM)]
    size_distribution: TermSampleDist,

    /// How to sample the GOAL terms.
    #[arg(long, default_value_t = SampleStrategy::Count)]
    goal_sample_strategy: SampleStrategy,

    /// Only process the first N seeds.
    #[arg(long)]
    take_first: Option<usize>,

    /// How much to grow `max_size` on each precompute retry.
    #[arg(long, default_value_t = 1)]
    retry_step: usize,

    /// How many times to retry precompute with a larger `max_size` before
    /// giving up on a seed.
    #[arg(long, default_value_t = 50)]
    max_retries: usize,

    /// How many sizes need to be present in the precomputed histogram of root
    #[arg(long, default_value_t = 5)]
    min_sizes: usize,

    /// Output run folder for logs and metadata (auto-generated if omitted).
    #[arg(short, long)]
    output: Option<String>,
}

fn main() {
    let args = Args::parse();
    let eqsat = read_folder_args(&args.path);
    let language = read_folder_language(&args.path);

    println!("Input folder: {}", args.path.display());
    println!("Language: {language:?}");

    match language {
        AvailableLanguages::Diospyros => {
            main_inner::<_, ()>(&args, &eqsat, &diospyros::rules(false, false));
        }
        AvailableLanguages::Math => {
            main_inner::<_, math::ConstantFold>(&args, &eqsat, &math::rules());
        }
        AvailableLanguages::Prop => {
            main_inner::<_, prop::ConstantFold>(&args, &eqsat, &prop::rules());
        }
    }
}

fn main_inner<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    eqsat: &EqsatConfig,
    rules: &[Rewrite<L, N>],
) {
    let seeds = parse_seeds::<L>(SeedInput::JSON(args.path.join("terms.json")));
    let take_n = args.take_first.unwrap_or(seeds.len()).min(seeds.len());
    println!("Distribution: {}", args.size_distribution);
    println!("Seeds to process: {} (of {} total)", take_n, seeds.len());

    let style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} seeds ({eta})",
    )
    .expect("valid template")
    .progress_chars("=>-");
    let bar = ProgressBar::new(take_n as u64).with_style(style);

    let enriched_map = seeds
        .into_par_iter()
        .take(take_n)
        .enumerate()
        .map(|(idx, (seed_str, seed_expr, _max_size))| {
            // Buffer this seed's log lines and flush them in a single atomic
            // write so that output from concurrent seeds does not interleave.
            let mut log = format!("[seed {}/{take_n}] {seed_str}\n", idx + 1);
            let enriched = process_seed(args, eqsat, &seed_expr, rules, &mut log);
            match &enriched {
                Ok(g) => {
                    writeln!(log, "Successfully generated {} goals!", g.goals.len()).unwrap();
                }
                Err(e) => {
                    writeln!(log, "Failed to generate any goals due to: {e}").unwrap();
                }
            }
            bar.println(log.trim_end());
            (seed_str, enriched)
        })
        .progress_with(bar.clone())
        .collect();
    bar.finish();

    write_enriched_terms(&args.path, &enriched_map);
    println!(
        "\nWrote {} enriched seed(s) to {}",
        enriched_map.len(),
        args.path.join("terms.json").display()
    );
}

fn process_seed<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    eqsat: &EqsatConfig,
    seed_expr: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    log: &mut String,
) -> Result<GoalGenMetadata<u128>, String> {
    let Some(result) = eqsat::run_eqsat(seed_expr, rules.iter(), eqsat) else {
        return Err("big eqsat failed".to_owned());
    };

    let stop_reason = format!("{:?}", result.stop_reason());
    let SplitMetadata { guide, goal } = result.split_metadata();

    writeln!(
        log,
        "goal_iters={} guide_iters={} stop={stop_reason}",
        goal.iters, guide.iters
    )
    .unwrap();
    writeln!(
        log,
        "guide egraph: {} nodes, {} classes in {:.2}s",
        guide.nodes, guide.classes, guide.time
    )
    .unwrap();
    writeln!(
        log,
        "goal egraph:  {} nodes, {} classes in {:.2}s",
        goal.nodes, goal.classes, goal.time
    )
    .unwrap();

    let now = Instant::now();

    let start_size = AstSize.cost_rec(seed_expr);
    let (used_max_size, pp) = PrecomputePackage::<u128, L, _>::backoff_precompute(
        &result,
        start_size,
        args.max_retries,
        args.retry_step,
        args.min_sizes,
        log,
    )
    .map_err(|tried_max_size| {
        format!(
            "goal precompute returned None after {} retries (goal_iters={}, max_size={})",
            args.max_retries, goal.iters, tried_max_size
        )
    })?;
    writeln!(
        log,
        "Precompute built in {:.2}s",
        now.elapsed().as_secs_f64()
    )
    .unwrap();
    pp.log_root(log);

    let goals = pp
        .sample_frontier_terms(
            args.goals,
            args.size_distribution,
            args.goal_sample_strategy,
            [0, 0],
            true,
        )
        .ok_or_else(|| "sample_frontier failed".to_owned())?;

    let goal_strings = goals
        .iter()
        .map(|g| lower(g.clone()).to_string())
        .collect::<Vec<_>>();

    let frontier_histogram = pp
        .root_histogram()
        .iter()
        .map(|(s, c)| (s.to_string(), *c))
        .collect();

    Ok(GoalGenMetadata {
        eqsat_config: *eqsat,
        max_size: used_max_size,
        goal_egraph: goal,
        guide_egraph: guide,
        goals: goal_strings,
        frontier_histogram,
        stop_reason,
    })
}

/// Rewrite `<folder>/terms.json`, preserving the `[size, {term: payload}]`
/// grouping but replacing each payload slot with the enriched per-seed data.
/// Seeds not present in `enriched_map` (e.g. dropped by `--take-first`) are
/// omitted from the output.
fn write_enriched_terms(
    folder: &Path,
    enriched_map: &HashMap<String, Result<GoalGenMetadata<u128>, String>>,
) {
    let in_path = folder.join("terms.json");
    let file = File::open(&in_path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", in_path.display()));
    let original: serde_json::Value = serde_json::from_reader(file)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", in_path.display()));

    let groups = original
        .as_array()
        .unwrap_or_else(|| panic!("Expected top-level array in {}", in_path.display()));

    let mut out_groups = Vec::new();
    for group in groups {
        let pair = group
            .as_array()
            .expect("Expected [size, {terms}] pair in terms.json");
        let size = pair[0].clone();
        let inner = pair[1]
            .as_object()
            .expect("Expected term object as second element");

        let mut new_inner = serde_json::Map::new();
        for term in inner.keys() {
            if let Some(payload) = enriched_map.get(term) {
                let value =
                    serde_json::to_value(payload).expect("EnrichedSeed serialization failed");
                new_inner.insert(term.clone(), value);
            }
        }

        if !new_inner.is_empty() {
            out_groups.push(serde_json::Value::Array(vec![
                size,
                serde_json::Value::Object(new_inner),
            ]));
        }
    }

    let out_path = folder.join("terms.json");
    let out_file = File::create(&out_path)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", out_path.display()));
    serde_json::to_writer_pretty(
        BufWriter::new(out_file),
        &serde_json::Value::Array(out_groups),
    )
    .expect("Failed to write enriched terms.json");
}

/// Either a single seed s-expression or a path to a JSON file with objects containing `size` and `term` fields.
#[derive(Debug, Clone)]
pub enum SeedInput {
    Single { term: String, max_size: usize },
    JSON(PathBuf),
}

/// Parse a `SeedInput` into a list of `(seed_string, parsed_expr, max_size)` triples.
///
/// # Panics
///
/// Panics on malformed seed expressions or unreadable JSON files.
#[must_use]
pub fn parse_seeds<L: MyLanguage>(input: SeedInput) -> Vec<(String, RecExpr<L>, usize)> {
    match input {
        SeedInput::Single { term, max_size } => {
            let expr = term
                .parse::<RecExpr<L>>()
                .unwrap_or_else(|e| panic!("Failed to parse seed: {e}"));
            vec![(term, expr, max_size)]
        }
        SeedInput::JSON(path) => {
            let reader = File::open(&path)
                .unwrap_or_else(|e| panic!("Failed to open JSON {}: {e}", path.display()));
            serde_json::from_reader::<_, serde_json::Value>(reader)
                .unwrap()
                .as_array()
                .unwrap_or_else(|| panic!("Expected top-level JSON array in {}", path.display()))
                .iter()
                .flat_map(|group| {
                    let pair = group.as_array().expect("Expected [size, {{terms}}]");
                    let max_size = 2 * pair[0]
                        .as_u64()
                        .expect("Expected size as u64 in")
                        .to_usize()
                        .unwrap();

                    pair[1]
                        .as_object()
                        .expect("Expected term object as second element")
                        .keys()
                        .map(move |term| {
                            let expr = term
                                .parse::<RecExpr<L>>()
                                .unwrap_or_else(|e| panic!("Failed to parse seed '{term}': {e}"));
                            (term.clone(), expr, max_size)
                        })
                })
                .collect()
        }
    }
}
