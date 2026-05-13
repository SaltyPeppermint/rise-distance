use std::env::current_dir;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use egg::RecExpr;
use hashbrown::HashMap;
use num::BigUint;
use serde::Serialize;

use rise_distance::cli::argparse::{
    EqsatConfig, SampleStrategy, SeedInput, TermSampleDist, parse_seeds, read_folder_args,
};
use rise_distance::cli::types::{EnrichedSeed, EnrichedSeedFailed, GoalGenMetadata};
use rise_distance::cli::{EqsatMetadata, PrecomputePackage, init_log};
use rise_distance::egg::math::{Math, RULES};
use rise_distance::egg::{big_eqsat, lower};
use rise_distance::tee_println;

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

    /// Output run folder for logs and metadata (auto-generated if omitted).
    #[arg(short, long)]
    output: Option<String>,
}

fn main() {
    let log_file = current_dir().unwrap();
    init_log(&log_file);
    let args = Args::parse();
    let eqsat = read_folder_args(&args.path);

    tee_println!("Input folder: {}", args.path.display());

    let seeds = parse_seeds(SeedInput::JSON(args.path.join("terms.json")));
    let take_n = args.take_first.unwrap_or(seeds.len()).min(seeds.len());
    tee_println!("Distribution: {}", args.size_distribution);
    tee_println!("Seeds to process: {} (of {} total)", take_n, seeds.len());

    let mut enriched_map: HashMap<String, EnrichedSeed> = HashMap::new();

    for (i, (seed_str, seed_expr, max_size)) in seeds.into_iter().take(take_n).enumerate() {
        tee_println!("\n=== Seed {i}: {seed_str} (max_size={max_size}) ===");
        let enriched = process_seed(&args, &eqsat, &seed_expr, max_size);
        enriched_map.insert(seed_str.clone(), enriched);
    }

    write_enriched_terms(&args.path, &enriched_map);
    tee_println!(
        "\nWrote {} enriched seed(s) to {}",
        enriched_map.len(),
        args.path.join("terms.json").display()
    );
}

fn process_seed(
    args: &Args,
    eqsat: &EqsatConfig,
    seed_expr: &RecExpr<Math>,
    max_size: usize,
) -> EnrichedSeed {
    let Some(result) = big_eqsat(seed_expr, RULES.iter(), eqsat) else {
        return EnrichedSeed::Failed(EnrichedSeedFailed {
            max_size,
            fail_reason: "big eqsat failed".to_owned(),
        });
    };

    let goal_iters = result.iters();
    let guide_iters = goal_iters / 2;
    let stop_reason = format!("{:?}", result.stop_reason());

    let guide_secs = result.data()[..=guide_iters]
        .iter()
        .map(|i| i.total_time)
        .sum::<f64>();
    let goal_secs = result.data().iter().map(|i| i.total_time).sum::<f64>();

    let curr_guide = &result.data()[guide_iters].data.0;
    let guide_nodes = curr_guide.total_number_of_nodes();
    let guide_classes = curr_guide.classes().len();
    let goal_nodes = result.curr().total_number_of_nodes();
    let goal_classes = result.curr().classes().len();

    tee_println!("goal_iters={goal_iters} guide_iters={guide_iters} stop={stop_reason}");
    tee_println!("guide egraph: {guide_nodes} nodes, {guide_classes} classes in {guide_secs:.2}s");
    tee_println!("goal egraph:  {goal_nodes} nodes, {goal_classes} classes in {goal_secs:.2}s");

    let now = Instant::now();
    let Some(pp) = PrecomputePackage::<BigUint, Math, _>::precompute(&result, max_size) else {
        return EnrichedSeed::Failed(EnrichedSeedFailed {
            max_size,
            fail_reason: format!("goal precompute returned None (goal_iters={goal_iters})"),
        });
    };
    tee_println!("Precompute built in {:.2}s", now.elapsed().as_secs_f64());
    pp.log_root();

    let goals = match pp.sample_frontier_terms(
        args.goals,
        args.size_distribution,
        args.goal_sample_strategy,
        [0, 0],
        true,
    ) {
        Ok(g) => g,
        Err(e) => {
            return EnrichedSeed::Failed(EnrichedSeedFailed {
                max_size,
                fail_reason: format!("sample_frontier failed (e={e})"),
            });
        }
    };

    let goal_strings = goals
        .iter()
        .map(|g| lower(g.clone()).to_string())
        .collect::<Vec<_>>();

    let frontier_histogram = pp
        .root_histogram()
        .iter()
        .map(|(s, c)| (s.to_string(), c.clone()))
        .collect();

    let ok = GoalGenMetadata {
        eqsat_config: *eqsat,
        max_size,
        goal_egraph: EqsatMetadata {
            nodes: goal_nodes,
            classes: goal_classes,
            time: goal_secs,
            iters: goal_iters,
        },
        guide_egraph: EqsatMetadata {
            nodes: guide_nodes,
            classes: guide_classes,
            time: guide_secs,
            iters: guide_iters,
        },
        goals: goal_strings,
        frontier_histogram,
        stop_reason: stop_reason.clone(),
    };

    EnrichedSeed::Ok(ok)
}

/// Rewrite `<folder>/terms.json`, preserving the `[size, {term: payload}]`
/// grouping but replacing each payload slot with the enriched per-seed data.
/// Seeds not present in `enriched_map` (e.g. dropped by `--take-first`) are
/// omitted from the output.
fn write_enriched_terms(folder: &Path, enriched_map: &HashMap<String, EnrichedSeed>) {
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
