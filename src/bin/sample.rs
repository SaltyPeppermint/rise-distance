//! Produce the guide-candidate menu for each seed/goal pair.
//!
//! This is the "first half" of the old `guide` binary: for every enriched seed
//! it replays the guide-phase eqsat, builds a [`PrecomputePackage`], and samples
//! the six-strategy guide menu. The individual search legs (the second half) are
//! now driven by `driver.py`, which feeds chosen guide subsets to `verify`.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use num::BigUint;
use serde::Serialize;
use time::OffsetDateTime;

use rise_distance::cli::sample::{
    GoalGenMetadata, GuideExpr, SeedSamples, Strategy, read_enriched_terms,
};
use rise_distance::cli::{get_run_folder, read_folder_args, read_folder_language, write_config};
use rise_distance::eqsat::{EqsatConfig, run_eqsat};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::sampling::{PrecomputePackage, TermSampleDist};
use rise_distance::{MyAnalysis, MyLanguage, OriginLang};

#[derive(Parser, Serialize)]
#[command(
    about = "Sample the guide-candidate menu per seed/goal pair (feeds driver.py)",
    after_help = "\
Example:
  # Pre-generate goals first:
  goal data/seed_terms/dusky-cramp --goals 10
  # Then sample the guide menu for one seed (driver.py fans these out):
  sample --seed-index 0 data/seed_terms/dusky-cramp
"
)]
struct Args {
    /// Folder containing `terms.json` (enriched by `goal`) and `args.json`.
    path: PathBuf,

    /// How to distribute the guide sample budget across sizes.
    #[arg(long, default_value_t = TermSampleDist::GREEDY)]
    size_distribution: TermSampleDist,

    /// Output folder (generated if omitted).
    #[arg(short, long)]
    output: Option<String>,

    /// Which seed to sample, as a 0-based index into the flattened seed list
    /// (see `read_enriched_terms` for the ordering). Exactly one seed is
    /// processed per invocation; `driver.py` fans these out across parallel
    /// subprocesses, one per seed.
    #[arg(long)]
    seed_index: usize,

    /// How many guide candidates to draw per sampling strategy. The menu must be
    /// large enough for the driver to pick its widest `k` and to reshuffle
    /// across restarts. `Smallest` always contributes exactly one term.
    #[arg(long, default_value_t = 1000)]
    samples_per_strategy: usize,

    /// How much to grow `max_size` on each precompute retry.
    #[arg(long, default_value_t = 5)]
    retry_step: usize,

    /// How many times to retry precompute with a larger `max_size` before
    /// giving up on a seed.
    #[arg(long, default_value_t = 20)]
    max_retries: usize,

    /// How many sizes need to be present in the precomputed histogram of root
    #[arg(long, default_value_t = 5)]
    sample_sizes: usize,
}

fn main() {
    let args = Args::parse();
    let eqsat = read_folder_args(&args.path);
    let language = read_folder_language(&args.path);

    let prefix = format!("iters-{}", eqsat.max_iters);
    let run_folder = get_run_folder(args.output.as_deref(), "guide_samples", &prefix);

    println!("Starting at {}", OffsetDateTime::now_local().unwrap());
    println!("Run folder: {}", run_folder.to_string_lossy());
    println!("Input folder: {}", args.path.display());
    println!("Language: {language:?}");
    println!("Distribution: {}", args.size_distribution);

    let samples_path = run_folder.join("samples.json");
    let count = match language {
        AvailableLanguages::Diospyros => main_inner::<_, ()>(
            &args,
            &eqsat,
            &diospyros::rules(false, false),
            &samples_path,
        ),
        AvailableLanguages::Math => {
            main_inner::<_, math::ConstantFold>(&args, &eqsat, &math::rules(), &samples_path)
        }
        AvailableLanguages::Prop => {
            main_inner::<_, prop::ConstantFold>(&args, &eqsat, &prop::rules(), &samples_path)
        }
    };
    write_config(&run_folder, &args);

    println!(
        "\nWrote {count} seed record(s) to {}",
        samples_path.display()
    );
    println!("Finished at {}", OffsetDateTime::now_local().unwrap());
}

/// Sample every seed for one concrete language and write the `SeedSamples`
/// records to `samples_path`. Returns how many records were written. The output
/// is written here (rather than returned) because the `SeedSamples<L>` type is
/// language-specific and the caller's `match` arms can't unify on it.
fn main_inner<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    eqsat: &EqsatConfig,
    rules: &[Rewrite<L, N>],
    samples_path: &Path,
) -> usize {
    let seeds = read_enriched_terms::<BigUint>(&args.path);
    let i = args.seed_index;
    let (seed_str, payload) = seeds.get(i).unwrap_or_else(|| {
        panic!(
            "--seed-index {i} out of range (folder has {} seeds)",
            seeds.len()
        )
    });
    println!("Processing seed {i} of {} total", seeds.len());

    let mut out = Vec::new();
    match payload {
        Err(e) => {
            println!("\n=== Seed {i} : {seed_str} SKIPPED (failed in goal stage with {e}) ===");
        }
        Ok(ok) => {
            println!(
                "\n=== Seed {i}: {seed_str} (max_size={}, guide_iters={}) ===",
                ok.max_size, ok.guide_egraph.iters
            );
            match sample_seed(args, eqsat, seed_str, ok, rules) {
                Ok(record) => out.push(record),
                Err(e) => println!("ERROR OCCURRED:\n{e}"),
            }
            println!("Finished seed at {}", OffsetDateTime::now_local().unwrap());
        }
    }

    let file = File::create(samples_path).expect("create samples.json");
    serde_json::to_writer(BufWriter::new(file), &out).expect("write samples.json");
    out.len()
}

fn sample_seed<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    eqsat: &EqsatConfig,
    seed_str: &str,
    payload: &GoalGenMetadata<BigUint>,
    rules: &[Rewrite<L, N>],
) -> Result<SeedSamples<L>, String> {
    let seed_expr = seed_str
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed '{seed_str}': {e}"));

    // Replay the guide phase with the recorded iteration count.
    let replay_config = EqsatConfig {
        max_iters: payload.guide_egraph.iters,
        ..*eqsat
    };
    eqsat.warn_on_config_drift(&payload.eqsat_config);

    let result = run_eqsat(&seed_expr, rules.iter(), &replay_config).ok_or("Eqsat failed")?;
    println!("Guide replay stop reason: {:?}", result.stop_reason());

    let guide_nodes = result.curr().total_number_of_nodes();
    let guide_classes = result.curr().classes().len();
    let guide_iters = result.data().len();
    let guide_time = result.data().iter().map(|i| i.total_time).sum();
    println!("Guide egraph (replay): {guide_nodes} nodes, {guide_classes} classes");

    let mut root_log = String::new();

    let start_size = AstSize.cost_rec(&seed_expr);
    let (max_size, pc) = PrecomputePackage::<BigUint, _, _>::backoff_precompute(
        &result,
        start_size,
        args.max_retries,
        args.retry_step,
        args.sample_sizes,
        false,
        &mut root_log,
    )
    .map_err(|tried_max_size| {
        format!(
            "goal precompute returned None after {} retries (max_size={})",
            args.max_retries, tried_max_size
        )
    })?;
    println!("PC computation succeeded with max_size {max_size}!");
    pc.log_root(&mut root_log);
    print!("{root_log}");

    let mut candidates = BTreeMap::new();
    for strategy in Strategy::ALL {
        let terms = draw_candidates(args, strategy, &pc);
        candidates.insert(
            strategy.name().to_owned(),
            terms.iter().map(GuideExpr::from_recexpr).collect(),
        );
    }

    Ok(SeedSamples {
        seed: seed_str.to_owned(),
        goals: payload.goals.clone(),
        candidates,
        max_size: payload.max_size,
        guide_nodes,
        guide_classes,
        guide_iters,
        guide_time,
        stop_reason: format!("{:?}", result.stop_reason()),
    })
}

/// Draw the guide-candidate pool for one strategy. The four sampling strategies
/// draw `samples_per_strategy` terms in a single batch; `Smallest` contributes
/// the one smallest (novel or overall) root term.
fn draw_candidates<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    strategy: Strategy,
    pc: &PrecomputePackage<BigUint, L, N>,
) -> Vec<RecExpr<OriginLang<L>>> {
    match strategy {
        // Replacement is a driver concern (how it re-draws subsets from the
        // pool across restarts); the pool itself is one novel sampled batch
        // either way. Sampling strategies always draw novel terms.
        Strategy::Sample(s) => pc
            .sample_frontier_terms(
                args.samples_per_strategy,
                args.size_distribution,
                s,
                [args.samples_per_strategy as u64, strategy.seed_of()],
                true,
            )
            .unwrap_or_else(|| {
                eprintln!(
                    "WARNING: strategy {} drew 0 candidates (empty novel frontier); \
                     driver legs for this strategy will have no guides to pick from",
                    strategy.name()
                );
                Vec::new()
            }),
        Strategy::Smallest { novel } => vec![pc.smallest(pc.root(), novel)],
    }
}
