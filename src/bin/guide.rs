// #[global_allocator]
// static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use clap::Parser;
use egg::RecExpr;
use hashbrown::HashMap;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use num::BigUint;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;

use rise_distance::cli::argparse::{EqsatConfig, SampleStrategy, TermSampleDist, read_folder_args};
use rise_distance::cli::parquet::dump_summary_parquet;
use rise_distance::cli::types::{EnrichedSeed, GoalGenMetadata, GoalSummary, TrialsPerK};
use rise_distance::cli::{EqsatMetadata, get_run_folder, init_log, write_config, write_metadata};
use rise_distance::egg::math::{ConstantFold, Math, RULES};
use rise_distance::egg::{Goal, run_eqsat, verify_reachability};
use rise_distance::search::PrecomputePackage;
use rise_distance::{Counter, tee_println};
use time::OffsetDateTime;

#[derive(Parser, Serialize)]
#[command(
    about = "Evaluate distance metrics as guide predictors for equality saturation",
    after_help = "\
Example:
  # Pre-generate goals first:
  goal data/seed_terms/dusky-cramp --goals 10
  # Then run guide experiments:
  guide --full-union --take-first 10 data/seed_terms/dusky-cramp
"
)]
struct Args {
    /// Folder containing `terms.json` (enriched by `goal`) and `args.json`.
    path: PathBuf,

    /// How to distribute the guide sample budget across sizes.
    #[arg(long, default_value_t = TermSampleDist::UNIFORM)]
    size_distribution: TermSampleDist,

    /// Output folder (generated if omitted)
    #[arg(short, long)]
    output: Option<String>,

    /// Use the experimental `add_with_full_union` for the new egraph
    #[arg(long)]
    full_union: bool,

    /// Only process the first N seeds (useful for quick experiments)
    #[arg(long)]
    take_first: Option<usize>,

    /// Number of repetitions per (k, seed, goal) combination
    #[arg(long, default_value_t = 100)]
    repetitions: usize,
}

const TRIAL_SIZE: [usize; 6] = [1, 2, 5, 10, 50, 100];

fn main() {
    let args = Args::parse();
    let folder_args = read_folder_args(&args.path);

    let prefix = format!(
        "iters-{}-fullunion-{}",
        folder_args.max_iters, args.full_union
    );
    let run_folder = get_run_folder(args.output.as_deref(), "guide_eval", &prefix);
    init_log(&run_folder);

    tee_println!("Starting at {}", OffsetDateTime::now_local().unwrap());
    tee_println!("Run Folder: {}", run_folder.to_string_lossy());
    tee_println!("Input folder: {}", args.path.display());

    let seeds = read_enriched_terms(&args.path);
    let take_n = args.take_first.unwrap_or(seeds.len()).min(seeds.len());
    tee_println!("Distribution: {}", args.size_distribution);
    tee_println!("Seeds to process: {} (of {} total)", take_n, seeds.len());

    let mut all: HashMap<String, Vec<GoalResults>> = HashMap::new();
    let mut all_metadata = Vec::new();

    for (i, (seed_str, payload)) in seeds.iter().take(take_n).enumerate() {
        let EnrichedSeed::Ok(ok) = payload else {
            tee_println!("\n=== Seed {i}: {seed_str} SKIPPED (failed in goal stage) ===");
            continue;
        };
        tee_println!(
            "\n=== Seed {i}: {seed_str} (max_size={}, guide_iters={}) ===",
            ok.max_size,
            ok.guide_egraph.iters
        );
        if let Some((r, metadata)) = process_seed(&args, &folder_args, seed_str, ok) {
            all_metadata.push(metadata);
            for (name, r_s) in r {
                all.entry(name).or_default().extend(r_s);
            }
        }
        tee_println!(
            "\nFinished seed at {}",
            OffsetDateTime::now_local().unwrap()
        );
    }

    write_top_k_outputs(&run_folder, all);
    write_config(&run_folder, &args);
    write_metadata(&run_folder, &all_metadata);
    tee_println!("\nFinished at {}", OffsetDateTime::now_local().unwrap());
}

/// Read enriched `terms.json`. Returns a flat list in deterministic order
/// (groups in JSON order, terms within each group sorted alphabetically) so
/// `--take-first` is stable across runs.
fn read_enriched_terms(folder: &Path) -> Vec<(String, EnrichedSeed)> {
    let path = folder.join("terms.json");
    let file =
        File::open(&path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));

    // Read directly into a typed schema. Going via `serde_json::Value` first
    // would force HashMap<usize, _> keys through string → usize conversion
    // that `from_value` doesn't do. The inner map is a BTreeMap so its
    // iteration order is deterministic; a HashMap here would make
    // `--take-first` pick a different subset each run.
    let groups: Vec<(usize, BTreeMap<String, EnrichedSeed>)> = serde_json::from_reader(file)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to parse {}: {e}. Did you run `goal` on this folder first?",
                path.display()
            )
        });

    groups
        .into_iter()
        .flat_map(|(_size, inner)| inner)
        .collect()
}

fn process_seed(
    args: &Args,
    folder_args: &EqsatConfig,
    seed_str: &str,
    payload: &GoalGenMetadata,
) -> Option<(HashMap<String, Vec<GoalResults>>, serde_json::Value)> {
    let seed_expr = seed_str
        .parse::<RecExpr<Math>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed '{seed_str}': {e}"));

    // Replay the guide phase with the recorded iteration count.
    let replay_config = EqsatConfig {
        max_iters: payload.guide_egraph.iters,
        ..*folder_args
    };
    let result = run_eqsat(&seed_expr, RULES.iter(), &replay_config)?;
    tee_println!("Guide replay stop reason: {:?}", result.stop_reason());

    let guide_nodes = result.curr().total_number_of_nodes();
    let guide_classes = result.curr().classes().len();
    let guide_time = result.data().last().unwrap().total_time;
    let guide_iters = result.data().len();
    tee_println!("Guide egraph (replay): {guide_nodes} nodes, {guide_classes} classes");
    folder_args.warn_on_config_drift(&payload.eqsat_config);

    let pc = PrecomputePackage::<BigUint, _, _>::precompute(&result, payload.max_size)?;
    pc.log_root();

    let goals = payload
        .goals
        .iter()
        .map(|g| {
            g.parse::<RecExpr<Math>>()
                .unwrap_or_else(|e| panic!("Failed to parse stored goal '{g}': {e}"))
        })
        .collect::<Vec<_>>();

    let mp = MultiProgress::new();
    let strategies = [
        Strategy::NoReplacement(SampleStrategy::Count),
        Strategy::WithReplacement(SampleStrategy::Count),
        Strategy::NoReplacement(SampleStrategy::Naive),
        Strategy::WithReplacement(SampleStrategy::Naive),
        Strategy::Smallest { novel: false },
        Strategy::Smallest { novel: true },
    ];

    tee_println!("\nRunning {} strategies in parallel...", strategies.len());
    let r = strategies
        .into_par_iter()
        .map(|strat| {
            let name = strat.name().to_owned();
            let res = strat.run_trials(args, folder_args, seed_str, &goals, &pc, &mp);
            (name, res)
        })
        .collect();

    let metadata = json!({
        "seed": seed_str,
        "goal_eqsat": payload,
        "guide_eqsat" :EqsatMetadata{
            nodes: guide_nodes,
            classes: guide_classes,
            time: guide_time,
            iters: guide_iters
        }
    });

    Some((r, metadata))
}

fn write_top_k_outputs(run_folder: &Path, all: HashMap<String, Vec<GoalResults>>) {
    for (name, results) in all {
        let output_path = run_folder.join(format!("{name}_top_k.json"));
        let output_file = File::create(output_path).expect("Failed to create output json file");
        let mut output_writer = BufWriter::new(output_file);
        serde_json::to_writer(&mut output_writer, &results).expect("write top-k json");

        let summaries = results
            .iter()
            .map(|r| GoalSummary::from_entries(&r.seed, &r.goal, &r.runs))
            .collect::<Vec<_>>();
        let summary_path = run_folder.join(format!("{name}_top_k_summary.json"));
        let summary_file = File::create(summary_path).expect("Failed to create summary json file");
        let summary_writer = BufWriter::new(summary_file);
        serde_json::to_writer(summary_writer, &summaries).expect("write top-k summary json");

        let parquet_path = run_folder.join(format!("{name}_top_k_summary.parquet"));
        dump_summary_parquet(&parquet_path, &summaries);
    }
}

fn strategy_bar(mp: &MultiProgress, name: &str, total: u64) -> ProgressBar {
    let style = ProgressStyle::with_template("{msg:>25} [{bar:40}] {pos}/{len}")
        .unwrap()
        .progress_chars("=> ");
    let pb = mp.add(ProgressBar::new(total).with_style(style));
    pb.set_message(name.to_owned());
    pb
}

#[derive(Serialize)]
struct GoalResults {
    seed: String,
    goal: String,
    runs: TrialsPerK,
}

/// One guide-sampling strategy. Sampling variants always draw novel terms;
/// only `Smallest` exposes the novel/overall choice.
#[derive(Copy, Clone)]
enum Strategy {
    NoReplacement(SampleStrategy),
    WithReplacement(SampleStrategy),
    Smallest { novel: bool },
}

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Strategy::NoReplacement(SampleStrategy::Count) => "no_replacement_count",
            Strategy::NoReplacement(SampleStrategy::Naive) => "no_replacement_naive",
            Strategy::WithReplacement(SampleStrategy::Count) => "with_replacement_count",
            Strategy::WithReplacement(SampleStrategy::Naive) => "with_replacement_naive",
            Strategy::Smallest { novel: true } => "smallest_novel",
            Strategy::Smallest { novel: false } => "smallest_overall",
        }
    }

    /// Number of `pb.inc(1)` ticks [`Self::run_trial`] issues for a single goal.
    fn work_per_goal(self, args: &Args) -> u64 {
        match self {
            Strategy::NoReplacement(_) | Strategy::WithReplacement(_) => {
                (TRIAL_SIZE.len() * args.repetitions) as u64
            }
            Strategy::Smallest { .. } => 1,
        }
    }

    fn run_trials<C: Counter>(
        self,
        args: &Args,
        eqsat: &EqsatConfig,
        seed_str: &str,
        goals: &[RecExpr<Math>],
        pp: &PrecomputePackage<C, Math, ConstantFold>,
        mp: &MultiProgress,
    ) -> Vec<GoalResults> {
        let pb = strategy_bar(
            mp,
            self.name(),
            self.work_per_goal(args) * goals.len() as u64,
        );
        let r = goals
            .par_iter()
            .map(|goal| GoalResults {
                seed: seed_str.to_owned(),
                goal: goal.to_string(),
                runs: self.run_trial(args, eqsat, goal, pp, &pb),
            })
            .collect();
        pb.finish();
        r
    }

    fn run_trial<C: Counter>(
        self,
        args: &Args,
        eqsat: &EqsatConfig,
        goal_recexpr: &RecExpr<Math>,
        pp: &PrecomputePackage<C, Math, ConstantFold>,
        pb: &ProgressBar,
    ) -> TrialsPerK {
        let goal = Goal::Expr(goal_recexpr.clone());
        match self {
            Strategy::WithReplacement(strategy) => TRIAL_SIZE
                .par_iter()
                .map(|&k| {
                    let trials = (0..args.repetitions)
                        .into_par_iter()
                        .map(|s| {
                            let subset = pp.sample_frontier_terms(
                                k,
                                args.size_distribution,
                                strategy,
                                [k as u64, s as u64],
                                true,
                            )?;
                            let r =
                                verify_reachability(&subset, &goal, &RULES, eqsat, args.full_union)
                                    .map_err(|e| e.into());
                            pb.inc(1);
                            r
                        })
                        .collect::<Vec<_>>();
                    (k, trials)
                })
                .collect(),
            Strategy::NoReplacement(strategy) => TRIAL_SIZE
                .par_iter()
                .map(|&k| {
                    let samples = pp.sample_frontier_terms(
                        k * args.repetitions,
                        args.size_distribution,
                        strategy,
                        [k as u64, 0],
                        true,
                    );
                    let trials = match samples {
                        Err(e) => {
                            pb.inc(args.repetitions as u64);
                            vec![Err(e); args.repetitions]
                        }
                        Ok(samples) => samples
                            .par_chunks(k)
                            .map(|subset| {
                                let r = verify_reachability(
                                    subset,
                                    &goal,
                                    &RULES,
                                    eqsat,
                                    args.full_union,
                                )
                                .map_err(Into::into);
                                pb.inc(1);
                                r
                            })
                            .collect(),
                    };
                    (k, trials)
                })
                .collect(),
            Strategy::Smallest { novel } => {
                let r = verify_reachability(
                    std::slice::from_ref(&pp.smallest(pp.root(), novel)),
                    &goal,
                    &RULES,
                    eqsat,
                    args.full_union,
                )
                .map_err(Into::into);
                pb.inc(1);
                HashMap::from([(1, vec![r])])
            }
        }
    }
}
