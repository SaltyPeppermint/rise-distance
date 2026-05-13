use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use clap::Parser;
use egg::RecExpr;
use hashbrown::HashMap;
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::BigUint;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;

use rise_distance::cli::argparse::{EqsatConfig, SampleStrategy, TermSampleDist, read_folder_args};
use rise_distance::cli::parquet::dump_summary_parquet;
use rise_distance::cli::types::{EnrichedSeed, EnrichedSeedOk, GoalSummary, TrialsPerK};
use rise_distance::cli::{PrecomputePackage, get_run_folder, init_log, write_config, write_stats};
use rise_distance::egg::math::{ConstantFold, Math, RULES};
use rise_distance::egg::{guide_only_eqsat, verify_reachability};
use rise_distance::{Counter, tee_println};

#[derive(Parser, Serialize)]
#[command(
    about = "Evaluate distance metrics as guide predictors for equality saturation",
    after_help = "\
Example:
  # Pre-generate goals first:
  goal data/seed_terms/dusky-cramp --goals 10
  # Then run guide experiments:
  guide -g 1000 --full-union --take-first 10 data/seed_terms/dusky-cramp
"
)]
struct Args {
    /// Folder containing `terms.json` (enriched by `goal`) and `args.json`.
    path: PathBuf,

    /// Number of guide candidates to sample from the n-1 frontier
    #[arg(short, long, default_value_t = 0)]
    guides: usize,

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

    /// Number of trials per (k, seed, goal) combination
    #[arg(long, default_value_t = 100)]
    trials: usize,
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
    tee_println!("Run Folder: {}", run_folder.to_string_lossy());
    tee_println!("Input folder: {}", args.path.display());

    let seeds = read_enriched_terms(&args.path);
    let take_n = args.take_first.unwrap_or(seeds.len()).min(seeds.len());
    tee_println!("Distribution: {}", args.size_distribution);
    tee_println!("Seeds to process: {} (of {} total)", take_n, seeds.len());

    let mut all: HashMap<String, Vec<GoalResults>> = HashMap::new();
    let mut all_stats = Vec::new();

    for (i, (seed_str, payload)) in seeds.iter().take(take_n).enumerate() {
        let SeedEntry::Ok(ok) = payload else {
            tee_println!("\n=== Seed {i}: {seed_str} SKIPPED (failed in goal stage) ===");
            continue;
        };
        tee_println!(
            "\n=== Seed {i}: {seed_str} (max_size={}, guide_iters={}) ===",
            ok.max_size,
            ok.guide_iters
        );
        if let Some((r, stats)) = process_seed(&args, &folder_args, seed_str, ok) {
            all_stats.push(stats);
            for (name, r_s) in r {
                all.entry(name).or_default().extend(r_s);
            }
        }
    }

    write_top_k_outputs(&run_folder, all);
    write_config(&run_folder, &args);
    write_stats(&run_folder, &all_stats);
}

enum SeedEntry {
    Ok(Box<EnrichedSeedOk>),
    Failed,
}

/// Read enriched `terms.json`. Returns a flat list preserving JSON order so
/// `--take-first` is deterministic.
fn read_enriched_terms(folder: &Path) -> Vec<(String, SeedEntry)> {
    let path = folder.join("terms.json");
    let file =
        File::open(&path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));

    // Read directly into a typed schema. Going via `serde_json::Value` first
    // would force HashMap<usize, _> keys through string → usize conversion
    // that `from_value` doesn't do.
    let groups: Vec<(usize, HashMap<String, EnrichedSeed>)> = serde_json::from_reader(file)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to parse {}: {e}. Did you run `goal` on this folder first?",
                path.display()
            )
        });

    let mut out = Vec::new();
    for (_size, inner) in groups {
        for (term, enriched) in inner {
            let entry = match enriched {
                EnrichedSeed::Ok(ok) => SeedEntry::Ok(ok),
                EnrichedSeed::Failed(_) => SeedEntry::Failed,
            };
            out.push((term, entry));
        }
    }
    out
}

fn process_seed(
    args: &Args,
    folder_args: &EqsatConfig,
    seed_str: &str,
    payload: &EnrichedSeedOk,
) -> Option<(HashMap<String, Vec<GoalResults>>, serde_json::Value)> {
    let seed_expr = seed_str
        .parse::<RecExpr<Math>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed '{seed_str}': {e}"));

    let result = guide_only_eqsat(
        &seed_expr,
        RULES.iter(),
        folder_args,
        payload.guide_iters,
        folder_args.backoff_scheduler,
    )?;
    tee_println!("Guide replay stop reason: {:?}", result.stop_reason());

    let guide_nodes = result.curr().total_number_of_nodes();
    let guide_classes = result.curr().classes().len();
    let guide_time = result.data().last().unwrap().total_time;
    let guide_iters = result.data().len();
    tee_println!("Guide egraph (replay): {guide_nodes} nodes, {guide_classes} classes");
    if guide_nodes != payload.guide_egraph_nodes || guide_classes != payload.guide_egraph_classes {
        tee_println!(
            "WARNING: replay differs from stored stats ({} nodes, {} classes stored)",
            payload.guide_egraph_nodes,
            payload.guide_egraph_classes
        );
    }

    let pc = PrecomputePackage::<BigUint, _, _>::precompute(
        result.curr(),
        result.prev(),
        result.root(),
        payload.max_size,
    )?;
    pc.log_root();

    let goals = payload
        .goals
        .iter()
        .map(|g| {
            g.parse::<RecExpr<Math>>()
                .unwrap_or_else(|e| panic!("Failed to parse stored goal '{g}': {e}"))
        })
        .collect::<Vec<_>>();

    tee_println!("\nRunning top_k experiments NO REPLACEMENT COUNTBASED...");
    let no_repl_count = GuideSetNoReplacement::new(&pc, true, SampleStrategy::Count).run_trials(
        args,
        folder_args,
        seed_str,
        &goals,
    );
    tee_println!("\nRunning top_k experiments WITH REPLACEMENT COUNTBASED...");
    let with_repl_count = GuideSetWithReplacement::new(&pc, true, SampleStrategy::Count)
        .run_trials(args, folder_args, seed_str, &goals);

    tee_println!("\nRunning top_k experiments NO REPLACEMENT NAIVE...");
    let no_repl_naive = GuideSetNoReplacement::new(&pc, true, SampleStrategy::Naive).run_trials(
        args,
        folder_args,
        seed_str,
        &goals,
    );
    tee_println!("\nRunning top_k experiments WITH REPLACEMENT NAIVE...");
    let with_repl_naive = GuideSetWithReplacement::new(&pc, true, SampleStrategy::Naive)
        .run_trials(args, folder_args, seed_str, &goals);

    tee_println!("\nRunning single experiments ONLY SMALLEST...");
    let smallest_overall =
        Smallest::new(&pc, false).run_trials(args, folder_args, seed_str, &goals);
    tee_println!("\nRunning single experiments ONLY SMALLEST NOVEL...");
    let smallest_novel = Smallest::new(&pc, true).run_trials(args, folder_args, seed_str, &goals);

    let stats = json!({
        "seed": seed_str,
        "max_size": payload.max_size,
        "guide_iters": payload.guide_iters,
        "goal_iters": payload.goal_iters,
        "stored_guide_egraph_nodes": payload.guide_egraph_nodes,
        "stored_guide_egraph_classes": payload.guide_egraph_classes,
        "stored_guide_egraph_time": payload.guide_eqsat_time,
        "stored_guide_egraph_iters": payload.guide_iters,
        "replay_guide_egraph_nodes": guide_nodes,
        "replay_guide_egraph_classes": guide_classes,
        "replay_guide_egraph_time": guide_time,
        "replay_guide_egraph_iters": guide_iters,
    });

    let r = HashMap::from([
        ("no_replacement_count".to_owned(), no_repl_count),
        ("with_replacement_count".to_owned(), with_repl_count),
        ("no_replacement_naive".to_owned(), no_repl_naive),
        ("with_replacement_naive".to_owned(), with_repl_naive),
        ("smallest_overall".to_owned(), smallest_overall),
        ("smallest_novel".to_owned(), smallest_novel),
    ]);
    Some((r, stats))
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

fn progress_bars(args: &Args) -> Vec<(usize, ProgressBar)> {
    let mp = MultiProgress::new();
    let style = ProgressStyle::with_template("{msg:>6} [{bar:40}] {pos}/{len}")
        .unwrap()
        .progress_chars("=> ");

    TRIAL_SIZE
        .iter()
        .map(|&k| {
            let pb = mp.add(ProgressBar::new(args.trials as u64).with_style(style.clone()));
            pb.set_message(format!("k={k}"));
            (k, pb)
        })
        .collect::<Vec<_>>()
}

#[derive(Serialize)]
struct GoalResults {
    seed: String,
    goal: String,
    runs: TrialsPerK,
}

trait Strategy<'p, C: Counter> {
    fn run_trial(
        &self,
        args: &Args,
        eqsat: &EqsatConfig,
        goal_recexpr: &RecExpr<Math>,
    ) -> TrialsPerK;

    fn run_trials(
        &self,
        args: &Args,
        eqsat: &EqsatConfig,
        seed_str: &str,
        goals: &[RecExpr<Math>],
    ) -> Vec<GoalResults> {
        goals
            .iter()
            .map(|goal| {
                tee_println!("Current goal: {goal}");
                GoalResults {
                    seed: seed_str.to_owned(),
                    goal: goal.to_string(),
                    runs: self.run_trial(args, eqsat, goal),
                }
            })
            .collect()
    }
}

struct GuideSetWithReplacement<'p, C: Counter> {
    pp: &'p PrecomputePackage<'p, C, Math, ConstantFold>,
    novel: bool,
    strategy: SampleStrategy,
}

impl<'p, C: Counter> GuideSetWithReplacement<'p, C> {
    fn new(
        pp: &'p PrecomputePackage<'p, C, Math, ConstantFold>,
        novel: bool,
        strategy: SampleStrategy,
    ) -> Self {
        Self {
            pp,
            novel,
            strategy,
        }
    }
}

impl<'p, C: Counter> Strategy<'p, C> for GuideSetWithReplacement<'p, C> {
    fn run_trial(
        &self,
        args: &Args,
        eqsat: &EqsatConfig,
        goal_recexpr: &RecExpr<Math>,
    ) -> TrialsPerK {
        let bars = progress_bars(args);
        bars.into_par_iter()
            .map(|(k, pb)| {
                let trials = (0..args.trials)
                    .into_par_iter()
                    .map(|s| {
                        let subset = self.pp.sample_frontier_terms(
                            k,
                            args.size_distribution,
                            self.strategy,
                            [k as u64, s as u64],
                            self.novel,
                        )?;
                        verify_reachability(&subset, goal_recexpr, &RULES, eqsat, args.full_union)
                            .map_err(|e| e.into())
                    })
                    .progress_with(pb)
                    .collect::<Vec<_>>();

                (k, trials)
            })
            .collect()
    }
}

struct GuideSetNoReplacement<'p, C: Counter> {
    pp: &'p PrecomputePackage<'p, C, Math, ConstantFold>,
    novel: bool,
    strategy: SampleStrategy,
}

impl<'p, C: Counter> GuideSetNoReplacement<'p, C> {
    fn new(
        pp: &'p PrecomputePackage<'p, C, Math, ConstantFold>,
        novel: bool,
        strategy: SampleStrategy,
    ) -> Self {
        Self {
            pp,
            novel,
            strategy,
        }
    }
}

impl<'p, C: Counter> Strategy<'p, C> for GuideSetNoReplacement<'p, C> {
    fn run_trial(
        &self,
        args: &Args,
        eqsat: &EqsatConfig,
        goal_recexpr: &RecExpr<Math>,
    ) -> TrialsPerK {
        let bars = progress_bars(args);
        bars.into_par_iter()
            .map(|(k, pb)| {
                let samples = self.pp.sample_frontier_terms(
                    k * args.trials,
                    args.size_distribution,
                    self.strategy,
                    [k as u64, 0],
                    self.novel,
                );
                let trials = match samples {
                    Err(e) => vec![Err(e); args.trials],
                    Ok(samples) => samples
                        .par_chunks(k)
                        .map(|subset| {
                            verify_reachability(
                                subset,
                                goal_recexpr,
                                &RULES,
                                eqsat,
                                args.full_union,
                            )
                            .map_err(Into::into)
                        })
                        .progress_with(pb)
                        .collect(),
                };
                (k, trials)
            })
            .collect()
    }
}

struct Smallest<'p, C: Counter> {
    pp: &'p PrecomputePackage<'p, C, Math, ConstantFold>,
    novel: bool,
}

impl<'p, C: Counter> Smallest<'p, C> {
    fn new(pp: &'p PrecomputePackage<'p, C, Math, ConstantFold>, novel: bool) -> Self {
        Self { pp, novel }
    }
}

impl<'p, C: Counter> Strategy<'p, C> for Smallest<'p, C> {
    fn run_trials(
        &self,
        args: &Args,
        eqsat: &EqsatConfig,
        seed_str: &str,
        goals: &[RecExpr<Math>],
    ) -> Vec<GoalResults> {
        goals
            .into_par_iter()
            .map(|goal| {
                tee_println!("Current goal: {goal}");
                GoalResults {
                    seed: seed_str.to_owned(),
                    goal: goal.to_string(),
                    runs: self.run_trial(args, eqsat, goal),
                }
            })
            .collect()
    }

    fn run_trial(
        &self,
        args: &Args,
        eqsat: &EqsatConfig,
        goal_recexpr: &RecExpr<Math>,
    ) -> TrialsPerK {
        let r = verify_reachability(
            std::slice::from_ref(&self.pp.smallest(self.pp.root(), self.novel)),
            goal_recexpr,
            &RULES,
            eqsat,
            args.full_union,
        )
        .map_err(Into::into);
        HashMap::from([(1, vec![r])])
    }
}
