use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use egg::{Extractor, RecExpr};

use num::BigUint;
use rise_distance::eqsat::{EqsatConfig, EqsatMetadata, EqsatResult};
use rise_distance::langs::diospyros::VecLang;
use rise_distance::langs::diospyros::cost::VecCostFn;
use rise_distance::langs::diospyros::rewriteconcats::list_to_concats;
use rise_distance::langs::diospyros::rules::{filter_applicable_rules, rules};
use rise_distance::langs::diospyros::stringconversion::convert_string;
use rise_distance::sampling::{PrecomputePackage, SampleStrategy, TermSampleDist};
use rise_distance::{eqsat, lower};

#[derive(Parser)]
#[command(
    about = "Cost-minimizing vectorisation search over Diospyros benchmarks",
    after_help = "\
Examples:
  # Brute-force a single benchmark (time limit 180s):
  diospyros --input data/diospyros/mat_mul_2x2_2x2.sexp brute --timeout 180
  # Run all benchmarks in brute mode:
  diospyros --all brute --timeout 120
  # Cut after 10 iterations, sample 50 frontier terms, continue from each:
  diospyros --input data/diospyros/conv_2d_3x3_3x3.sexp cut --cut-iters 10 --sample-count 50
"
)]
struct Args {
    /// Path to a single .sexp benchmark file.
    #[arg(long, conflicts_with = "all")]
    input: Option<PathBuf>,

    /// Run every .sexp file found in data/diospyros/.
    #[arg(long, conflicts_with = "input")]
    all: bool,

    /// Disable associativity/commutativity rules.
    #[arg(long)]
    no_ac: bool,

    /// Disable vector rules.
    #[arg(long)]
    no_vec: bool,

    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand, Clone, Debug)]
enum Mode {
    /// Grow one continuous egraph and extract the cheapest term.
    Brute(BruteArgs),
    /// Grow the egraph to a cut point, sample novel frontier terms, restart
    /// eqsat from each, and keep the best cost found across all restarts.
    Cut(CutArgs),
}

#[derive(clap::Args, Clone, Debug)]
struct BruteArgs {
    /// Wall-clock timeout in seconds.
    #[arg(long, default_value_t = 180.0)]
    timeout: f64,

    /// Maximum egraph nodes before stopping.
    #[arg(long, default_value_t = 10_000_000)]
    max_nodes: usize,

    /// Maximum eqsat iterations.
    #[arg(long, default_value_t = 10_000)]
    max_iters: usize,
}

#[derive(clap::Args, Clone, Debug)]
struct CutArgs {
    /// Stop the first phase after this many iterations and sample the frontier.
    #[arg(long, default_value_t = 10)]
    cut_iters: usize,

    /// Maximum egraph nodes per phase.
    #[arg(long, default_value_t = 10_000_000)]
    max_nodes: usize,

    /// Wall-clock timeout (seconds) per phase.
    #[arg(long, default_value_t = 60.0)]
    timeout: f64,

    /// Maximum eqsat iterations in the second (verify) phase.
    #[arg(long, default_value_t = 10_000)]
    verify_iters: usize,

    /// Number of novel frontier terms to sample at the cut point.
    #[arg(long, default_value_t = 50)]
    sample_count: usize,

    /// Maximum term size considered when enumerating frontier terms.
    #[arg(long, default_value_t = 30)]
    max_size: usize,
}

struct RunResult {
    cost: f64,
    best: RecExpr<VecLang>,
    meta: Vec<EqsatMetadata>,
}

fn main() {
    let args = Args::parse();

    let benchmarks = if args.all {
        let dir = PathBuf::from("data/diospyros");
        let mut paths = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("Cannot read {}: {e}", dir.display()))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "sexp"))
            .collect::<Vec<_>>();
        paths.sort();
        paths
    } else if let Some(p) = args.input {
        vec![p]
    } else {
        eprintln!("Provide --input <file> or --all");
        std::process::exit(1);
    };

    for bench in &benchmarks {
        println!("\n=== {} ===", bench.display());
        run_benchmark(bench, &args.mode, args.no_ac, args.no_vec);
    }
}

fn load_benchmark(path: &Path) -> RecExpr<VecLang> {
    let display = path.display();
    let src =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("Cannot read {display}: {e}"));
    let converted = convert_string(&src)
        .unwrap_or_else(|e| panic!("string conversion failed for {display}: {e}"));
    list_to_concats(&converted)
        .unwrap_or_else(|e| panic!("list_to_concats failed for {display}: {e}"))
        .parse()
        .unwrap_or_else(|e| panic!("parse failed for {display}: {e}"))
}

fn run_benchmark(path: &Path, mode: &Mode, no_ac: bool, no_vec: bool) {
    let prog = load_benchmark(path);
    let result = match mode {
        Mode::Brute(a) => run_brute(&prog, a, no_ac, no_vec),
        Mode::Cut(a) => run_cut(&prog, a, no_ac, no_vec),
    };

    let Some(result) = result else {
        warn(&format!("SKIPPING {}. Search failed", path.display()));
        return;
    };

    println!("Best cost: {:.6}", result.cost);
    println!("Best expr:\n{}", result.best.pretty(80));
    for (i, m) in result.meta.iter().enumerate() {
        println!(
            "phase {i}: {} nodes, {} classes, {:.2}s, {} iters",
            m.nodes, m.classes, m.time, m.iters,
        );
    }
}

// ---------------------------------------------------------------------------

/// Print an impossible-to-miss banner to stderr.
fn warn(msg: &str) {
    eprintln!("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    eprintln!("!! {msg}");
    eprintln!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
}

const fn config(max_iters: usize, max_nodes: usize, timeout: f64) -> EqsatConfig {
    EqsatConfig {
        max_iters,
        max_nodes,
        max_time: timeout,
        backoff_scheduler: false,
    }
}

/// Extract the cheapest term from an eqsat result.
fn extract(result: EqsatResult<VecLang, ()>) -> (EqsatMetadata, f64, RecExpr<VecLang>) {
    let meta = EqsatMetadata::from_iterations(result.data());
    let (eg, root) = result.into_curr();
    let (cost, expr) = Extractor::new(&eg, VecCostFn { egraph: &eg }).find_best(root);
    (meta, cost, expr)
}

// ---------------------------------------------------------------------------

fn run_brute(
    prog: &RecExpr<VecLang>,
    args: &BruteArgs,
    no_ac: bool,
    no_vec: bool,
) -> Option<RunResult> {
    let mut rule_set = rules(no_ac, no_vec);
    filter_applicable_rules(&mut rule_set, prog);

    let cfg = config(args.max_iters, args.max_nodes, args.timeout);
    let Some(result) = eqsat::run_eqsat::<VecLang, (), _>(prog, rule_set.iter(), &cfg) else {
        warn("run_eqsat returned None. Not enough distinct iterations");
        return None;
    };
    println!("Stopped with stop reason: {:?}", result.stop_reason());

    let (meta, cost, best) = extract(result);
    Some(RunResult {
        cost,
        best,
        meta: vec![meta],
    })
}

// ---------------------------------------------------------------------------

fn run_cut(
    prog: &RecExpr<VecLang>,
    args: &CutArgs,
    no_ac: bool,
    no_vec: bool,
) -> Option<RunResult> {
    let mut rule_set = rules(no_ac, no_vec);
    filter_applicable_rules(&mut rule_set, prog);

    // Phase 1: grow to the cut point, then sample novel frontier terms.
    let cut_cfg = config(args.cut_iters, args.max_nodes, args.timeout);
    let Some(cut_result) = eqsat::run_eqsat::<VecLang, (), _>(prog, rule_set.iter(), &cut_cfg)
    else {
        warn("Cut phase 1 returned None");
        return None;
    };
    println!(
        "Cut phase 1 stopped with stop reason: {:?}",
        cut_result.stop_reason()
    );
    let cut_meta = EqsatMetadata::from_iterations(cut_result.data());

    let Some(pp) =
        PrecomputePackage::<BigUint, VecLang, ()>::precompute(&cut_result, args.max_size)
    else {
        warn("Precompute returned None (empty frontier)");
        return None;
    };

    let Some(sampled) = pp.sample_frontier_terms(
        args.sample_count,
        TermSampleDist::UNIFORM,
        SampleStrategy::Count,
        [args.cut_iters as u64, 0],
        true,
    ) else {
        warn("Sampling failed");
        return None;
    };

    println!(
        "Cut: sampled {} frontier terms after {} iters",
        sampled.len(),
        cut_result.iters()
    );

    // Phase 2: run eqsat from each sampled term and keep the best cost.
    let verify_cfg = config(args.verify_iters, args.max_nodes, args.timeout);
    let runs = sampled.iter().enumerate().filter_map(|(i, sample)| {
        let start = lower(sample.clone());
        eqsat::run_eqsat::<VecLang, (), _>(&start, &rule_set, &verify_cfg)
            .or_else(|| {
                warn(&format!("Sample {i}: run_eqsat returned None, skipping"));
                None
            })
            .map(|result| {
                println!(
                    "Sample {i}: stopped with stop reason: {:?}",
                    result.stop_reason()
                );
                extract(result)
            })
    });

    let mut meta = vec![cut_meta];
    let mut best = None;
    for (vmeta, cost, expr) in runs {
        meta.push(vmeta);
        if best.as_ref().is_none_or(|(b, _)| cost < *b) {
            best = Some((cost, expr));
        }
    }

    let Some((cost, best)) = best else {
        warn("Cut: every sampled restart failed");
        return None;
    };
    Some(RunResult { cost, best, meta })
}
