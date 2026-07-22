use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use egg::{Analysis, Iteration, IterationData, Language, RecExpr, Rewrite, Runner};
use serde::Serialize;

use rise_distance::langs::diospyros::VecLang;
use rise_distance::utils::live_heap_bytes;

use rise_distance::eqsat::EqsatConfig;
use rise_distance::langs::AvailableLanguages;
use rise_distance::langs::diospyros;
use rise_distance::langs::math::{self, Math};
use rise_distance::langs::prop::{self, Prop};
use rise_distance::{MyAnalysis, MyLanguage};

#[derive(Parser)]
#[command(
    about = "Run eqsat on a single term and print per-iteration stats and live-heap use as JSON."
)]
struct Args {
    /// Term to run eqsat on (s-expression).
    #[arg(long)]
    term: String,

    /// Language the term is drawn from.
    #[arg(long)]
    language: AvailableLanguages,

    #[command(flatten)]
    eqsat: EqsatConfig,
}

/// Live-heap baseline (jemalloc `stats.allocated`) captured before the eqsat and
/// subtracted from each iteration's reading, so `allocated` isolates the
/// egraph's footprint. Set in [`run`] before the runner is built.
static PRE_HEAP: AtomicU64 = AtomicU64::new(0);

/// Per-iteration annotation egg stores in each [`Iteration`]'s `data` slot.
/// egg calls [`IterationData::make`] at the *start* of every iteration, so
/// `allocated` is the live-heap growth over [`PRE_HEAP`] there, aligned with
/// egg's own start-of-iteration `egraph_nodes`/`egraph_classes`.
#[derive(Serialize)]
struct HeapData {
    allocated: u64,
}

impl<L: Language, N: Analysis<L>> IterationData<L, N> for HeapData {
    fn make(_runner: &Runner<L, N, Self>) -> Self {
        Self {
            allocated: live_heap_bytes().saturating_sub(PRE_HEAP.load(Ordering::Relaxed)),
        }
    }
}

/// The whole stdout payload: egg's per-iteration stats, each carrying its
/// start-of-iteration live-heap growth (over the pre-eqsat baseline) in the
/// flattened `data` slot.
#[derive(Serialize)]
struct Output {
    iterations: Vec<Iteration<HeapData>>,
}

fn main() {
    let args = Args::parse();

    let output = match args.language {
        AvailableLanguages::Diospyros => {
            run::<VecLang, ()>(&args, &diospyros::rules(false, false))
        }
        AvailableLanguages::Math => run::<Math, math::ConstantFold>(&args, &math::silly_rules()),
        AvailableLanguages::Prop => run::<Prop, prop::ConstantFold>(&args, &prop::rules()),
    };

    // Print the payload as a single JSON object (the whole stdout).
    println!(
        "{}",
        serde_json::to_string(&output).expect("serializing measure-size output failed")
    );
}

fn run<L: MyLanguage, N: MyAnalysis<L>>(args: &Args, rules: &[Rewrite<L, N>]) -> Output {
    let expr: RecExpr<L> = args
        .term
        .parse()
        .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));

    // Baseline for `HeapData`'s per-iteration deltas (see `PRE_HEAP`).
    PRE_HEAP.store(live_heap_bytes(), Ordering::Relaxed);

    // `HeapData` in the `D` slot makes egg sample live-heap bytes at each
    // iteration start and stash it in `Iteration::data` — no hook or shared cell
    // needed.
    let runner = args.eqsat.build_runner::<_, _, HeapData>(&expr).run(rules);

    Output {
        iterations: runner.iterations,
    }
}
