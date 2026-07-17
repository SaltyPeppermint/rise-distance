use clap::Parser;
use egg::{Analysis, Iteration, IterationData, Language, RecExpr, Rewrite, Runner};
use serde::Serialize;

use rise_distance::langs::diospyros::VecLang;
use rise_distance::utils::{peak_rss_bytes, process_rss_bytes};
use rlimit::{Resource, setrlimit};

use rise_distance::eqsat::EqsatConfig;
use rise_distance::langs::AvailableLanguages;
use rise_distance::langs::diospyros;
use rise_distance::langs::math::{self, Math};
use rise_distance::langs::prop::{self, Prop};
use rise_distance::{MyAnalysis, MyLanguage};

#[derive(Parser)]
#[command(about = "Run eqsat on a single term and print per-iteration stats and peak RSS as JSON.")]
struct Args {
    /// Term to run eqsat on (s-expression).
    #[arg(long)]
    term: String,

    /// Language the term is drawn from.
    #[arg(long)]
    language: AvailableLanguages,

    #[command(flatten)]
    eqsat: EqsatConfig,

    /// Hard cap on virtual address space (bytes), enforced via `RLIMIT_AS`.
    /// Allocators reserve more virtual memory than they commit, so set this
    /// generously vs. the real RSS budget (unlike `--max-memory`, which is the
    /// graceful RSS ceiling; this one kills allocations outright).
    #[arg(long)]
    rlimit_as: Option<u64>,
}

/// Per-iteration annotation egg stores in each [`Iteration`]'s `data` slot.
/// egg calls [`IterationData::make`] once at the *start* of every iteration
/// (before search/apply), so `rss` is the process RSS (bytes) at that point,
/// aligned with egg's own start-of-iteration `egraph_nodes`/`egraph_classes`.
/// `None` if the platform RSS reader is unavailable.
#[derive(Serialize)]
struct RssData {
    rss: Option<u64>,
}

impl<L: Language, N: Analysis<L>> IterationData<L, N> for RssData {
    fn make(_runner: &Runner<L, N, Self>) -> Self {
        Self {
            rss: process_rss_bytes(),
        }
    }
}

/// The whole stdout payload: the process peak RSS (`VmHWM`, matching htop) and
/// egg's per-iteration stats, each carrying its start-of-iteration `rss` in the
/// flattened `data` slot.
#[derive(Serialize)]
struct Output {
    /// Peak resident set size of the whole process (`VmHWM`), in bytes.
    peak: u64,
    iterations: Vec<Iteration<RssData>>,
}

fn main() {
    let args = Args::parse();

    // The RLIMIT_AS backstop caps virtual address space, which always exceeds
    // RSS: a cap at or below the graceful `--max-memory` ceiling would kill the
    // process before the RSS hook ever trips.
    if let (Some(rlimit), Some(rss_limit)) = (args.rlimit_as, args.eqsat.max_memory) {
        assert!(
            rlimit > rss_limit,
            "--rlimit-as ({rlimit}) must exceed --max-memory ({rss_limit}), \
             or the hard RLIMIT_AS kill preempts the graceful RSS stop"
        );
    }
    if let Some(cap) = args.rlimit_as {
        setrlimit(Resource::AS, cap, cap).expect("setrlimit RLIMIT_AS failed");
    }

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

    // `RssData` in the `D` slot makes egg sample RSS at each iteration start and
    // stash it in `Iteration::data` for us — no hook or shared cell needed.
    let runner = args.eqsat.build_runner::<_, _, RssData>(&expr).run(rules);

    Output {
        peak: peak_rss_bytes(),
        iterations: runner.iterations,
    }
}
