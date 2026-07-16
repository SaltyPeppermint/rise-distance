use clap::Parser;
use egg::{RecExpr, Rewrite};

use rise_distance::langs::diospyros::VecLang;
use rlimit::{Resource, setrlimit};

use rise_distance::eqsat::EqsatConfig;
use rise_distance::langs::AvailableLanguages;
use rise_distance::langs::diospyros;
use rise_distance::langs::math::{self, Math};
use rise_distance::langs::prop::{self, Prop};
use rise_distance::{MyAnalysis, MyLanguage};

#[derive(Parser)]
#[command(about = "Run eqsat on a single term and print peak RSS in bytes.")]
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

    match args.language {
        AvailableLanguages::Diospyros => {
            run::<VecLang, ()>(&args, &diospyros::rules(false, false));
        }
        AvailableLanguages::Math => run::<Math, math::ConstantFold>(&args, &math::silly_rules()),
        AvailableLanguages::Prop => run::<Prop, prop::ConstantFold>(&args, &prop::rules()),
    }

    println!("{}", peak_rss_bytes());
}

fn run<L: MyLanguage, N: MyAnalysis<L>>(args: &Args, rules: &[Rewrite<L, N>]) {
    let expr: RecExpr<L> = args
        .term
        .parse()
        .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));

    let runner = args.eqsat.build_runner::<_, _, ()>(&expr).run(rules);

    drop(runner);
}

/// Peak resident set size of this process in bytes, read from
/// `/proc/self/status` (`VmHWM`). Matches what htop reports.
fn peak_rss_bytes() -> u64 {
    let status =
        std::fs::read_to_string("/proc/self/status").expect("failed to read /proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .expect("malformed VmHWM line");
            return kb * 1024;
        }
    }
    panic!("VmHWM not found in /proc/self/status");
}
