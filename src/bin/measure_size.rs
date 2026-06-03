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

    /// Iter limit for the runner
    #[arg(long, default_value_t = 11)]
    max_iters: usize,

    /// Node limit for the runner
    #[arg(long, default_value_t = 100_000)]
    max_nodes: usize,

    /// Time limit for the runner (seconds)
    #[arg(long, default_value_t = 1.0)]
    max_time: f64,

    /// Hard cap on virtual address space (bytes), enforced via `RLIMIT_AS`.
    /// Allocators reserve more virtual memory than they commit, so set
    /// this generously vs. the real RSS budget.
    #[arg(long)]
    max_memory: Option<u64>,

    /// Use egg's `BackoffScheduler` instead of the `SimpleScheduler`
    #[arg(long, default_value_t = false)]
    backoff_scheduler: bool,
}

fn main() {
    let args = Args::parse();

    if let Some(cap) = args.max_memory {
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

    let config = EqsatConfig {
        max_iters: args.max_iters,
        max_nodes: args.max_nodes,
        max_time: args.max_time,
        backoff_scheduler: args.backoff_scheduler,
    };
    let runner = config.build_runner::<_, _, ()>(&expr).run(rules);

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
