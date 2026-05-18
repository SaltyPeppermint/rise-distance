use std::time::Duration;

use clap::Parser;
use egg::{Analysis, BackoffScheduler, FromOp, RecExpr, Rewrite, Runner, SimpleScheduler};
use rlimit::{Resource, setrlimit};

use rise_distance::cli::argparse::Language;
use rise_distance::egg::math::{self, Math};
use rise_distance::egg::prop::{self, Prop};
use rise_distance::MyLanguage;

#[derive(Parser)]
#[command(about = "Run eqsat on a single term and print peak RSS in bytes.")]
struct Args {
    /// Term to run eqsat on (s-expression).
    #[arg(long)]
    term: String,

    /// Language the term is drawn from.
    #[arg(long)]
    language: Language,

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
        Language::Math => run::<Math, math::ConstantFold>(&args, &math::RULES),
        Language::Prop => run::<Prop, prop::ConstantFold>(&args, &prop::RULES),
    }

    println!("{}", peak_rss_bytes());
}

fn run<L, N>(args: &Args, rules: &[Rewrite<L, N>])
where
    L: MyLanguage + FromOp + 'static,
    L::Error: std::fmt::Display,
    N: Analysis<L> + Default + 'static,
    N::Data: Clone,
{
    let expr: RecExpr<L> = args
        .term
        .parse()
        .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));

    let runner = Runner::default()
        .with_expr(&expr)
        .with_iter_limit(args.max_iters)
        .with_node_limit(args.max_nodes)
        .with_time_limit(Duration::from_secs_f64(args.max_time));
    let runner = if args.backoff_scheduler {
        runner.with_scheduler(BackoffScheduler::default())
    } else {
        runner.with_scheduler(SimpleScheduler)
    };
    let runner = runner.run(rules);

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
