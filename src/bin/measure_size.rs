use std::time::Duration;

use clap::Parser;
use egg::{BackoffScheduler, RecExpr, Runner, SimpleScheduler};
use peak_alloc::PeakAlloc;
use rise_distance::egg::math::{Math, RULES};
use rlimit::{Resource, setrlimit};

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;

#[derive(Parser)]
#[command(about = "Run eqsat on a single term and print peak heap usage in bytes.")]
struct Cli {
    /// Term to run eqsat on (s-expression).
    #[arg(long)]
    term: String,

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
    let cli = Cli::parse();

    if let Some(cap) = cli.max_memory {
        setrlimit(Resource::AS, cap, cap).expect("setrlimit RLIMIT_AS failed");
    }

    let expr: RecExpr<Math> = cli
        .term
        .parse()
        .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", cli.term));

    let runner = Runner::default()
        .with_expr(&expr)
        .with_iter_limit(cli.max_iters)
        .with_node_limit(cli.max_nodes)
        .with_time_limit(Duration::from_secs_f64(cli.max_time));
    let runner = if cli.backoff_scheduler {
        runner.with_scheduler(BackoffScheduler::default())
    } else {
        runner.with_scheduler(SimpleScheduler)
    };
    let runner = runner.run(&*RULES);

    drop(runner);

    println!("{}", ALLOC.peak_usage());
}
