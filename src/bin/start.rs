use std::time::Instant;

use clap::Parser;
use egg::{Iteration, RecExpr, Rewrite, StopReason};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::Serialize;

use rise_distance::cli::Measured;
use rise_distance::eqsat::EqsatConfig;
use rise_distance::generator::{Samplable, SizeUniformSampler};
use rise_distance::langs::{AvailableLanguages, math, prop};
use rise_distance::{MyAnalysis, MyLanguage};

#[derive(Parser)]
#[command(
    about = "Generate one exact-size validated term",
    after_help = "\
Examples:

  start --size 17 --seed 42 --language math \\
    --max-iters 50 --max-nodes 100000 --max-time 10
"
)]
struct Args {
    /// Exact term size to generate.
    #[arg(long)]
    size: usize,

    /// RNG seed for deterministic sampling.
    #[arg(long)]
    seed: u64,

    /// Maximum candidate draws before giving up.
    #[arg(long, default_value_t = 10000)]
    retry_limit: usize,

    /// Language to sample terms from
    #[arg(long)]
    language: AvailableLanguages,

    /// Eqsat limits for this term's validity check.
    #[command(flatten)]
    eqsat: EqsatConfig,
}

fn main() {
    let args = Args::parse();
    let result = match args.language {
        AvailableLanguages::Diospyros => unimplemented!("Dios has no sampler"),
        AvailableLanguages::Math => run_one(&args, &args.eqsat, &math::rules()),
        AvailableLanguages::Prop => run_one(&args, &args.eqsat, &prop::rules()),
    };
    serde_json::to_writer(std::io::stdout().lock(), &Measured::now(result)).unwrap();
    println!();
}

fn run_one<L: Samplable, N: MyAnalysis<L>>(
    args: &Args,
    validity_config: &EqsatConfig,
    rules: &[Rewrite<L, N>],
) -> GoalTerm {
    let sampler = SizeUniformSampler::<L>::new(args.size, None);
    let mut rng = ChaCha12Rng::seed_from_u64(args.seed);
    for attempts in 1..=args.retry_limit {
        let candidate = sampler.sample(&mut rng);
        if let Some(measurement) = validity_check(&candidate, validity_config, rules) {
            return GoalTerm {
                term: candidate.to_string(),
                attempt: attempts,
                payload: measurement,
            };
        }
    }
    panic!(
        "no valid term of size {} within {} attempts",
        args.size, args.retry_limit
    );
}

#[derive(Serialize)]
struct GoalTerm {
    term: String,
    attempt: usize,
    payload: Measurement,
}

/// Eqsat iterations and absolute process live-heap measurements.
/// Byte counts use the same scale as `memory_limit`.
#[derive(Debug, Serialize)]
pub struct Measurement {
    pub iterations: Vec<Iteration<()>>,
    pub eqsat_mem_tracking_allocated: u64,
    pub eqsat_mem_tracking_peak_allocated: u64,
    pub eqsat_memory_limit: Option<u64>,
    pub stop_time: f64,
    pub stop_reason: StopReason,
}

#[must_use]
pub fn validity_check<L: MyLanguage, N: MyAnalysis<L> + Default>(
    expr: &RecExpr<L>,
    config: &EqsatConfig,
    rules: &[Rewrite<L, N>],
) -> Option<Measurement> {
    let runner = config.build_runner().with_expr(expr);

    let start = Instant::now();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules)))
        .map_err(|_panic| {
            eprintln!("panic caught in iter_check_hook for expr: {expr}");
            eprintln!("It is safe to ignore the output of egg here");
        })
        .ok()?;
    let stop_time = start.elapsed().as_secs_f64();

    // Resource exhaustion passes; saturation does not.
    if !matches!(
        r.stop_reason.as_ref()?,
        StopReason::IterationLimit(_)
            | StopReason::NodeLimit(_)
            | StopReason::TimeLimit(_)
            | StopReason::MemoryLimit(_)
    ) {
        return None;
    }

    let mem_report = r.final_memory_report()?;
    let stop_reason = r.stop_reason?;

    Some(Measurement {
        iterations: r.iterations,
        eqsat_mem_tracking_allocated: mem_report.final_reading,
        eqsat_mem_tracking_peak_allocated: mem_report.peak_reading,
        eqsat_memory_limit: mem_report.absolute_limit,
        stop_time,
        stop_reason,
    })
}
