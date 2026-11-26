use std::fmt::Display;
use std::time::Instant;

use clap::Parser;
use egg::{Analysis, Iteration, Language, RecExpr, Rewrite, StopReason};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use rise_distance::cli::Measured;
use rise_distance::eqsat::EqsatConfig;
use rise_distance::generator::{Samplable, SizeUniformSampler};
use rise_distance::langs::{Lang, math, prop};

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

    /// Maximum samples draws before giving up.
    #[arg(long, default_value_t = 10000)]
    retry_limit: usize,

    /// Language to sample terms from
    #[arg(long)]
    language: Lang,

    /// Eqsat limits for this term's validity check.
    #[command(flatten)]
    eqsat: EqsatConfig,
}

fn main() {
    let args = Args::parse();
    let result = match args.language {
        Lang::Math => run_one(&args, &args.eqsat, &math::rules()),
        Lang::Prop => run_one(&args, &args.eqsat, &prop::rules()),
    };
    serde_json::to_writer(std::io::stdout().lock(), &Measured::now(result)).unwrap();
    println!();
}

fn run_one<L: Samplable + Display, N: Analysis<L> + Default>(
    args: &Args,
    validity_config: &EqsatConfig,
    rules: &[Rewrite<L, N>],
) -> StartTerm {
    let sampler = SizeUniformSampler::<L>::new(args.size, None);
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    for draws in 1..=args.retry_limit {
        let sample = sampler.sample(&mut rng);
        if let Some(measurement) = validity_check(&sample, validity_config, rules) {
            return StartTerm {
                term: sample.to_string(),
                draws,
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
struct StartTerm {
    term: String,
    /// How many were drawn until one passed
    draws: usize,
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
pub fn validity_check<L: Language + Display, N: Analysis<L> + Default>(
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
