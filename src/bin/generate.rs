use std::time::Instant;

use clap::{Args as ClapArgs, Parser, Subcommand};
use egg::{RecExpr, Rewrite, StopReason};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rise_distance::sampling::Distribution;
use serde::Serialize;

use rise_distance::eqsat::{EqsatConfig, HeapData, Measurement};
use rise_distance::generator::{Samplable, SizeUniformSampler};
use rise_distance::langs::{AvailableLanguages, math, prop};
use rise_distance::{MyAnalysis, MyLanguage};

#[derive(Parser)]
#[command(
    about = "Plan seed allocation or generate one exact-size validated term",
    after_help = "\
Examples:
  generate plan --total-samples 1000 --min-size 5 --max-size 50 --distribution uniform

  generate one --size 17 --seed 42 --language math \\
    --max-iters 50 --max-nodes 100000 --max-time 10
"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Emit the exact size/count allocation as JSON.
    Plan(PlanArgs),
    /// Generate and validate one term of exactly the requested size.
    One(OneArgs),
}

#[derive(ClapArgs)]
struct PlanArgs {
    /// Total number of samples to generate across all sizes
    #[arg(long)]
    total_samples: usize,

    /// Minimum term size (inclusive)
    #[arg(long)]
    min_size: usize,

    /// Maximum term size (inclusive)
    #[arg(long)]
    max_size: usize,

    /// Size distribution used to allocate samples across sizes
    #[arg(long)]
    distribution: Distribution,
}

#[derive(ClapArgs)]
struct OneArgs {
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
    match args.command {
        Command::Plan(args) => emit_plan(&args),
        Command::One(args) => generate_one(&args),
    }
}

fn emit_plan(args: &PlanArgs) {
    let sizes = (args.min_size..=args.max_size).collect::<Vec<_>>();
    let plan = args
        .distribution
        .samples_per_size(&sizes, args.total_samples);
    serde_json::to_writer(std::io::stdout().lock(), &plan).unwrap();
    println!();
}

fn generate_one(args: &OneArgs) {
    let result = match args.language {
        AvailableLanguages::Diospyros => unimplemented!("Dios has no sampler"),
        AvailableLanguages::Math => {
            run_one::<math::Math, math::ConstantFold>(args, &args.eqsat, &math::rules())
        }
        AvailableLanguages::Prop => {
            run_one::<prop::Prop, prop::ConstantFold>(args, &args.eqsat, &prop::rules())
        }
    };
    serde_json::to_writer(std::io::stdout().lock(), &result).unwrap();
    println!();
}

fn run_one<L, N>(
    args: &OneArgs,
    validity_config: &EqsatConfig,
    rules: &[Rewrite<L, N>],
) -> OneOutput
where
    L: Samplable,
    N: MyAnalysis<L> + Default,
{
    let sampler = SizeUniformSampler::<L>::new(args.size, None);
    let mut rng = ChaCha12Rng::seed_from_u64(args.seed);
    for attempts in 1..=args.retry_limit {
        let candidate = sampler.sample(&mut rng);
        if let Some((validation, measurement)) = validity_hook(&candidate, validity_config, rules) {
            return OneOutput {
                term: candidate.to_string(),
                payload: (attempts, validation, measurement),
            };
        }
    }
    panic!(
        "no valid term of size {} within {} attempts",
        args.size, args.retry_limit
    );
}

#[derive(Serialize)]
struct OneOutput {
    term: String,
    payload: (usize, ValidationResult, Measurement),
}

#[derive(Serialize)]
pub struct ValidationResult {
    pub stop_reason: StopReason,
    pub stop_nodes: usize,
    pub stop_classes: usize,
    pub stop_time: f64,
    pub last_nodes: usize,
    pub last_classes: usize,
    pub last_time: f64,
    pub iterations: usize,
}

#[must_use]
pub fn validity_hook<L: MyLanguage, N: MyAnalysis<L> + Default>(
    expr: &RecExpr<L>,
    config: &EqsatConfig,
    rules: &[Rewrite<L, N>],
) -> Option<(ValidationResult, Measurement)> {
    let runner = config.build_runner::<_, _, HeapData>(expr);

    // Treat runner panics as failed validation.
    let start = Instant::now();
    let Ok(r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules))) else {
        eprintln!("panic caught in iter_check_hook for expr: {expr}");
        eprintln!("It is safe to ignore the output of egg here");
        return None;
    };
    let stop_time = start.elapsed().as_secs_f64();

    let result = (|| {
        let stop_reason = r.stop_reason.clone()?;
        // Resource exhaustion passes; saturation does not.
        let hit_limit = matches!(
            stop_reason,
            StopReason::IterationLimit(_)
                | StopReason::NodeLimit(_)
                | StopReason::TimeLimit(_)
                | StopReason::MemoryLimit(_)
        );
        if !hit_limit {
            return None;
        }
        let validation = ValidationResult {
            stop_reason,
            stop_nodes: r.egraph.nodes().len(),
            stop_classes: r.egraph.classes().len(),
            stop_time,
            last_nodes: r.iterations.last()?.egraph_nodes,
            last_classes: r.iterations.last()?.egraph_classes,
            last_time: r.iterations.last()?.total_time,
            iterations: r.iterations.len(),
        };
        Some(validation)
    })();

    let report = r.final_memory_report()?;
    result.map(|validation| (validation, Measurement::from_run(report, r.iterations)))
}
