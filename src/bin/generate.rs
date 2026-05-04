use std::fmt::Display;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use clap::Parser;
use csv::Writer;
use egg::{Analysis, BackoffScheduler, Language, Rewrite, Runner, SimpleScheduler, StopReason};
use hashbrown::{HashMap, hash_map::Entry};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::Serialize;

use rise_distance::Label;
use rise_distance::egg::ToEgg;
use rise_distance::egg::math::{BoltzmannSampler, RULES};

use rise_distance::cli::argparse::Distribution;

#[derive(Parser, Serialize)]
#[command(
    about = "Generate random math terms and write them to a CSV file",
    after_help = "\
Examples:
  # Generate 1000 uniform samples between size 5 and 50
  generate --total-samples 1000 --min-size 5 --max-size 50 --distribution uniform --seed 42 --path output.csv

  # Generate with a normal distribution (default sigma=2.6)
  generate --total-samples 1000 --min-size 5 --max-size 50 --distribution normal --seed 42 --path output.csv

  # Generate with a normal distribution and custom sigma
  generate --total-samples 1000 --min-size 5 --max-size 50 --distribution normal:3.0 --seed 42 --path output.csv

  # Adjust retry limit and Boltzmann tolerance
  generate --total-samples 500 --min-size 10 --max-size 30 --distribution uniform --seed 1 --path out.csv --tolerance 2 --retry-limit 5000

  # Use the BackoffScheduler with custom egg limits
  cargo run --release --bin generate -- --total-samples 1000 --min-size 10 --max-size 50 --distribution uniform --seed 42 --path output_new.csv --max-iters 50 --max-nodes 100000 --max-time 10 --backoff-scheduler
"
)]
struct Cli {
    /// Total number of samples to generate across all sizes
    #[arg(long)]
    total_samples: usize,

    /// Minimum term size (inclusive)
    #[arg(long)]
    min_size: usize,

    /// Maximum term size (inclusive)
    #[arg(long)]
    max_size: usize,

    /// Size tolerance for the Boltzmann sampler
    #[arg(long, default_value_t = 1)]
    tolerance: usize,

    /// RNG seed for deterministic sampling
    #[arg(long)]
    seed: u64,

    /// Maximum retries when rejecting previously seen terms (mainly relevant for small term sizes)
    #[arg(long, default_value_t = 10000)]
    retry_limit: usize,

    /// Size distribution used to allocate samples across sizes
    #[arg(long)]
    distribution: Distribution,

    /// Output CSV path
    #[arg(long)]
    path: PathBuf,

    /// Maximum egg iterations per term
    #[arg(long, default_value_t = 11)]
    max_iters: usize,

    /// Maximum egraph nodes per term
    #[arg(long, default_value_t = 100_000)]
    max_nodes: usize,

    /// Maximum runtime in seconds per term
    #[arg(long, default_value_t = 1.0)]
    max_time: f64,

    /// Number of rayon worker threads (defaults to rayon's default; lower to reduce memory pressure)
    #[arg(long)]
    parallelism: Option<usize>,

    /// Use egg's `BackoffScheduler` instead of the `SimpleScheduler`
    #[arg(long, default_value_t = false)]
    backoff_scheduler: bool,
}

const COLUMN_NAMES: [&str; 10] = [
    "size",
    "term",
    "attempts",
    "stop_reason",
    "stop_nodes",
    "stop_classes",
    "stop_time",
    "last_nodes",
    "last_classes",
    "last_time",
];

fn main() {
    let cli = Cli::parse();
    // set parallelism not too high or else memory-out
    if let Some(p) = cli.parallelism {
        rayon::ThreadPoolBuilder::new()
            .num_threads(p)
            .build_global()
            .unwrap();
    }

    let samples_per_size =
        cli.distribution
            .samples_per_size(cli.min_size, cli.max_size, cli.total_samples);

    let validity_config = (&cli).into();

    // Derive one RNG per size by advancing the root RNG sequentially, making it deterministic and ordered.
    let mut root_rng = ChaCha12Rng::seed_from_u64(cli.seed);
    let mut sized_rngs = samples_per_size
        .iter()
        .map(|(size, n)| {
            let rng = ChaCha12Rng::from_rng(&mut root_rng).expect("RNG derivation failed");
            (size, n, rng)
        })
        .collect::<Vec<_>>();
    // Sort by size to make the whole thing deterministic
    sized_rngs.sort_by_key(|(size, _, _)| *size);

    let style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} sizes ({eta})",
    )
    .expect("valid template")
    .progress_chars("=>-");

    // No parallelism, otherwise the memory hook wouldnt work correctly
    let big_collector = sized_rngs
        .into_par_iter()
        .progress_with_style(style)
        .map(|(size, n, mut rng)| {
            let sampler = BoltzmannSampler::new(*size, cli.tolerance, None);
            let mut collector = HashMap::new();
            while (collector.len() as u64) < *n {
                let mut total_attempts = 0;
                let inserted = 'retry: {
                    for _ in 0..cli.retry_limit {
                        let (candidate, validation_result, attempts) = sampler
                            .sample(&mut rng, &|t| valididty_hook(t, &validity_config, &RULES))
                            .expect("Too many failed sample attempts");
                        total_attempts += attempts;
                        if let Entry::Vacant(e) = collector.entry(candidate) {
                            e.insert((total_attempts, validation_result));
                            break 'retry true;
                        }
                    }
                    false
                };
                assert!(inserted, "Sampled previously seen term too often");
            }
            (size, collector)
        })
        .collect::<Vec<_>>();

    println!(
        "Took a total of {} attempts for {} terms.",
        big_collector
            .iter()
            .map(|x| x.1.values().map(|v| v.0).sum::<usize>())
            .sum::<usize>(),
        big_collector.iter().map(|x| x.1.len()).sum::<usize>()
    );
    let mut writer = Writer::from_path(&cli.path).expect("File does not exist");

    writer.write_record(COLUMN_NAMES).unwrap();

    for (size, terms) in big_collector {
        let size_str = size.to_string();
        for (tree, (attempts, vr)) in terms {
            writer
                .write_record([
                    &size_str,
                    &tree.to_string(),
                    &attempts.to_string(),
                    &format!("{:?}", vr.stop_reason),
                    &vr.stop_nodes.to_string(),
                    &vr.stop_classes.to_string(),
                    &vr.stop_time.to_string(),
                    &vr.last_nodes.to_string(),
                    &vr.last_classes.to_string(),
                    &vr.last_time.to_string(),
                ])
                .unwrap();
        }
    }
}

pub struct ValidationResult {
    pub stop_reason: StopReason,
    pub stop_nodes: usize,
    pub stop_classes: usize,
    pub stop_time: f64,
    pub last_nodes: usize,
    pub last_classes: usize,
    pub last_time: f64,
}

pub struct ValidityConfig {
    pub max_iters: usize,
    pub max_nodes: usize,
    pub max_time: f64,
    pub backoff_scheduler: bool,
}

impl From<&Cli> for ValidityConfig {
    fn from(cli: &Cli) -> Self {
        Self {
            max_iters: cli.max_iters,
            max_nodes: cli.max_nodes,
            max_time: cli.max_time,
            backoff_scheduler: cli.backoff_scheduler,
        }
    }
}

pub fn valididty_hook<L: Label + Language + Display, N: Analysis<L> + Default, T: ToEgg<L>>(
    tree: &T,
    config: &ValidityConfig,
    rules: &[Rewrite<L, N>],
) -> Option<ValidationResult> {
    let expr = tree.to_rec_expr();
    // egg's Runner can panic on certain malformed inside its merge check.
    // Fixing this would require only constructing correct terms and that is too complicated
    // We use catch_unwind to treat such cases as "not passing the check" rather than crashing the process.
    // An example would be the expression
    // '(cos (* (sqrt (* x (sqrt (i (/ 0 x) x)))) (sin (+ (pow 1 (/ 1 2)) (cos 2)))))'
    // The issue is that the binder check does not catch (i (/ 0 x) x) although (/ 0 x)
    // trivially simplifies to 0
    let runner = Runner::default()
        .with_expr(&expr)
        .with_iter_limit(config.max_iters)
        .with_node_limit(config.max_nodes)
        .with_time_limit(Duration::from_secs_f64(config.max_time));
    let runner = if config.backoff_scheduler {
        runner.with_scheduler(BackoffScheduler::default())
    } else {
        runner.with_scheduler(SimpleScheduler)
    };

    // Setting and unsetting the panic hook so we dont get debug spam. it is fine to ignore the output
    // Afterwards we reinstall the old default panic hook
    let start = Instant::now();
    let Ok(r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules))) else {
        println!("panic caught in iter_check_hook for expr: {expr}");
        println!("It is safe to ignore the output of egg here");
        return None;
    };
    let stop_time = start.elapsed().as_secs_f64();

    let stop_reason = r.stop_reason.clone()?;

    if matches!(
        stop_reason,
        StopReason::IterationLimit(_) | StopReason::NodeLimit(_) | StopReason::TimeLimit(_)
    ) {
        return Some(ValidationResult {
            stop_reason,
            stop_nodes: r.egraph.nodes().len(),
            stop_classes: r.egraph.classes().len(),
            stop_time,
            last_nodes: r.iterations.last()?.egraph_nodes,
            last_classes: r.iterations.last()?.egraph_classes,
            last_time: r.iterations.last()?.total_time,
        });
    }
    None
}
