use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use egg::{RecExpr, Rewrite, StopReason};
use hashbrown::{HashMap, hash_map::Entry};
use indicatif::{ProgressIterator, ProgressStyle};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rise_distance::sampling::Distribution;
use serde::Serialize;

use rise_distance::eqsat::{EqsatConfig, HeapData, Measurement};
use rise_distance::generator::{Samplable, SizeUniformSampler};
use rise_distance::langs::{AvailableLanguages, math, prop};
use rise_distance::{MyAnalysis, MyLanguage};

#[derive(Parser, Serialize)]
#[command(
    about = "Generate random math terms and write them to a JSON file",
    after_help = "\
The --max-* eqsat limits bound each term's validity check, not the whole run.
Examples:
  # Generate 1000 uniform samples between size 5 and 50
  generate --total-samples 1000 --min-size 5 --max-size 50 --distribution uniform \\
    --seed 42 --path output.json --max-iters 11 --max-nodes 100000 --max-time 1

  # Normal distribution with custom sigma, backoff scheduler, custom limits
  generate --total-samples 1000 --min-size 5 --max-size 50 --distribution normal:3.0 \\
    --seed 42 --path output.json --max-iters 50 --max-nodes 100000 --max-time 10 \\
    --backoff-scheduler
"
)]
struct Args {
    /// Total number of samples to generate across all sizes
    #[arg(long)]
    total_samples: usize,

    /// Minimum term size (inclusive)
    #[arg(long)]
    min_size: usize,

    /// Maximum term size (inclusive)
    #[arg(long)]
    max_size: usize,

    /// RNG seed for deterministic sampling
    #[arg(long)]
    seed: u64,

    /// Maximum draws per distinct, validated term
    #[arg(long, default_value_t = 10000)]
    retry_limit: usize,

    /// Size distribution used to allocate samples across sizes
    #[arg(long)]
    distribution: Distribution,

    /// Language to sample terms from
    #[arg(long)]
    language: AvailableLanguages,

    /// Output JSON path
    #[arg(long)]
    path: PathBuf,

    /// Eqsat limits for each term's validity check
    #[command(flatten)]
    eqsat: EqsatConfig,
}

fn main() {
    let args = Args::parse();

    let sizes = (args.min_size..=args.max_size).collect::<Vec<_>>();
    let samples_per_size = args
        .distribution
        .samples_per_size(&sizes, args.total_samples);

    // Give each size a deterministic RNG.
    let mut root_rng = ChaCha12Rng::seed_from_u64(args.seed);
    let mut sized_rngs = samples_per_size
        .iter()
        .map(|(size, n)| {
            let rng = ChaCha12Rng::from_rng(&mut root_rng).expect("RNG derivation failed");
            (*size, *n, rng)
        })
        .collect::<Vec<_>>();
    sized_rngs.sort_by_key(|(size, _, _)| *size);

    let big_collector = match args.language {
        AvailableLanguages::Diospyros => unimplemented!("Dios has no sampler"),
        AvailableLanguages::Math => run_language::<math::Math, math::ConstantFold>(
            &args,
            &args.eqsat,
            sized_rngs,
            &math::rules(),
        ),
        AvailableLanguages::Prop => run_language::<prop::Prop, prop::ConstantFold>(
            &args,
            &args.eqsat,
            sized_rngs,
            &prop::rules(),
        ),
    };

    println!(
        "Took a total of {} attempts for {} terms.",
        big_collector
            .iter()
            .map(|x| x.1.values().map(|v| v.0).sum::<usize>())
            .sum::<usize>(),
        big_collector.iter().map(|x| x.1.len()).sum::<usize>()
    );
    let mut writer = BufWriter::new(File::create(&args.path).unwrap());

    serde_json::to_writer(&mut writer, &big_collector).unwrap();
    writer.flush().unwrap();
}

type SizeBucket = (
    usize,
    HashMap<String, (usize, ValidationResult, Measurement)>,
);

fn run_language<L, N>(
    args: &Args,
    validity_config: &EqsatConfig,
    sized_rngs: Vec<(usize, u64, ChaCha12Rng)>,
    rules: &[Rewrite<L, N>],
) -> Vec<SizeBucket>
where
    L: Samplable,
    N: MyAnalysis<L> + Default,
{
    let style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} sizes ({eta})",
    )
    .expect("valid template")
    .progress_chars("=>-");

    sized_rngs
        .into_iter()
        .map(|(size, n, mut rng)| {
            let collector = collect_for_size::<L, N>(
                size,
                n,
                &mut rng,
                args.retry_limit,
                validity_config,
                rules,
            );
            (size, collector)
        })
        .progress_with_style(style)
        .collect()
}

/// Collect `n` distinct validated terms of size exactly `size`.
fn collect_for_size<L, N>(
    size: usize,
    n: u64,
    rng: &mut ChaCha12Rng,
    retry_limit: usize,
    validity_config: &EqsatConfig,
    rules: &[Rewrite<L, N>],
) -> HashMap<String, (usize, ValidationResult, Measurement)>
where
    L: Samplable,
    N: MyAnalysis<L>,
{
    let sampler = SizeUniformSampler::<L>::new(size, None);
    let mut collector = HashMap::new();
    while (collector.len() as u64) < n {
        let mut attempts = 0;
        let inserted = 'retry: {
            for _ in 0..retry_limit {
                attempts += 1;
                let candidate = sampler.sample(rng);
                let Some((validation_result, measurement)) =
                    valididty_hook(&candidate, validity_config, rules)
                else {
                    continue;
                };
                if let Entry::Vacant(e) = collector.entry(candidate.to_string()) {
                    e.insert((attempts, validation_result, measurement));
                    break 'retry true;
                }
            }
            false
        };
        assert!(
            inserted,
            "no new valid term of size {size} within {retry_limit} attempts"
        );
    }
    collector
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
pub fn valididty_hook<L: MyLanguage, N: MyAnalysis<L> + Default>(
    expr: &RecExpr<L>,
    config: &EqsatConfig,
    rules: &[Rewrite<L, N>],
) -> Option<(ValidationResult, Measurement)> {
    // build_runner captures the sole baseline before constructing the runner
    // and returns it for measurement after the run.
    let (runner, heap) = config.build_runner::<_, _, HeapData>(expr);

    // Treat runner panics as failed validation.
    let start = Instant::now();
    let Ok(r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules))) else {
        println!("panic caught in iter_check_hook for expr: {expr}");
        println!("It is safe to ignore the output of egg here");
        return None;
    };
    let stop_time = start.elapsed().as_secs_f64();

    let result = (|| {
        let stop_reason = r.stop_reason.clone()?;
        // Resource exhaustion passes; saturation does not.
        let hit_limit = matches!(
            stop_reason,
            StopReason::IterationLimit(_) | StopReason::NodeLimit(_) | StopReason::TimeLimit(_)
        ) || matches!(&stop_reason, StopReason::Other(s) if s.contains("memory limit exceeded"));
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

    result.map(|validation| (validation, Measurement::from_run(heap, r.iterations)))
}
