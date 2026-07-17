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

use rise_distance::eqsat::{EqsatConfig, HEAP_TRIM_AVAILABLE, trim_heap};
use rise_distance::generator::BoltzmannSampler;
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

    /// Language to sample terms from
    #[arg(long)]
    language: AvailableLanguages,

    /// Output JSON path
    #[arg(long)]
    path: PathBuf,

    /// Eqsat limits for each term's validity check (bound each per-term run,
    /// not the whole generation).
    #[command(flatten)]
    eqsat: EqsatConfig,
}

fn main() {
    let args = Args::parse();

    // The `--max-memory` gate compares absolute process RSS against the target,
    // which is only meaningful if we can hand freed pages back to the OS between
    // terms (`trim_heap`). Off glibc that trim is a no-op, so leftover pages
    // from earlier terms would pollute later RSS readings and silently corrupt
    // the gate. Refuse to run rather than produce wrong results.
    assert!(
        args.eqsat.max_memory.is_none() || HEAP_TRIM_AVAILABLE,
        "--max-memory needs heap trimming (glibc's malloc_trim), which is \
         unavailable on this target (non-glibc allocator). Build against glibc \
         to use the memory ceiling, or omit --max-memory."
    );

    let sizes = (args.min_size..=args.max_size).collect::<Vec<_>>();
    let samples_per_size = args
        .distribution
        .samples_per_size(&sizes, args.total_samples);

    // Derive one RNG per size by advancing the root RNG sequentially, making it deterministic and ordered.
    let mut root_rng = ChaCha12Rng::seed_from_u64(args.seed);
    let mut sized_rngs = samples_per_size
        .iter()
        .map(|(size, n)| {
            let rng = ChaCha12Rng::from_rng(&mut root_rng).expect("RNG derivation failed");
            (*size, *n, rng)
        })
        .collect::<Vec<_>>();
    // Sort by size to make the whole thing deterministic
    sized_rngs.sort_by_key(|(size, _, _)| *size);

    let big_collector = match args.language {
        AvailableLanguages::Diospyros => unimplemented!("Dios has no sampler"),
        AvailableLanguages::Math => run_language::<math::MathSampler, math::ConstantFold>(
            &args,
            &args.eqsat,
            sized_rngs,
            &math::rules(),
        ),
        AvailableLanguages::Prop => run_language::<prop::PropSampler, prop::ConstantFold>(
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

type SizeBucket = (usize, HashMap<String, (usize, ValidationResult)>);

fn run_language<S, N>(
    args: &Args,
    validity_config: &EqsatConfig,
    sized_rngs: Vec<(usize, u64, ChaCha12Rng)>,
    rules: &[Rewrite<S::Lang, N>],
) -> Vec<SizeBucket>
where
    S: BoltzmannSampler,
    S::Lang: 'static,
    N: MyAnalysis<S::Lang> + Default,
{
    let style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} sizes ({eta})",
    )
    .expect("valid template")
    .progress_chars("=>-");

    sized_rngs
        .into_iter()
        .map(|(size, n, mut rng)| {
            let collector = collect_for_size::<S, N>(
                size,
                n,
                &mut rng,
                args.tolerance,
                args.retry_limit,
                validity_config,
                rules,
            );
            (size, collector)
        })
        .progress_with_style(style)
        .collect()
}

fn collect_for_size<S, N>(
    size: usize,
    n: u64,
    rng: &mut ChaCha12Rng,
    tolerance: usize,
    retry_limit: usize,
    validity_config: &EqsatConfig,
    rules: &[Rewrite<S::Lang, N>],
) -> HashMap<String, (usize, ValidationResult)>
where
    S: BoltzmannSampler,
    N: MyAnalysis<S::Lang>,
{
    let sampler = S::new(size, tolerance, None);
    let mut collector = HashMap::new();
    while (collector.len() as u64) < n {
        let mut total_attempts = 0;
        let inserted = 'retry: {
            for _ in 0..retry_limit {
                let (candidate, validation_result, attempts) = sampler
                    .sample(rng, &|t| valididty_hook(t, validity_config, rules))
                    .expect("Too many failed sample attempts");
                let candidate_str = candidate.to_string();
                total_attempts += attempts;
                if let Entry::Vacant(e) = collector.entry(candidate_str) {
                    e.insert((total_attempts, validation_result));
                    break 'retry true;
                }
            }
            false
        };
        assert!(inserted, "Sampled previously seen term too often");
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
) -> Option<ValidationResult> {
    // egg's Runner can panic on certain malformed inside its merge check.
    // Fixing this would require only constructing correct terms and that is too complicated
    // We use catch_unwind to treat such cases as "not passing the check" rather than crashing the process.
    // An example would be the expression
    // '(cos (* (sqrt (* x (sqrt (i (/ 0 x) x)))) (sin (+ (pow 1 (/ 1 2)) (cos 2)))))'
    // The issue is that the binder check does not catch (i (/ 0 x) x) although (/ 0 x)
    // trivially simplifies to 0
    let runner = config.build_runner::<_, _, ()>(expr);

    // Setting and unsetting the panic hook so we dont get debug spam. it is fine to ignore the output
    // Afterwards we reinstall the old default panic hook
    let start = Instant::now();
    let Ok(r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules))) else {
        println!("panic caught in iter_check_hook for expr: {expr}");
        println!("It is safe to ignore the output of egg here");
        return None;
    };
    let stop_time = start.elapsed().as_secs_f64();

    // We keep a term only if it can grow an egraph to the `--max-memory`
    // target: the per-term RSS hook stops the run once absolute process RSS
    // crosses that ceiling, which egg surfaces as `StopReason::Other` carrying
    // the `memory_limit_hook` message (see `eqsat::memory_limit_hook`). Every
    // other stop reason means the term saturated or hit a safety backstop
    // (iteration/node/time) *before* reaching the target, so it is too easy and
    // we reject it.
    //
    // Extract everything we need up front, then drop the runner and trim the
    // heap *before* returning: every attempt (accepted or rejected) builds a
    // fresh egraph, and glibc retains those freed pages, so without trimming a
    // later attempt's RSS would be polluted by earlier ones and the absolute
    // ceiling would misfire. Trimming here keeps each attempt's RSS baseline
    // clean.
    let result = (|| {
        let stop_reason = r.stop_reason.clone()?;
        // Accept a term if its run was cut off by any resource limit. In
        // practice the iteration/node/time limits are set so high that only the
        // `--max-memory` ceiling fires (surfaced as `StopReason::Other` with the
        // `memory_limit_hook` message), but hitting any of them still means the
        // term is hard enough to keep. Only saturation (the term simplifies away
        // before exhausting any budget) is rejected.
        let hit_limit = matches!(
            stop_reason,
            StopReason::IterationLimit(_) | StopReason::NodeLimit(_) | StopReason::TimeLimit(_)
        ) || matches!(&stop_reason, StopReason::Other(s) if s.contains("memory limit exceeded"));
        if !hit_limit {
            return None;
        }
        Some(ValidationResult {
            stop_reason,
            stop_nodes: r.egraph.nodes().len(),
            stop_classes: r.egraph.classes().len(),
            stop_time,
            last_nodes: r.iterations.last()?.egraph_nodes,
            last_classes: r.iterations.last()?.egraph_classes,
            last_time: r.iterations.last()?.total_time,
            iterations: r.iterations.len(),
        })
    })();

    drop(r);
    trim_heap();
    result
}
