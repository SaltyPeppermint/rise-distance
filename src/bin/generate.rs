use std::fmt::Display;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use clap::Parser;
use csv::Writer;
use egg::{Analysis, Language, Rewrite, Runner, SimpleScheduler, StopReason};
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
"
)]
struct Cli {
    /// Total number of samples
    #[arg(long)]
    total_samples: usize,

    /// Min term size
    #[arg(long)]
    min_size: usize,

    /// Max term size
    #[arg(long)]
    max_size: usize,

    /// Tolerance in the boltzman sampler
    #[arg(long, default_value_t = 1)]
    tolerance: usize,

    /// Seed
    #[arg(long)]
    seed: u64,

    /// Retry limit when rejecting previously seen terms
    /// Should only be an issue for really small terms
    #[arg(long, default_value_t = 10000)]
    retry_limit: usize,

    /// Size Distribution to sample
    #[arg(long)]
    distribution: Distribution,

    /// Size Distribution to sample
    #[arg(long)]
    path: PathBuf,

    /// Minimum iter complexity for the term
    #[arg(long, default_value_t = 11)]
    max_iters: usize,

    /// Minimum nodes complexity for the term
    #[arg(long, default_value_t = 100_000)]
    max_nodes: usize,

    /// Minimum time complexity for the term
    #[arg(long, default_value_t = 1.0)]
    max_time: f64,

    #[arg(long)]
    parallelism: Option<usize>,
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
                            .sample(&mut rng, &|t| {
                                valididty_hook(
                                    t,
                                    cli.max_iters,
                                    cli.max_nodes,
                                    cli.max_time,
                                    &RULES,
                                )
                            })
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

pub fn valididty_hook<L: Label + Language + Display, N: Analysis<L> + Default, T: ToEgg<L>>(
    tree: &T,
    max_iters: usize,
    max_nodes: usize,
    max_time: f64,
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
        .with_iter_limit(max_iters)
        .with_node_limit(max_nodes)
        .with_time_limit(Duration::from_secs_f64(max_time))
        .with_scheduler(SimpleScheduler);

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

// /// Structural estimate of the heap bytes held by `g`.
// ///
// /// Walks only the public API ([`EGraph::nodes`], [`EGraph::classes`],
// /// [`EGraph::total_size`], [`EGraph::total_number_of_nodes`],
// /// [`EGraph::number_of_classes`]) and assumes:
// /// - `L` and `N::Data` are `'static` and own no heap allocations themselves
// ///   (true for our `Math`/`Lambda` languages).
// /// - Explanations are disabled, so `explain: Option<Explain<L>>` contributes
// ///   only its discriminant.
// /// - The egraph has just been rebuilt, so `pending` and `analysis_pending`
// ///   are effectively empty.
// ///
// /// Approximations:
// /// - `memo` and `classes` are sized assuming hashbrown's load factor (≤ 7/8)
// ///   and a power-of-two bucket count, plus one metadata byte per slot.
// /// - `unionfind` is treated as a `Vec<Id>` with one slot per id ever added
// ///   (i.e. `nodes().len()`); the real `UnionFind` has identical asymptotic
// ///   storage but may carry a small constant of bookkeeping.
// /// - `classes_by_op` is sized by walking `g.nodes()` to collect the distinct
// ///   discriminants actually present, then querying [`EGraph::classes_for_op`]
// ///   for each to learn the per-op `HashSet<Id>` length. Each set is sized
// ///   individually with the same hashbrown formula. This is exact in the
// ///   payload (one `Id` per canonical enode) and tight on the per-op bucket
// ///   overhead, but still ignores any unused-but-allocated capacity in those
// ///   sets.
// ///
// /// Things deliberately not counted:
// /// - `explain` contents (assumed disabled).
// /// - Transient queues (`pending`, `analysis_pending`).
// /// - Allocator slack and per-allocation headers.
// /// - Any heap data hanging off `L` or `N::Data` (caller's responsibility).
// pub fn estimate_egraph_bytes<L, N>(g: &EGraph<L, N>) -> usize
// where
//     L: Language,
//     N: Analysis<L>,
// {
//     let n_nodes_total = g.nodes().len();

//     let mut bytes = mem::size_of::<EGraph<L, N>>();

//     // `nodes: Vec<L>` with one slot per id ever added (non-canonical included).
//     bytes += mem::size_of_val(g.nodes());

//     // `unionfind`: one parent Id per id ever added.
//     bytes += n_nodes_total * mem::size_of::<Id>();

//     // `memo: HashMap<L, Id>`
//     bytes += hashbrown_bytes::<L, Id>(g.total_size());

//     // Per-class storage.
//     for class in g.classes() {
//         bytes += mem::size_of::<EClass<L, N::Data>>();
//         bytes += class.nodes.len() * mem::size_of::<L>();
//         bytes += class.parents().len() * mem::size_of::<Id>();
//     }

//     // `classes_by_op`: discriminants present, then per-op HashSet<Id> sizes.
//     let mut discriminants = HashSet::new();
//     for node in g.nodes() {
//         discriminants.insert(node.discriminant());
//     }
//     bytes += hashbrown_bytes::<L::Discriminant, ()>(discriminants.len());
//     for disc in &discriminants {
//         let len = g.classes_for_op(disc).map_or(0, |it| it.len());
//         bytes += hashbrown_bytes::<Id, ()>(len);
//     }

//     // `classes: HashMap<Id, EClass<..>>` shell (EClass bodies counted above).
//     bytes += hashbrown_bytes::<Id, ()>(g.number_of_classes());

//     bytes
// }

// fn hashbrown_bytes<K, V>(len: usize) -> usize {
//     if len == 0 {
//         return 0;
//     }
//     let cap = (len * 8).div_ceil(7).next_power_of_two();
//     cap * (mem::size_of::<(K, V)>() + 1)
// }

// const INTERCEPT: f64 = 1.524_602_184_0e+06;
// const COEFS: [f64; 5] = [
//     2.846_220_556_5e+01,  // stop_nodes
//     -4.064_005_731_4e+02, // stop_classes
//     -1.503_030_448_2e+02, // last_nodes
//     3.727_468_060_5e+02,  // last_classes
//     2.433_465_461_1e+00,  // estimated_mem_egraph
// ];

// /// Predicted `measured_mem_total` in bytes. Coefficients fit in
// /// `analysis/size_estimate.ipynb` (linear regression, R^2 ~ 0.9875 on a held-out
// /// 20% split of `output_new_100.csv`).
// fn predict_mem(
//     stop_nodes: usize,
//     stop_classes: usize,
//     last_nodes: usize,
//     last_classes: usize,
//     egraph_bytes: usize,
// ) -> f64 {
//     #[expect(clippy::cast_precision_loss)]
//     let xs = [
//         stop_nodes as f64,
//         stop_classes as f64,
//         last_nodes as f64,
//         last_classes as f64,
//         egraph_bytes as f64,
//     ];

//     let mut acc = INTERCEPT;
//     for k in 0..COEFS.len() {
//         acc += COEFS[k] * xs[k];
//     }
//     acc
// }
