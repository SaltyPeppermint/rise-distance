use std::path::PathBuf;

use clap::Parser;
use csv::Writer;
use hashbrown::{HashMap, HashSet};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rise_distance::egg::math::generate::BoltzmannSampler;
use serde::Serialize;

use rise_distance::cli::argtypes::Distribution;

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

    #[arg(long, default_value_t = 10000)]
    retry_limit: usize,

    /// Size Distribution to sample
    #[arg(long)]
    distribution: Distribution,

    /// Size Distribution to sample
    #[arg(long)]
    path: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    let samples_per_size =
        cli.distribution
            .samples_per_size(cli.min_size, cli.max_size, cli.total_samples);

    // Derive one RNG per size by advancing the root RNG sequentially, making it deterministic and ordered.
    let mut root_rng = ChaCha12Rng::seed_from_u64(cli.seed);
    let mut sized_rngs = samples_per_size
        .iter()
        .map(|(&size, &n)| {
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

    let big_collector = sized_rngs
        .into_par_iter()
        .progress_with_style(style)
        .map(|(size, n, mut rng)| {
            let sampler = BoltzmannSampler::new(size, cli.tolerance, None);
            let mut collector = HashSet::new();
            while (collector.len() as u64) < n {
                let mut consecutive_dupes = 0;
                loop {
                    assert!(
                        consecutive_dupes < cli.retry_limit,
                        "Sampled previously seen term too often"
                    );
                    let new_term = sampler
                        .sample(&mut rng)
                        .expect("Too many failed sample attempts");
                    if collector.insert(new_term) {
                        break;
                    }
                    consecutive_dupes += 1;
                }
            }
            (size, collector)
        })
        .collect::<HashMap<_, _>>();

    let mut writer = Writer::from_path(&cli.path).expect("File does not exist");
    writer.write_record(["size", "term"]).unwrap();

    for (size, c) in big_collector {
        let size_str = size.to_string();
        for term in c {
            writer.write_record([&size_str, &term.to_string()]).unwrap();
        }
    }
}
