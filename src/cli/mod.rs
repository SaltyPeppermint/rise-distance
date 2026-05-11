pub mod argparse;
pub mod logging;
pub mod parquet;
pub mod types;

pub use logging::{_tee_print, init_log};
pub use types::*;

use std::env::current_dir;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use egg::{EGraph, Id, Iteration, RecExpr};
use num::ToPrimitive;
use serde::Serialize;

use crate::cli::argparse::{SampleStrategy, TermSampleDist};
use crate::count::{Counter, PlainTermCount};
use crate::sampling::{CountWeigher, NaiveWeigher, Sampler};
use crate::{
    MyAnalysis, MyLanguage, NovelSampler, NovelTermCount, OriginLang, PlainSampler, tee_println,
};

pub fn trial_avg<
    F: Fn(&Vec<Iteration<()>>) -> Option<T>,
    T: for<'a> std::iter::Sum<&'a T> + ToPrimitive,
>(
    trials: &[Option<Vec<Iteration<()>>>],
    f: F,
) -> Option<f64> {
    let values = trials
        .iter()
        .filter_map(|x| x.as_ref().and_then(&f))
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let avg = values.iter().sum::<T>().to_f64()? / values.len().to_f64()?;
    Some(avg)
}

#[expect(clippy::missing_panics_doc)]
pub fn min_med_max<T: Ord + Copy, I, F: Fn(&I) -> T>(items: &[I], f: F) -> (T, T, T) {
    let min = items.iter().map(&f).min().unwrap();
    let max = items.iter().map(&f).max().unwrap();
    let med = f(&items[items.len() / 2]);
    (min, med, max)
}

pub struct PrecomputePackage<'a, C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    tc: NovelTermCount<'a, C, L, N>,
    min_size: usize,
    max_size: usize,
    root: Id,
}

impl<'a, C, L, N> PrecomputePackage<'a, C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    /// Enumerate all frontier terms from `egraph` that are NOT present in `prev_raw_egg` for the sampling process later
    pub fn precompute(
        curr: &'a EGraph<L, N>,
        prev: &'a EGraph<L, N>,
        root: Id,
        max_size: usize,
    ) -> Option<PrecomputePackage<'a, C, L, N>> {
        let tc = NovelTermCount::new(
            max_size,
            curr,
            prev,
            PlainTermCount::<C>::new(max_size, curr),
        );

        // `data()` is keyed by canonical curr ids; the caller's `root` may not
        // be canonical (see `GuideGoalResult::root`).
        let root = curr.find(root);
        let histogram = tc.data().get(&root)?;

        let min_size = histogram.keys().min().copied().unwrap_or(1);
        Some(PrecomputePackage {
            tc,
            min_size,
            max_size,
            root,
        })
    }

    /// Log the stats about the root
    ///
    /// # Panics
    ///
    /// Panics if there are no terms in the root
    pub fn log_root(&self) {
        let histogram = self
            .tc
            .data()
            .get(&self.root)
            .expect("Somehow the root does not contain any terms?");
        let mut sorted_hist = histogram
            .iter()
            .map(|(a, b)| (*a, b.to_owned()))
            .collect::<Vec<_>>();
        sorted_hist.sort_unstable();
        tee_println!("Terms in frontier:");
        for (k, v) in &sorted_hist {
            tee_println!("{v} terms of size {k}");
        }
    }

    /// Sample frontier goal terms from `egraph` that are NOT present in `prev_raw_egg`.
    #[expect(clippy::missing_errors_doc)]
    pub fn sample_frontier_terms<const PARALLEL: bool>(
        &self,
        count: usize,
        distribution: TermSampleDist,
        sample_strategy: SampleStrategy,
        seed: [u64; 2],
        novel: bool,
    ) -> Result<Vec<RecExpr<OriginLang<L>>>, ExperimentError> {
        let histogram = self
            .tc
            .data()
            .get(&self.root)
            .ok_or(ExperimentError::InsufficientSamples)?;

        let samples_per_size =
            distribution.samples_per_size(histogram, self.min_size, self.max_size, count);
        let samples = match (sample_strategy, novel) {
            (SampleStrategy::Naive, true) => {
                NovelSampler::new(&self.tc, self.root, NaiveWeigher)
                    .sample_batch_root::<PARALLEL>(&samples_per_size, seed)
            }
            (SampleStrategy::CountBased, true) => {
                NovelSampler::new(&self.tc, self.root, CountWeigher)
                    .sample_batch_root::<PARALLEL>(&samples_per_size, seed)
            }
            (SampleStrategy::Naive, false) => {
                PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, NaiveWeigher)
                    .sample_batch_root::<PARALLEL>(&samples_per_size, seed)
            }
            (SampleStrategy::CountBased, false) => {
                PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, CountWeigher)
                    .sample_batch_root::<PARALLEL>(&samples_per_size, seed)
            }
        };
        if samples.len() == count {
            return Ok(samples.into_iter().collect());
        }
        Err(ExperimentError::InsufficientSamples)
    }

    #[must_use]
    pub fn smallest(&self, id: Id, novel: bool) -> RecExpr<OriginLang<L>> {
        if novel {
            NovelSampler::new(&self.tc, self.root, NaiveWeigher).smallest(id)
        } else {
            PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, NaiveWeigher).smallest(id)
        }
    }

    #[must_use]
    pub fn smallest_n(&self, id: Id, novel: bool, n: u64) -> Vec<RecExpr<OriginLang<L>>> {
        if novel {
            NovelSampler::new(&self.tc, self.root, NaiveWeigher).n_smallest(id, n)
        } else {
            PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, NaiveWeigher)
                .n_smallest(id, n)
        }
    }

    #[must_use]
    pub fn root(&self) -> Id {
        self.root
    }
}

/// Create an output folder for a run.
///
/// If `output` is `Some`, uses that path directly. Otherwise, auto-generates a
/// path like `data/<subdir>/run-<prefix>-sampling.<N>` where `<N>` is one higher
/// than the largest existing run number.
#[expect(clippy::missing_panics_doc)]
pub fn get_run_folder(output: Option<&str>, subdir: &str, prefix: &str) -> PathBuf {
    let this_run_dir = output.map_or_else(
        || {
            let runs_dir = current_dir().unwrap().join("data").join(subdir);
            std::fs::create_dir_all(&runs_dir).expect("Failed to create output directory");
            let max_existing = runs_dir
                .read_dir()
                .unwrap()
                .filter_map(|e| {
                    let d = e.ok()?;
                    if d.file_type().ok()?.is_dir() && prefix == d.path().file_stem()? {
                        return d.path().extension()?.to_str()?.parse::<usize>().ok();
                    }
                    None
                })
                .max()
                .unwrap_or(0);
            runs_dir
                .join(prefix)
                .with_extension((max_existing + 1).to_string())
        },
        PathBuf::from,
    );
    std::fs::create_dir_all(&this_run_dir).expect("Failed to create output directory");
    this_run_dir
}

/// Write the CLI configuration to `config.json` in the run folder.
///
/// # Panics
///
/// Panics if the file cannot be created or serialization fails.
#[expect(clippy::impl_trait_in_params)]
pub fn write_config(run_folder: &Path, cli: &impl Serialize) {
    let config_path = run_folder.join("config.json");
    let config_file = File::create(config_path).expect("Failed to create output config.json file");
    let config_writer = BufWriter::new(config_file);
    serde_json::to_writer_pretty(config_writer, cli).unwrap();
}

/// Write per-seed stats to `stats.json` in the run folder.
///
/// # Panics
///
/// Panics if the file cannot be created or serialization fails.
pub fn write_stats(run_folder: &Path, stats: &[serde_json::Value]) {
    let stats_path = run_folder.join("stats.json");
    let stats_file = File::create(&stats_path).expect("Failed to create stats.json");
    let stats_writer = BufWriter::new(stats_file);
    serde_json::to_writer_pretty(stats_writer, stats).expect("write stats json");
}
