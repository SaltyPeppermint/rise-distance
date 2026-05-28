pub mod argparse;
pub mod parquet;
pub mod types;

pub use types::*;

use std::env::current_dir;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use egg::Iteration;
use num::ToPrimitive;
use serde::Serialize;

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

/// Write per-seed metadata to `metadata.json` in the run folder.
///
/// # Panics
///
/// Panics if the file cannot be created or serialization fails.
pub fn write_metadata(run_folder: &Path, metadata: &[serde_json::Value]) {
    let path = run_folder.join("metadata.json");
    let file = File::create(&path).expect("Failed to create metadata.json");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, metadata).expect("write metadata json");
}
