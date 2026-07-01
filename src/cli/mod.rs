pub mod sample;
pub mod types;

pub use types::*;

use std::env::current_dir;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::eqsat::EqsatConfig;
use crate::langs::AvailableLanguages;

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

/// Read `<folder>/args.json` into an `EqsatConfig`.
///
/// # Panics
///
/// Panics if `args.json` is missing, unreadable, or the required fields are missing/wrong-typed.
#[must_use]
pub fn read_folder_args(folder: &Path) -> EqsatConfig {
    let path = folder.join("args.json");
    let reader =
        File::open(&path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    serde_json::from_reader(reader)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()))
}

/// Read the `language` field from `<folder>/args.json`.
///
/// `args.json` is the full arg object written by `generate_and_measure.py`, so
/// pull the `language` field out of it rather than deserializing the whole file
/// as a bare enum.
///
/// # Panics
///
/// Panics if `args.json` is missing, unreadable, malformed, or lacks a
/// `language` field (older runs predating the field must be regenerated).
#[must_use]
pub fn read_folder_language(folder: &Path) -> AvailableLanguages {
    #[derive(serde::Deserialize)]
    struct LanguageField {
        language: AvailableLanguages,
    }

    let path = folder.join("args.json");
    let reader =
        File::open(&path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let parsed: LanguageField = serde_json::from_reader(reader)
        .unwrap_or_else(|e| panic!("Failed to parse `language` from {}: {e}", path.display()));
    parsed.language
}
