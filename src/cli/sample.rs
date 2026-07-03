//! Shared types and helpers for the `sample` / `verify` split of the guide
//! experiment. `sample` replays the guide-phase eqsat, samples the six-strategy
//! guide-candidate menu, and writes it as `samples.json`; `driver.py` then feeds
//! chosen subsets back to `verify` one leg at a time.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use egg::RecExpr;
use serde::{Deserialize, Serialize};

use crate::cli::types::EnrichedSeed;
use crate::sampling::SampleStrategy;
use crate::{MyLanguage, OriginLang};

/// The six guide-sampling strategies. Sampling variants always draw novel
/// terms; only `Smallest` exposes the novel/overall choice. Mirrors the enum
/// that used to live in `guide.rs`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Strategy {
    Sample(SampleStrategy),
    Smallest { novel: bool },
}

impl Strategy {
    /// All six strategies, in the order `guide.rs` used.
    pub const ALL: [Strategy; 4] = [
        Strategy::Sample(SampleStrategy::Count),
        Strategy::Sample(SampleStrategy::Naive),
        Strategy::Smallest { novel: false },
        Strategy::Smallest { novel: true },
    ];

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Strategy::Sample(SampleStrategy::Count) => "sample_count",
            Strategy::Sample(SampleStrategy::Naive) => "sample_naive",
            Strategy::Smallest { novel: true } => "smallest_novel",
            Strategy::Smallest { novel: false } => "smallest_overall",
        }
    }

    /// Deterministic per-strategy RNG salt so the two `SampleStrategy` variants
    /// don't share a seed within a seed record.
    #[must_use]
    pub fn seed_of(&self) -> u64 {
        match self {
            Strategy::Sample(SampleStrategy::Count) => 1,
            Strategy::Sample(SampleStrategy::Naive) => 2,
            Strategy::Smallest { .. } => 0,
        }
    }
}

/// One guide candidate on the wire. Stored as its node list rather than an
/// s-expression string so the per-node `origin` id survives the
/// Rust -> Python -> Rust round trip.
/// egg's `RecExpr` serde goes through `Display`,
/// which drops the origin and would break `--full-union`.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "L: MyLanguage")]
pub struct GuideExpr<L: MyLanguage> {
    pub nodes: Vec<OriginLang<L>>,
}

impl<L: MyLanguage> GuideExpr<L> {
    #[must_use]
    pub fn from_recexpr(expr: &RecExpr<OriginLang<L>>) -> Self {
        Self {
            nodes: expr.as_ref().to_vec(),
        }
    }

    #[must_use]
    pub fn into_recexpr(self) -> RecExpr<OriginLang<L>> {
        RecExpr::from(self.nodes)
    }
}

/// A per-seed sampling record written to `samples.json`. Carries the goal menu
/// (lowered s-expression strings, as stored by `goal`) and, per strategy, the
/// guide candidates Python may restart with, plus enough replay metadata for
/// Python's logging.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "L: MyLanguage")]
pub struct SeedSamples<L: MyLanguage> {
    pub seed: String,
    /// Lowered goal s-expressions (parse straight into `RecExpr<L>` in `verify`).
    pub goals: Vec<String>,
    /// Guide candidates keyed by [`Strategy::name`]. Sampling strategies hold up
    /// to `samples_per_strategy` terms; `Smallest` holds exactly one.
    pub candidates: BTreeMap<String, Vec<GuideExpr<L>>>,
    pub max_size: usize,
    pub guide_nodes: usize,
    pub guide_classes: usize,
    pub guide_iters: usize,
    /// Total wall-clock time (seconds) of the guide-phase replay, so the driver
    /// can add the guide overhead to each leg's `total_time`.
    pub guide_time: f64,
    pub stop_reason: String,
}

/// Read enriched `terms.json`. Returns a flat list in deterministic order
/// (groups in JSON order, terms within each group sorted alphabetically) so
/// `--take-first` is stable across runs. Lifted verbatim from `guide.rs`.
///
/// # Panics
///
/// Panics if `terms.json` is missing or was not enriched by `goal` first.
#[must_use]
pub fn read_enriched_terms(folder: &Path) -> Vec<(String, EnrichedSeed)> {
    let path = folder.join("terms.json");
    let file =
        File::open(&path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));

    // Read directly into a typed schema. Going via `serde_json::Value` first
    // would force HashMap<usize, _> keys through string → usize conversion
    // that `from_value` doesn't do. The inner map is a BTreeMap so its
    // iteration order is deterministic; a HashMap here would make
    // `--take-first` pick a different subset each run.
    let groups: Vec<(usize, BTreeMap<String, EnrichedSeed>)> = serde_json::from_reader(file)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to parse {}: {e}. Did you run `goal` on this folder first?",
                path.display()
            )
        });

    groups
        .into_iter()
        .flat_map(|(_size, inner)| inner)
        .collect()
}
