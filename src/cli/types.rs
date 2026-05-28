use egg::{Iteration, RecExpr, StopReason};
use hashbrown::HashMap;
use num::BigUint;
use serde::{Deserialize, Serialize};
use strum::Display;
use thiserror::Error;

use crate::cli::argparse::EqsatConfig;
use crate::{MyLanguage, OriginLang};

#[derive(Debug, Error, Display, Serialize, Clone)]
pub enum ExperimentError {
    Guide(#[from] GuideError),
    InsufficientSamples,
    NothingInHistogram,
}

#[derive(Debug, Error, Display, Serialize, Clone)]
pub enum GuideError {
    Unreached(StopReason),
    PanicWhileAttempt,
}

#[derive(Serialize, Debug, Clone)]
pub struct GuideEval<L: MyLanguage> {
    pub guide: RecExpr<OriginLang<L>>,
    pub zs_distance: usize,
    pub iterations: Result<Vec<Iteration<()>>, GuideError>,
}

/// Type alias for the per-k trial data: maps each guide-set size `k` to its
/// trial results (one `Option` per trial. `None` means the goal was not reached).
pub type TrialsPerK = HashMap<usize, Vec<Result<Vec<Iteration<()>>, ExperimentError>>>;

/// Same as `TrialsPerK` but with pre-computed summaries instead of full iteration data.
pub type SummaryPerK = HashMap<usize, Vec<Result<TrialSummary, ExperimentError>>>;

/// Pre-computed per-trial summary. Much smaller than the full `Iteration`
/// vectors, so Python can load it instantly.
#[derive(Serialize)]
pub struct TrialSummary {
    /// Number of eqsat iterations to reach the goal.
    pub iters: usize,
    /// Egraph node count at the final iteration.
    pub nodes: usize,
    /// Egraph e-class count at the final iteration.
    pub classes: usize,
    /// Total number of rule applications across all iterations.
    pub total_applied: usize,
    /// Total wall-clock time (seconds) across all iterations.
    pub total_time: f64,
}

/// Per-seed payload written by `goal` and consumed by `guide`. Stored as the
/// value slot in `terms.json` (one entry per seed s-expression). Tagged on
/// `status` so the two variants share a flat JSON shape.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EnrichedSeed {
    Ok(GoalGenMetadata),
    Failed(EnrichedSeedFailed),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GoalGenMetadata {
    /// Snapshot of the `EqsatConfig` that `goal` ran under. `guide` compares
    /// this against its current `args.json` to detect config drift.
    pub eqsat_config: EqsatConfig,
    pub max_size: usize,
    pub goals: Vec<String>,
    /// Histogram of novel root extractions by size. Keys are size-as-string
    /// because JSON object keys must be strings and `serde_json` doesn't
    /// auto-convert numeric strings back to `usize` on read.
    pub frontier_histogram: HashMap<String, BigUint>,
    pub stop_reason: String,
    pub guide_egraph: EqsatMetadata,
    pub goal_egraph: EqsatMetadata,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EqsatMetadata {
    pub nodes: usize,
    pub classes: usize,
    pub time: f64,
    pub iters: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EnrichedSeedFailed {
    pub max_size: usize,
    pub fail_reason: String,
}

#[derive(Serialize)]
pub struct GoalSummary {
    pub seed: String,
    pub goal: String,
    pub entries_per_k: SummaryPerK,
}

impl GoalSummary {
    /// Build a summary from the full per-k trial data for a given goal.
    ///
    /// # Panics
    /// Panics if a reachable trial has an empty iteration list.
    #[must_use]
    pub fn from_entries(seed: &str, goal: &str, entries: &TrialsPerK) -> Self {
        Self {
            seed: seed.to_owned(),
            goal: goal.to_owned(),
            entries_per_k: entries
                .iter()
                .map(|(k, trials)| {
                    (
                        *k,
                        trials
                            .iter()
                            .map(|trial| {
                                trial
                                    .as_ref()
                                    .map(|iters| {
                                        let last = iters.last().expect("non-empty iteration list");
                                        TrialSummary {
                                            iters: iters.len(),
                                            nodes: last.egraph_nodes,
                                            classes: last.egraph_classes,
                                            total_applied: iters
                                                .iter()
                                                .map(|i| i.applied.values().sum::<usize>())
                                                .sum(),
                                            total_time: iters.iter().map(|i| i.total_time).sum(),
                                        }
                                    })
                                    .map_err(|e| e.clone())
                            })
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}
