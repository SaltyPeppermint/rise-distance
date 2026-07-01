use hashbrown::HashMap;
use num::BigUint;
use serde::{Deserialize, Serialize};

use crate::eqsat::{EqsatConfig, EqsatMetadata};

/// Per-seed payload written by `goal` and consumed by `sample`. Stored as the
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
pub struct EnrichedSeedFailed {
    pub max_size: usize,
    pub fail_reason: String,
}
