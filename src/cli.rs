//! Shared wire types for the guide experiment's `goal`, `sample`, and `verify`
//! binaries.

use std::collections::BTreeMap;

use clap::ValueEnum;
use egg::RecExpr;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::Counter;
use crate::eqsat::EqsatMetadata;
use crate::sampling::SampleStrategy;
use crate::{MyLanguage, OriginLang};

/// One candidate pool emitted by `sample`.
///
/// The CLI and JSON names are identical so the Rust sampler and Python driver
/// share one vocabulary.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CandidatePool {
    #[value(name = "sample_independent")]
    SampleIndependent,
    #[value(name = "sample_naive")]
    SampleNaive,
    #[value(name = "sample_balanced")]
    SampleBalanced,
    #[value(name = "smallest_overall")]
    SmallestOverall,
    #[value(name = "smallest_novel")]
    SmallestNovel,
}

impl CandidatePool {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SampleIndependent => "sample_independent",
            Self::SampleNaive => "sample_naive",
            Self::SampleBalanced => "sample_balanced",
            Self::SmallestOverall => "smallest_overall",
            Self::SmallestNovel => "smallest_novel",
        }
    }

    /// Deterministic per-strategy RNG salt so sampling variants don't share a
    /// seed within a seed record.
    #[must_use]
    pub const fn seed_of(&self) -> u64 {
        match self {
            Self::SampleIndependent => 1,
            Self::SampleNaive => 2,
            Self::SampleBalanced => 3,
            Self::SmallestOverall | Self::SmallestNovel => 0,
        }
    }

    #[must_use]
    pub const fn sample_strategy(self) -> Option<SampleStrategy> {
        match self {
            Self::SampleIndependent => Some(SampleStrategy::Independent),
            Self::SampleNaive => Some(SampleStrategy::Naive),
            Self::SampleBalanced => Some(SampleStrategy::Balanced),
            Self::SmallestOverall | Self::SmallestNovel => None,
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
    /// Consume a sampled expression and reuse its node allocation for the wire
    /// representation.
    #[must_use]
    pub fn from_recexpr(expr: RecExpr<OriginLang<L>>) -> Self {
        Self { nodes: expr.into() }
    }

    #[must_use]
    pub fn into_recexpr(self) -> RecExpr<OriginLang<L>> {
        RecExpr::from(self.nodes)
    }
}

/// A per-seed sampling record `sample` prints to stdout (which `guided_search.py`
/// collects into `samples.json`). Carries, per strategy, the guide candidates
/// Python may restart with, plus replay metadata for Python's logging. The
/// goals and `max_size` are not here: the driver keeps them Python-side (from
/// `goal_terms.json`) and re-associates by seed.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "L: MyLanguage")]
pub struct SeedSamples<L: MyLanguage> {
    pub seed: String,
    /// Requested guide candidates keyed by [`CandidatePool::name`]. Sampling
    /// strategies hold up to `samples_per_strategy` terms; `Smallest` holds
    /// exactly one.
    pub candidates: BTreeMap<String, Vec<GuideExpr<L>>>,
    pub guide_nodes: usize,
    pub guide_classes: usize,
    pub guide_iters: usize,
    /// Total wall-clock time (seconds) of the guide-phase replay, so the driver
    /// can add the guide overhead to each leg's `total_time`.
    pub guide_time: f64,
    /// Guide-phase replay's absolute live allocation (bytes): jemalloc
    /// `stats.allocated` for the whole process, the same coordinate system the
    /// configured memory ceiling is expressed in. Includes heap the process
    /// already held before this run started.
    pub guide_memory: u64,
    /// Largest sampled absolute live heap during guide replay.
    pub guide_peak_live_heap: u64,
    pub stop_reason: String,
}

/// Per-seed payload written by `goal` into the value slot of `goal_terms.json` (one
/// entry per seed s-expression). Serializes via `Result`'s `{"Ok": ..}` /
/// `{"Err": ..}` shape (`goal` returns a `Result<GoalGenMetadata, String>`).
/// `guided_search.py` parses the enriched `goal_terms.json` and pulls each `Ok` seed's
/// goals from it (the replay budget comes from its own `--stop-*` flags).
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound(serialize = "C: Counter", deserialize = "C: Counter"))]
pub struct GoalGenMetadata<C: Counter> {
    pub max_size: usize,
    pub goals: Vec<String>,
    /// Histogram of novel root extractions by size. Keys are size-as-string
    /// because JSON object keys must be strings and `serde_json` doesn't
    /// auto-convert numeric strings back to `usize` on read.
    pub frontier_histogram: HashMap<String, C>,
    pub stop_reason: String,
    pub goal_egraph: EqsatMetadata,
    /// Eqsat's absolute live allocation (bytes): jemalloc `stats.allocated`
    /// for the whole process, the same coordinate system the configured memory
    /// ceiling is expressed in.
    pub base_memory: u64,
    /// Largest sampled absolute live heap during the unguided goal run.
    pub base_peak_live_heap: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_pool_names_include_balanced_frontier_sampling() {
        let names = [
            CandidatePool::SampleIndependent,
            CandidatePool::SampleNaive,
            CandidatePool::SampleBalanced,
            CandidatePool::SmallestOverall,
            CandidatePool::SmallestNovel,
        ]
        .map(CandidatePool::name);
        assert_eq!(
            names,
            [
                "sample_independent",
                "sample_naive",
                "sample_balanced",
                "smallest_overall",
                "smallest_novel",
            ]
        );
        assert_eq!(CandidatePool::SampleBalanced.seed_of(), 3);
        assert_eq!(
            serde_json::to_string(&CandidatePool::SampleBalanced).unwrap(),
            "\"sample_balanced\""
        );
    }
}
