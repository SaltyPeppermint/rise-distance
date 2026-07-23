//! Shared wire types for the guide experiment's `goal`, `sample`, and `verify`
//! binaries.

use std::collections::BTreeMap;

use egg::RecExpr;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::Counter;
use crate::eqsat::EqsatMetadata;
use crate::sampling::SampleStrategy;
use crate::{MyLanguage, OriginLang};

/// The five guide-sampling strategies. Sampling variants always draw frontier
/// terms; only `Smallest` exposes the novel/overall choice. Mirrors the enum
/// that used to live in `guide.rs`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Strategy {
    Sample(SampleStrategy),
    Smallest { novel: bool },
}

impl Strategy {
    /// Every guide strategy emitted by the `sample` binary.
    pub const ALL: [Strategy; 5] = [
        Strategy::Sample(SampleStrategy::Independent),
        Strategy::Sample(SampleStrategy::Naive),
        Strategy::Sample(SampleStrategy::Balanced),
        Strategy::Smallest { novel: false },
        Strategy::Smallest { novel: true },
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Strategy::Sample(SampleStrategy::Independent) => "sample_independent",
            Strategy::Sample(SampleStrategy::Naive) => "sample_naive",
            Strategy::Sample(SampleStrategy::Balanced) => "sample_balanced",
            Strategy::Smallest { novel: true } => "smallest_novel",
            Strategy::Smallest { novel: false } => "smallest_overall",
        }
    }

    /// Deterministic per-strategy RNG salt so sampling variants don't share a
    /// seed within a seed record.
    #[must_use]
    pub const fn seed_of(&self) -> u64 {
        match self {
            Strategy::Sample(SampleStrategy::Independent) => 1,
            Strategy::Sample(SampleStrategy::Naive) => 2,
            Strategy::Sample(SampleStrategy::Balanced) => 3,
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

/// A per-seed sampling record `sample` prints to stdout (which `guided_search.py`
/// collects into `samples.json`). Carries, per strategy, the guide candidates
/// Python may restart with, plus replay metadata for Python's logging. The
/// goals and `max_size` are not here: the driver keeps them Python-side (from
/// `goal_terms.json`) and re-associates by seed.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "L: MyLanguage")]
pub struct SeedSamples<L: MyLanguage> {
    pub seed: String,
    /// Guide candidates keyed by [`Strategy::name`]. Sampling strategies hold up
    /// to `samples_per_strategy` terms; `Smallest` holds exactly one.
    pub candidates: BTreeMap<String, Vec<GuideExpr<L>>>,
    pub guide_nodes: usize,
    pub guide_classes: usize,
    pub guide_iters: usize,
    /// Total wall-clock time (seconds) of the guide-phase replay, so the driver
    /// can add the guide overhead to each leg's `total_time`.
    pub guide_time: f64,
    /// Guide-phase replay's live-heap growth (bytes): jemalloc `stats.allocated`
    /// after the replay minus a sample before it, isolating the replay's
    /// footprint. (The `--stop-memory` budget is enforced against absolute
    /// live-heap in the eqsat's memory hook, not this delta.)
    pub guide_memory: u64,
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
    pub guide_egraph: EqsatMetadata,
    pub goal_egraph: EqsatMetadata,
    /// Eqsat's live-heap growth (bytes): jemalloc `stats.allocated` after the
    /// eqsat minus a sample before it, isolating the eqsat's footprint. A single
    /// combined reading, not splittable into `guide_egraph`/`goal_egraph` halves.
    pub base_memory: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guide_strategy_menu_includes_balanced_frontier_sampling() {
        let names = Strategy::ALL.map(Strategy::name);
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
        assert_eq!(
            Strategy::Sample(SampleStrategy::Balanced).seed_of(),
            3
        );
    }
}
