//! Shared wire types for the `goal` / `sample` / `verify` split of the guide
//! experiment. The three binaries touch no files: the Python drivers
//! (`generate_goals.py` / `guided_search.py`) own all I/O, passing eqsat limits and
//! language on argv (via [`crate::eqsat::EqsatConfig`], the shared clap flag
//! group) and per-seed/per-leg data as the JSON records defined here. `goal`
//! records [`GoalGenMetadata`] into `goal_terms.json`; `sample` emits [`SeedSamples`]
//! (guide menus of [`GuideExpr`], one pool per [`Strategy`]); `guided_search.py` feeds
//! chosen subsets back to `verify`.

use std::collections::BTreeMap;

use egg::RecExpr;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::Counter;
use crate::eqsat::EqsatMetadata;
use crate::sampling::SampleStrategy;
use crate::{MyLanguage, OriginLang};

/// The four guide-sampling strategies. Sampling variants always draw novel
/// terms; only `Smallest` exposes the novel/overall choice. Mirrors the enum
/// that used to live in `guide.rs`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Strategy {
    Sample(SampleStrategy),
    Smallest { novel: bool },
}

impl Strategy {
    /// All four strategies, in the order `guide.rs` used.
    pub const ALL: [Strategy; 4] = [
        Strategy::Sample(SampleStrategy::Count),
        Strategy::Sample(SampleStrategy::Naive),
        Strategy::Smallest { novel: false },
        Strategy::Smallest { novel: true },
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
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
    pub const fn seed_of(&self) -> u64 {
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
    /// Peak process RSS in bytes (`VmHWM`) sampled right after the replay.
    /// The replay is the first heavy phase of `sample`, so this is its peak —
    /// the number a `--stop-memory` budget is measured against.
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
    /// Peak process RSS (bytes) of the combined run. Unlike the node/class/time
    /// metadata, RSS is a process-wide high-water mark and can't be split into
    /// `guide_egraph`/`goal_egraph` halves, so it's the whole-run peak.
    pub base_memory: u64,
}
