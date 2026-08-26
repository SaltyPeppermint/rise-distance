//! Shared wire types for the guide experiment's `goal`, `candidates`, and `verify`
//! binaries.

use clap::ValueEnum;
use egg::RecExpr;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::Counter;
use crate::eqsat::EqsatMetadata;
use crate::{MyLanguage, OriginLang};

/// One guide-candidate construction pool emitted by `candidates`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    #[value(name = "count")]
    Count,
    #[value(name = "naive")]
    Naive,
    #[value(name = "smallest_overall")]
    SmallestOverall,
    #[value(name = "smallest_novel")]
    SmallestNovel,
}

impl std::fmt::Display for Policy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count => write!(f, "count"),
            Self::Naive => write!(f, "naive"),
            Self::SmallestOverall => write!(f, "smallest_overall"),
            Self::SmallestNovel => write!(f, "smallest_novel"),
        }
    }
}

impl Policy {
    /// Deterministic per-pool RNG salt so construction policies do not share a
    /// random stream within one seed record.
    #[must_use]
    pub const fn rng_salt(self) -> u64 {
        match self {
            Self::Count => 2,
            Self::Naive => 3,
            Self::SmallestOverall | Self::SmallestNovel => 0,
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
    /// Consume a constructed expression and reuse its node allocation for the wire
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

/// A per-seed candidate record the `candidates` binary prints to stdout (which
/// `guided_search.py` collects into `candidates.json`). Carries the
/// guide candidates for each requested pool
/// Python may restart with, plus replay metadata for Python's logging. The
/// goals and `max_size` are not here: the driver keeps them Python-side (from
/// `goal_terms.json`) and re-associates by seed.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "L: MyLanguage")]
pub struct SeedCandidates<L: MyLanguage> {
    pub start_term: String,
    pub policy: String,

    pub candidates: Vec<GuideExpr<L>>,
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
    /// Largest observed absolute live heap during guide replay.
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
    /// Largest observed absolute live heap during the unguided goal run.
    pub base_peak_live_heap: u64,
}
