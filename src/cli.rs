//! Shared wire types for the guide experiment's `candidates` and `verify`
//! binaries.

use clap::ValueEnum;
use egg::RecExpr;
use serde::{Deserialize, Serialize};

use crate::{MyLanguage, OriginLang};

/// How `candidates` samples the novel frontier when drawing a guides.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    #[value(name = "count")]
    Count,
    #[value(name = "uniform")]
    Uniform,
    // #[value(name = "smallest_overall")]
    // SmallestOverall,
    // #[value(name = "smallest_novel")]
    // SmallestNovel,
}

impl std::fmt::Display for Policy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count => write!(f, "count"),
            Self::Uniform => write!(f, "uniform"),
            // Self::SmallestOverall => write!(f, "smallest_overall"),
            // Self::SmallestNovel => write!(f, "smallest_novel"),
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
            Self::Uniform => 3,
            // Self::SmallestOverall | Self::SmallestNovel => 0,
        }
    }
}

/// For Serialization purposes we have to go via Vec instead of using `RecExpr`
#[derive(Serialize, Debug, Clone)]
pub struct Candidates<L: MyLanguage> {
    pub start_term: String,
    pub policy: String,

    pub candidates: Vec<Vec<OriginLang<L>>>,
    pub candidate_s_expr: Vec<RecExpr<L>>,
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
