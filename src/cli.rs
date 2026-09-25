use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::utils;

/// Envelope every JSON-emitting binary prints: the payload plus this process's
/// lifetime peak RSS.
#[derive(Serialize, Debug, Clone)]
pub struct Measured<T> {
    pub peak_rss_bytes: u64,
    pub payload: T,
}

impl<T> Measured<T> {
    /// Wrap `payload`, reading the peak RSS now.
    ///
    /// Call this once the run is over but before serializing, so the reading
    /// covers the search rather than the JSON write that follows.
    #[must_use]
    pub fn now(payload: T) -> Self {
        Self {
            peak_rss_bytes: utils::peak_rss_bytes(),
            payload,
        }
    }
}

/// How to sample the novel frontier when drawing a guides.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    #[value(name = "count")]
    Count,
    #[value(name = "uniform")]
    Uniform,
    #[value(name = "smallest")]
    Smallest,
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
            Self::Smallest => write!(f, "smallest"),
        }
    }
}
