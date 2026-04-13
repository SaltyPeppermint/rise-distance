use egg::Iteration;
use hashbrown::HashMap;
use serde::Serialize;
use strum::Display;
use thiserror::Error;

use crate::tree::OriginTree;
use crate::{Label, StructuralDistance};

#[derive(Debug, Error, Display, Serialize, Clone, Copy)]
pub enum GuideError {
    InsufficientSamples,
    Unreached,
}

#[derive(Serialize, Debug)]
pub struct GuideEval<L: Label> {
    pub guide: OriginTree<L>,
    pub measurements: Measurements,
    pub iterations: Option<Vec<Iteration<()>>>,
}

#[derive(Serialize, Debug, PartialEq, Eq, Hash, Clone)]
pub struct Measurements {
    pub zs_distance: usize,
    #[serde(flatten)]
    pub structural_distance: StructuralDistance,
}

/// Type alias for the per-k trial data: maps each guide-set size `k` to its
/// trial results (one `Option` per trial. `None` means the goal was not reached).
pub type TrialsPerK = HashMap<usize, Vec<Result<Vec<Iteration<()>>, GuideError>>>;

/// Same as `TrialsPerK` but with pre-computed summaries instead of full iteration data.
pub type SummaryPerK = HashMap<usize, Vec<Result<TrialSummary, GuideError>>>;

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
                                    .map_err(|e| *e)
                            })
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}
