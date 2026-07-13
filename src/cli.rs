//! Shared CLI + wire types for the `goal` / `sample` / `verify` split of the
//! guide experiment. The three binaries touch no files: the Python drivers
//! (`goal_driver.py` / `driver.py`) own all I/O, passing eqsat limits and
//! language on argv ([`EqsatArgs`]) and per-seed/per-leg data as the JSON
//! records defined here. `goal` records [`GoalGenMetadata`] into `terms.json`;
//! `sample` emits [`SeedSamples`] (guide menus of [`GuideExpr`], one pool per
//! [`Strategy`]); `driver.py` feeds chosen subsets back to `verify`.

use std::collections::BTreeMap;

use clap::Args;
use egg::RecExpr;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::Counter;
use crate::eqsat::{EqsatConfig, EqsatMetadata};
use crate::sampling::SampleStrategy;
use crate::{MyLanguage, OriginLang};

/// Eqsat resource limits shared by the `goal` / `sample` / `verify` binaries as
/// CLI flags. The Python drivers read these four values out of the folder's
/// `args.json` and forward them on argv, so the binaries never touch a file.
#[derive(Args, Clone, Copy)]
pub struct EqsatArgs {
    /// Maximum eqsat iterations.
    #[arg(long)]
    pub max_iters: usize,

    /// Maximum eqsat egraph nodes.
    #[arg(long)]
    pub max_nodes: usize,

    /// Maximum eqsat wall-clock seconds.
    #[arg(long)]
    pub max_time: f64,

    /// Use the backoff scheduler instead of the simple one.
    #[arg(long)]
    pub backoff_scheduler: bool,
}

impl From<EqsatArgs> for EqsatConfig {
    fn from(a: EqsatArgs) -> Self {
        Self {
            max_iters: a.max_iters,
            max_nodes: a.max_nodes,
            max_time: a.max_time,
            backoff_scheduler: a.backoff_scheduler,
        }
    }
}

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

/// A per-seed sampling record `sample` prints to stdout (which `driver.py`
/// collects into `samples.json`). Carries the goal menu (lowered s-expression
/// strings, as stored by `goal`) and, per strategy, the guide candidates Python
/// may restart with, plus enough replay metadata for Python's logging.
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

/// Per-seed payload written by `goal` into the value slot of `terms.json` (one
/// entry per seed s-expression). Serializes via `Result`'s `{"Ok": ..}` /
/// `{"Err": ..}` shape (`goal` returns a `Result<GoalGenMetadata, String>`).
/// `driver.py` parses the enriched `terms.json` and feeds each `Ok` seed's
/// replay inputs to `sample` on stdin.
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
}
