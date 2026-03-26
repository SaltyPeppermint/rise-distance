pub mod argtypes;
pub mod parquet;

use std::env::current_dir;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use egg::{Analysis, Iteration, Language, Rewrite};
use hashbrown::HashSet;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::{BigUint, ToPrimitive};
use rayon::prelude::*;
use serde::Serialize;

use crate::cli::argtypes::{SampleStrategy, SizeDistribution};
use crate::count::TermCount;
use crate::egg::math::ConstantFold;
use crate::egg::{Math, ToEgg};
use crate::sampling::{CountSampler, NaiveSampler, Sampler};
use crate::tree::{OriginTree, TreeShaped};
use crate::{
    Graph, Label, StructuralDistance, Tree, UnitCost, structural_diff, tree_distance_unit,
};

pub const TRIAL_SIZE: [usize; 6] = [1, 2, 5, 10, 50, 100];

static LOG_FILE: Mutex<Option<File>> = Mutex::new(None);

pub static RULES: OnceLock<Vec<Rewrite<Math, ConstantFold>>> = OnceLock::new();

/// Initialize the global log file. Call once at the start of `main` after creating the run folder.
///
/// # Panics
/// Panics if the log file cannot be created.
pub fn init_log(run_folder: &Path) {
    let file = File::create(run_folder.join("run.log")).expect("Failed to create run.log");
    *LOG_FILE.lock().unwrap() = Some(file);
}

/// Write a formatted message to both stdout and the log file.
#[doc(hidden)]
pub fn _tee_print(args: std::fmt::Arguments<'_>) {
    print!("{args}");
    if let Some(f) = LOG_FILE.lock().unwrap().as_mut() {
        let _ = f.write_fmt(args);
    }
}

/// Like `println!`, but also writes to the run log file.
#[macro_export]
macro_rules! tee_println {
    () => {
        $crate::cli::_tee_print(format_args!("\n"))
    };
    ($($arg:tt)*) => {{
        #[allow(clippy::used_underscore_items)]
        $crate::cli::_tee_print(format_args!($($arg)*));
        #[allow(clippy::used_underscore_items)]
        $crate::cli::_tee_print(format_args!("\n"));
    }};
}

/// Check if a term is in the frontier (i.e. NOT present in `prev_raw_egg`).
pub fn is_frontier<T, L, N, LL>(tree: &T, prev_raw_egg: &egg::EGraph<L, N>) -> bool
where
    L: Language,
    N: Analysis<L>,
    LL: Label,
    T: ToEgg<LL, Lang = L>,
{
    prev_raw_egg.lookup_expr(&tree.to_rec_expr()).is_none()
}

/// Create an output folder for a run.
///
/// If `output` is `Some`, uses that path directly. Otherwise, auto-generates a
/// path like `data/<subdir>/run-<prefix>-sampling.<N>` where `<N>` is one higher
/// than the largest existing run number.
#[expect(clippy::missing_panics_doc)]
pub fn get_run_folder(output: Option<&str>, subdir: &str, prefix: &str) -> PathBuf {
    let this_run_dir = output.map_or_else(
        || {
            let runs_dir = current_dir().unwrap().join("data").join(subdir);
            std::fs::create_dir_all(&runs_dir).expect("Failed to create output directory");
            let max_existing = runs_dir
                .read_dir()
                .unwrap()
                .filter_map(|e| {
                    let d = e.ok()?;
                    if d.file_type().ok()?.is_dir() && prefix == d.path().file_stem()? {
                        return d.path().extension()?.to_str()?.parse::<usize>().ok();
                    }
                    None
                })
                .max()
                .unwrap_or(0);
            runs_dir
                .join(prefix)
                .with_extension((max_existing + 1).to_string())
        },
        PathBuf::from,
    );
    std::fs::create_dir_all(&this_run_dir).expect("Failed to create output directory");
    this_run_dir
}

#[expect(clippy::cast_precision_loss)]
pub fn trial_avg<F: Fn(&Vec<Iteration<()>>) -> Option<usize>>(
    trials: &[Option<Vec<Iteration<()>>>],
    f: F,
) -> Option<f64> {
    let values: Vec<usize> = trials
        .iter()
        .filter_map(|x| x.as_ref().and_then(&f))
        .collect();
    if values.is_empty() {
        return None;
    }
    let avg = values.iter().sum::<usize>() as f64 / values.len() as f64;
    Some(avg)
}

#[expect(clippy::missing_panics_doc)]
pub fn min_med_max<T: Ord + Copy, I, F: Fn(&I) -> T>(items: &[I], f: F) -> (T, T, T) {
    let min = items.iter().map(&f).min().unwrap();
    let max = items.iter().map(&f).max().unwrap();
    let med = f(&items[items.len() / 2]);
    (min, med, max)
}

/// Measure guides by distance to the goal.
pub fn measure_guides<L: Label>(
    guides: &[OriginTree<L>],
    goal: &OriginTree<L>,
) -> Vec<MeasuredGuide<L>> {
    let goal_flat = goal.flatten(false);
    #[expect(clippy::missing_panics_doc)]
    let pb_style = ProgressStyle::with_template(
        "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}<{eta_precise}] ranking guides",
    )
    .unwrap();
    guides
        .par_iter()
        .progress_with_style(pb_style)
        .map(|guide| {
            let guide_flat = guide.flatten(false);
            let zs_dist = tree_distance_unit(&guide_flat, &goal_flat);
            let structural_dist = structural_diff(&goal_flat, &guide_flat, &UnitCost);
            MeasuredGuide {
                guide: guide.clone(),
                zs_distance: zs_dist,
                structural_distance: structural_dist,
            }
        })
        .collect()
}

/// Sample frontier goal terms from `egraph` that are NOT present in `prev_raw_egg`.
pub fn sample_frontier_terms<L, N, LL>(
    graph: &Graph<LL>,
    prev_raw_egg: &egg::EGraph<L, N>,
    count: usize,
    max_size: usize,
    distribution: SizeDistribution,
    sample_strategy: SampleStrategy,
) -> Vec<OriginTree<LL>>
where
    L: Language,
    N: Analysis<L>,
    LL: Label,
    OriginTree<LL>: ToEgg<LL, Lang = L>,
{
    let tc = TermCount::<BigUint>::new(max_size, false, graph);

    let Some(histogram) = tc.data.get(&graph.root()) else {
        return Vec::new();
    };

    let mut sorted_hist = histogram.iter().collect::<Vec<_>>();
    sorted_hist.sort_unstable();
    tee_println!("Terms in frontier:");
    for (k, v) in &sorted_hist {
        tee_println!("{v} terms of size {k}");
    }

    let min_size = histogram.keys().min().copied().unwrap_or(1);
    #[expect(clippy::cast_precision_loss)]
    let normal_center = (min_size + max_size) as f64 / 2.0;

    let mut result = HashSet::new();
    let mut oversample = 5;
    loop {
        let samples_per_size = distribution.samples_per_size(
            histogram,
            min_size,
            max_size,
            count * oversample,
            normal_center,
        );
        let batch = match sample_strategy {
            SampleStrategy::Naive => NaiveSampler::new(&tc, graph).sample_batch_root(
                min_size,
                max_size,
                &samples_per_size,
            ),
            SampleStrategy::CountBased => CountSampler::new(&tc, graph).sample_batch_root(
                min_size,
                max_size,
                &samples_per_size,
            ),
        };

        let prev_len = result.len();
        result.extend(batch.into_iter().filter(|t| is_frontier(t, prev_raw_egg)));
        if result.len() >= count || result.len() == prev_len {
            break;
        }
        oversample *= 2;
        tee_println!(
            "Have {}/{count} frontier terms, retrying with {oversample}x oversample...",
            result.len()
        );
    }
    result.into_iter().take(count).collect()
}

/// Enumerate all frontier terms from `egraph` that are NOT present in `prev_raw_egg`.
#[expect(clippy::missing_panics_doc)]
pub fn enumerate_frontier_terms<L, N, LL>(
    graph: &Graph<LL>,
    prev_raw_egg: &egg::EGraph<L, N>,
    max_size: usize,
) -> Vec<OriginTree<LL>>
where
    L: Language,
    N: Analysis<L>,
    LL: Label,
    OriginTree<LL>: ToEgg<LL, Lang = L>,
    Tree<LL>: ToEgg<LL, Lang = L>,
{
    let tc = TermCount::<BigUint>::new(max_size, false, graph);

    let Some(histogram) = tc.data.get(&graph.root()) else {
        return Vec::new();
    };

    let mut sorted_hist = histogram.iter().collect::<Vec<_>>();
    sorted_hist.sort_unstable();
    tee_println!("Terms in frontier:");
    for (k, v) in &sorted_hist {
        tee_println!("{v} terms of size {k}");
    }
    let start = Instant::now();
    let total_terms = histogram.values().cloned().sum::<BigUint>();
    tee_println!("Enumerating all {total_terms} terms up to size {max_size}");
    assert!(
        total_terms.to_usize().is_some(),
        "Cannot enumerate more than usize!"
    );

    let result = tc
        .enumerate_root(
            graph,
            max_size,
            Some(ProgressBar::new(max_size.try_into().unwrap())),
        )
        .into_iter()
        .filter(|t| is_frontier(t, prev_raw_egg))
        .collect::<Vec<_>>();
    tee_println!(
        "Spent {} seconds enumerating the terms",
        start.elapsed().as_secs()
    );
    result
}

#[derive(Serialize, Debug)]
pub struct GuideEval<'a, L: Label> {
    pub guide: &'a MeasuredGuide<L>,
    pub iterations: Option<Vec<Iteration<()>>>,
}

#[derive(Serialize, Debug, PartialEq, Eq, Hash)]
pub struct MeasuredGuide<L: Label> {
    pub guide: OriginTree<L>,
    pub zs_distance: usize,
    #[serde(flatten)]
    pub structural_distance: StructuralDistance,
}

#[derive(Serialize)]
pub struct GuideSetTrials {
    pub k: usize,
    pub trials: Vec<Option<Vec<Iteration<()>>>>,
}

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
pub struct GuideSetSummary {
    pub k: usize,
    pub trials: Vec<Option<TrialSummary>>,
}

#[derive(Serialize)]
pub struct GoalSummary {
    pub goal: String,
    pub entries: Vec<GuideSetSummary>,
}

impl GoalSummary {
    /// Build a summary from the full `GuideSetTrials` data for a given goal.
    ///
    /// # Panics
    /// Panics if a reachable trial has an empty iteration list.
    #[must_use]
    pub fn from_entries(goal: &str, entries: &[GuideSetTrials]) -> Self {
        Self {
            goal: goal.to_owned(),
            entries: entries
                .iter()
                .map(|e| GuideSetSummary {
                    k: e.k,
                    trials: e
                        .trials
                        .iter()
                        .map(|trial| {
                            trial.as_ref().map(|iters| {
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
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}
