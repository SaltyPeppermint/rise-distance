pub mod argtypes;
pub mod logging;
pub mod parquet;
pub mod types;

pub use logging::{_tee_print, init_log};
pub use types::*;

use std::env::current_dir;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use egg::{Analysis, Iteration, Language, Rewrite};
use hashbrown::HashSet;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::{BigUint, ToPrimitive};
use rayon::prelude::*;

use crate::cli::argtypes::{SampleStrategy, TermSampleDist};
use crate::count::TermCount;
use crate::egg::math::ConstantFold;
use crate::egg::{Math, ToEgg};
use crate::sampling::{CountSampler, NaiveSampler, Sampler, ZSDistanceSampler};
use crate::tee_println;
use crate::tree::{OriginTree, TreeShaped};
use crate::{Graph, Label, Tree, UnitCost, structural_diff, tree_distance_unit};

pub const TRIAL_SIZE: [usize; 6] = [1, 2, 5, 10, 50, 100];

pub static RULES: OnceLock<Vec<Rewrite<Math, ConstantFold>>> = OnceLock::new();

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

// const CUTOFF: usize = 1_000_000_000;

/// Sample frontier goal terms from `egraph` that are NOT present in `prev_raw_egg`.
pub fn sample_frontier_terms<L, N, LL>(
    graph: &Graph<LL>,
    prev_raw_egg: &egg::EGraph<L, N>,
    count: usize,
    max_size: usize,
    distribution: TermSampleDist,
    sample_strategy: SampleStrategy,
) -> Option<Vec<OriginTree<LL>>>
where
    L: Language,
    N: Analysis<L>,
    LL: Label,
    OriginTree<LL>: ToEgg<LL, Lang = L>,
{
    let tc = TermCount::<BigUint>::new(max_size, false, graph);

    let Some(histogram) = tc.data.get(&graph.root()) else {
        return Some(Vec::new());
    };

    let mut sorted_hist = histogram.iter().collect::<Vec<_>>();
    sorted_hist.sort_unstable();
    tee_println!("Terms in frontier:");
    for (k, v) in &sorted_hist {
        tee_println!("{v} terms of size {k}");
    }

    let min_size = histogram.keys().min().copied().unwrap_or(1);
    let mut result = HashSet::new();
    let mut oversample = 5;

    loop {
        let samples_per_size =
            distribution.samples_per_size(histogram, min_size, max_size, count * oversample);
        let batch = match sample_strategy {
            SampleStrategy::Naive => {
                NaiveSampler::new(&tc, graph).sample_batch_root(&samples_per_size)
            }
            SampleStrategy::CountBased => {
                CountSampler::new(&tc, graph).sample_batch_root(&samples_per_size)
            }
            SampleStrategy::ZSDiverseNaive => {
                ZSDistanceSampler::new(NaiveSampler::new(&tc, graph), UnitCost, 0.5, false)
                    .sample_batch_root(&samples_per_size)
            }
            SampleStrategy::ZSDiverseCountBased => {
                ZSDistanceSampler::new(CountSampler::new(&tc, graph), UnitCost, 0.5, false)
                    .sample_batch_root(&samples_per_size)
            }
        };

        result.extend(batch.into_iter().filter(|t| is_frontier(t, prev_raw_egg)));
        if result.len() >= count {
            break;
        }
        // None out if overflow
        oversample = oversample.checked_mul(2)?;
        tee_println!(
            "Have {}/{count} frontier terms, retrying with {oversample}x oversample...",
            result.len()
        );
    }
    Some(result.into_iter().take(count).collect())
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
