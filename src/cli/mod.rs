pub mod argtypes;
pub mod logging;
pub mod parquet;
pub mod types;

pub use logging::{_tee_print, init_log};
pub use types::*;

use std::env::current_dir;
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use egg::{Analysis, Iteration, Language, Rewrite};
use hashbrown::HashSet;
use indicatif::ProgressBar;
use num::{BigUint, ToPrimitive};
use rayon::prelude::*;

use crate::cli::argtypes::{SampleStrategy, TermSampleDist};
use crate::count::{Counter, TermCount};
use crate::egg::math::ConstantFold;
use crate::egg::{Math, ToEgg, convert};
use crate::sampling::{CountSampler, NaiveSampler, Sampler, ZSDistanceSampler};
use crate::tee_println;
use crate::tree::{OriginTree, TreeShaped, UnfoldedTree};
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

pub fn trial_avg<
    F: Fn(&Vec<Iteration<()>>) -> Option<T>,
    T: for<'a> std::iter::Sum<&'a T> + ToPrimitive,
>(
    trials: &[Option<Vec<Iteration<()>>>],
    f: F,
) -> Option<f64> {
    let values: Vec<_> = trials
        .iter()
        .filter_map(|x| x.as_ref().and_then(&f))
        .collect();
    if values.is_empty() {
        return None;
    }
    let avg = values.iter().sum::<T>().to_f64()? / values.len().to_f64()?;
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
pub fn measure_guide<L: Label>(
    guide: &OriginTree<L>,
    goal_unfolded: &UnfoldedTree<L>,
) -> Measurements {
    let guide_unfolded = guide.flatten(false);
    let zs_dist = tree_distance_unit(&guide_unfolded, goal_unfolded);
    let structural_dist = structural_diff(goal_unfolded, &guide_unfolded, &UnitCost);
    Measurements {
        zs_distance: zs_dist,
        structural_distance: structural_dist,
    }
}

const OVERSAMPLE_START: usize = 20;

const OVERSAMPLE_CUTOFF: usize = 1_000_000;

const OVERSAMPLE_SCHEDULE: [usize; 16] = {
    let mut arr = [0usize; 16];
    let mut v = OVERSAMPLE_START;
    let mut i = 0;
    while v < OVERSAMPLE_CUTOFF {
        arr[i] = v;
        i += 1;
        v *= 2;
    }
    arr
};

pub struct PrecomputePackage<C, LL, L, N>
where
    L: Language + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Sync,
    N::Data: Sync,
    LL: Label + for<'a> std::convert::From<&'a L>,
    OriginTree<LL>: ToEgg<LL, Lang = L>,
    C: Counter + Display + Ord,
{
    tc: TermCount<C>,
    min_size: usize,
    max_size: usize,
    prev_raw_egg: egg::EGraph<L, N>,
    graph: Graph<LL>,
}

impl<C, LL, L, N> PrecomputePackage<C, LL, L, N>
where
    L: Language + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Sync,
    N::Data: Sync,
    LL: Label + for<'a> std::convert::From<&'a L>,
    OriginTree<LL>: ToEgg<LL, Lang = L>,
    C: Counter + Display + Ord,
{
    pub fn precompute(
        guide_egg: &egg::EGraph<L, N>,
        prev_raw_egg: egg::EGraph<L, N>,
        root: egg::Id,
        max_size: usize,
    ) -> Option<PrecomputePackage<C, LL, L, N>> {
        let graph = convert(guide_egg, root);
        let tc = TermCount::<C>::new(max_size, false, &graph);
        let histogram = tc.data.get(&graph.root())?;
        let mut sorted_hist = histogram
            .iter()
            .map(|(a, b)| (*a, b.to_owned()))
            .collect::<Vec<_>>();
        sorted_hist.sort_unstable();
        tee_println!("Terms in frontier:");
        for (k, v) in &sorted_hist {
            tee_println!("{v} terms of size {k}");
        }

        let min_size = histogram.keys().min().copied().unwrap_or(1);
        Some(PrecomputePackage {
            tc,
            min_size,
            max_size,
            prev_raw_egg,
            graph,
        })
    }

    /// Sample frontier goal terms from `egraph` that are NOT present in `prev_raw_egg`.
    pub fn sample_frontier_terms(
        &self,
        count: usize,
        distribution: TermSampleDist,
        sample_strategy: SampleStrategy,
    ) -> Option<Vec<OriginTree<LL>>>
    where
        L: Language + Sync,
        L::Discriminant: Sync,
        N: Analysis<L> + Sync,
        N::Data: Sync,
        LL: Label,
        OriginTree<LL>: ToEgg<LL, Lang = L>,
    {
        let histogram = self.tc.data.get(&self.graph.root())?;
        OVERSAMPLE_SCHEDULE.iter().find_map(|oversample| {
        let samples_per_size =
            distribution.samples_per_size(histogram, self.min_size, self.max_size, count * oversample);
        let batch = match sample_strategy {
            SampleStrategy::Naive => {
                NaiveSampler::new(&self.tc, &self.graph).sample_batch_root(&samples_per_size)
            }
            SampleStrategy::CountBased => {
                CountSampler::new(&self.tc, &self.graph).sample_batch_root(&samples_per_size)
            }
            SampleStrategy::ZSDiverseNaive => {
                ZSDistanceSampler::new(NaiveSampler::new(&self.tc, &self.graph), UnitCost, 0.5, false)
                    .sample_batch_root(&samples_per_size)
            }
            SampleStrategy::ZSDiverseCountBased => {
                ZSDistanceSampler::new(CountSampler::new(&self.tc, &self.graph), UnitCost, 0.5, false)
                    .sample_batch_root(&samples_per_size)
            }
        };

        let results = batch
            .into_par_iter()
            .filter(|t| is_frontier(t, &self.prev_raw_egg))
            .collect::<HashSet<_>>();
        if results.len() >= count {
            Some(results.into_par_iter().take_any(count).collect())
        } else {
            tee_println!(
                "Have {}/{count} frontier terms with {oversample}x oversampling, retrying with double that...",
                results.len()
            );
            None
        }
    })
    }
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
