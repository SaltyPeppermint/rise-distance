use std::env::current_dir;
use std::fmt::Display;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use csv::WriterBuilder;
use egg::{Analysis, Language, Rewrite};
use hashbrown::{HashMap, HashSet};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use num::{BigUint, ToPrimitive};
use rayon::prelude::*;
use serde::Serialize;

use crate::count::TermCount;
use crate::egg::math::ConstantFold;
use crate::egg::{Math, VerifyResult};
use crate::sampling::Sampler;
use crate::sampling::count::CountSampler;
use crate::{
    Graph, Label, StructuralDistance, TreeNode, UnitCost, structural_diff, tree_distance_unit,
};

pub const N_RANDOM: [usize; 6] = [1, 2, 5, 10, 50, 100];

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

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum DistanceMetric {
    ZhangShasha,
    Structural,
}

impl Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZhangShasha => write!(f, "zhang-shasha"),
            Self::Structural => write!(f, "structural"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SampleDistribution {
    /// Uniform accross the term sizes
    Uniform,
    /// Proportional to the number of terms of that size with a minimum number per size
    Proportional(usize),
    /// As a normal distribution centered in the middle between the smallest and goal term
    /// value = sigma
    Normal(f64),
}

impl Display for SampleDistribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform => write!(f, "uniform"),
            Self::Proportional(min) => write!(f, "proportional:{min}"),
            Self::Normal(sigma) => write!(f, "normal:{sigma}"),
        }
    }
}

impl FromStr for SampleDistribution {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "uniform" {
            return Ok(Self::Uniform);
        }
        if s == "proportional" {
            return Ok(Self::Proportional(10));
        }

        if let Some(rest) = s.strip_prefix("proportional:") {
            let min = rest
                .parse::<usize>()
                .map_err(|e| format!("invalid min_per_size in 'proportional:{rest}': {e}"))?;
            return Ok(Self::Proportional(min));
        }

        if s == "normal" {
            return Ok(Self::Normal(2.6));
        }
        if let Some(rest) = s.strip_prefix("normal:") {
            let sigma = rest
                .parse::<f64>()
                .map_err(|e| format!("invalid sigma in 'normal:{rest}': {e}"))?;
            return Ok(Self::Normal(sigma));
        }
        Err(format!(
            "unknown distribution '{s}': expected 'uniform', 'proportional:<min>', or 'normal:<sigma>'"
        ))
    }
}

impl SampleDistribution {
    /// Build a `samples_per_size` map distributing `total_samples` across `[min_size, max_size]`.
    ///
    /// `histogram` maps size -> term count for the root e-class.
    /// `normal_center` is the center of the Gaussian (only used for `Normal`).
    #[must_use]
    pub fn samples_per_size(
        self,
        histogram: &HashMap<usize, BigUint>,
        min_size: usize,
        max_size: usize,
        total_samples: usize,
        normal_center: f64,
    ) -> HashMap<usize, u64> {
        match self {
            Self::Uniform => {
                let num_sizes = (max_size - min_size).max(1);
                let s = (total_samples / num_sizes).max(1) as u64;
                (min_size..=max_size).map(|size| (size, s)).collect()
            }
            Self::Proportional(min_per_size) => {
                let total_terms: BigUint = (min_size..=max_size)
                    .filter_map(|s| histogram.get(&s))
                    .sum();
                let budget = BigUint::from(total_samples);
                (min_size..=max_size)
                    .map(|size| {
                        let n = histogram.get(&size).map_or(0, |count| {
                            (count * &budget / &total_terms)
                                .to_u64()
                                .unwrap_or(u64::MAX)
                                .max(min_per_size as u64)
                        });
                        (size, n)
                    })
                    .collect()
            }
            Self::Normal(sigma) => {
                let weights: HashMap<usize, f64> = (min_size..=max_size)
                    .map(|s| {
                        #[expect(clippy::cast_precision_loss)]
                        let z = (s as f64 - normal_center) / sigma;
                        (s, (-0.5 * z * z).exp())
                    })
                    .collect();
                let total_weight: f64 = weights.values().sum();
                (min_size..=max_size)
                    .map(|size| {
                        let w = *weights.get(&size).unwrap_or(&0.0);
                        #[expect(
                            clippy::cast_precision_loss,
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss
                        )]
                        let n = (w / total_weight * total_samples as f64).round() as u64;
                        (size, n)
                    })
                    .collect()
            }
        }
    }
}

/// Check if a term is in the frontier (i.e. NOT present in `prev_raw_egg`).
pub fn is_frontier<L, N, LL>(tree: &TreeNode<LL>, prev_raw_egg: &egg::EGraph<L, N>) -> bool
where
    L: Language,
    N: Analysis<L>,
    LL: Label,
    for<'a> &'a TreeNode<LL>: Into<egg::RecExpr<L>>,
{
    prev_raw_egg.lookup_expr(&tree.into()).is_none()
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

#[derive(Serialize, Debug, PartialEq, Eq, Hash)]
pub struct EvalResult<'a, L: Label> {
    pub guide: &'a MeasuredGuide<L>,
    pub values: Option<VerifyResult>,
}

/// Dump the eval of every term to the csv, appending if the file already exists
///
/// # Panics
///
/// Panics if it cant create/open the file
pub fn dump_to_csv<L: Label>(run_folder: &Path, goal: &TreeNode<L>, results: &[EvalResult<'_, L>]) {
    let csv_path = run_folder.join("out.csv");
    let file_exists = csv_path.exists();
    let csv_output = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&csv_path)
        .expect("Failed to open output file");
    let mut csv_writer = WriterBuilder::new()
        .has_headers(!file_exists)
        .from_writer(csv_output);
    for r in results {
        csv_writer
            .serialize(CsvRow::new(goal, r.guide, r.values))
            .expect("write CSV row");
    }
    tee_println!("Wrote goal to CSV");
    csv_writer.flush().expect("flush output");
}

#[derive(Serialize)]
struct CsvRow {
    goal: String,
    guide: String,
    zs_distance: usize,
    structural_overlap: usize,
    structural_zs_sum: usize,
    iterations_to_reach: Option<usize>,
    nodes_to_reach: Option<usize>,
}

impl CsvRow {
    fn new<L: Label>(
        goal: &TreeNode<L>,
        guide: &MeasuredGuide<L>,
        values: Option<VerifyResult>,
    ) -> Self {
        Self {
            goal: goal.to_string(),
            guide: guide.guide.to_string(),
            zs_distance: guide.zs_distance,
            structural_overlap: guide.structural_distance.overlap(),
            structural_zs_sum: guide.structural_distance.zs_sum(),
            iterations_to_reach: values.map(|v| v.iters),
            nodes_to_reach: values.map(|v| v.nodes),
        }
    }
}

#[derive(Serialize, Debug, PartialEq, Eq, Hash)]
pub struct MeasuredGuide<L: Label> {
    pub guide: TreeNode<L>,
    pub zs_distance: usize,
    #[serde(flatten)]
    pub structural_distance: StructuralDistance,
}

#[derive(Serialize, PartialEq, Eq)]
pub struct RandomTrial {
    pub single_iters: Vec<Option<VerifyResult>>,
    pub combined_iters: Option<usize>,
    pub combined_nodes: Option<usize>,
}

#[derive(Serialize, PartialEq, Eq)]
pub struct RandomEntry {
    pub k: usize,
    pub trials: Vec<RandomTrial>,
}

#[expect(clippy::cast_precision_loss)]
pub fn trial_avg<F: Fn(&RandomTrial) -> Option<usize>>(
    trials: &[RandomTrial],
    f: F,
) -> Option<(f64, usize)> {
    let values: Vec<usize> = trials.iter().filter_map(&f).collect();
    if values.is_empty() {
        return None;
    }
    let avg = values.iter().sum::<usize>() as f64 / values.len() as f64;
    Some((avg, values.len()))
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
    guides: &[TreeNode<L>],
    goal: &TreeNode<L>,
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
    egraph: &Graph<LL>,
    prev_raw_egg: &egg::EGraph<L, N>,
    count: usize,
    max_size: usize,
    distribution: SampleDistribution,
) -> Vec<TreeNode<LL>>
where
    L: Language,
    N: Analysis<L>,
    LL: Label,
    for<'a> &'a TreeNode<LL>: Into<egg::RecExpr<L>>,
{
    let tc = TermCount::<BigUint, _>::new(max_size, false, egraph);

    let Some(histogram) = tc.of_root() else {
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
        let batch =
            CountSampler::new(&tc).sample_unique_root(min_size, max_size, &samples_per_size);
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
    egraph: &Graph<LL>,
    prev_raw_egg: &egg::EGraph<L, N>,
    max_size: usize,
) -> Vec<TreeNode<LL>>
where
    L: Language,
    N: Analysis<L>,
    LL: Label,
    for<'a> &'a TreeNode<LL>: Into<egg::RecExpr<L>>,
{
    let tc = TermCount::<BigUint, _>::new(max_size, false, egraph);

    let Some(histogram) = tc.of_root() else {
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
        .enumerate_root(max_size, Some(ProgressBar::new(max_size as u64)))
        .into_iter()
        .filter(|t| is_frontier(t, prev_raw_egg))
        .collect::<Vec<_>>();
    tee_println!(
        "Spent {} seconds enumerating the terms",
        start.elapsed().as_secs()
    );
    result
}
