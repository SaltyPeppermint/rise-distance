mod count;
mod sampler;
mod weigher;
// mod zs_min_distance;

use std::fmt::Display;
use std::iter::{Product, Sum};
use std::str::FromStr;

use egg::{Id, RecExpr};
use hashbrown::HashMap;
use num_traits::{NumAssignRef, NumRef};
use rand::distributions::uniform::SampleUniform;
use serde::Serialize;

use crate::eqsat::EqsatResult;
use crate::sampling::count::{NovelTermCount, PlainTermCount};
use crate::{MyAnalysis, MyLanguage, OriginLang};

// TODO: reenable zs_min_distance sampler
// pub use zs_min_distance::ZSDistanceSampler;
pub use sampler::{NovelSampler, PlainSampler, Sampler};
pub use weigher::{CountWeigher, NaiveWeigher, Weigher};

pub trait Counter:
    Clone
    + Send
    + Sync
    + NumRef
    + NumAssignRef
    + Default
    + std::fmt::Debug
    + Display
    + SampleUniform
    + PartialEq
    + Ord
    + for<'a> Sum<&'a Self>
    + TryInto<u64, Error: std::fmt::Debug>
    + TryFrom<u64, Error: std::fmt::Debug>
    + TryFrom<usize, Error: std::fmt::Debug>
    + Product // + Weight
{
}

impl<
    T: Clone
        + Send
        + Sync
        + NumRef
        + NumAssignRef
        + Default
        + std::fmt::Debug
        + Display
        + SampleUniform
        + PartialEq
        + Ord
        + for<'a> Sum<&'a Self>
        + TryInto<u64, Error: std::fmt::Debug>
        + TryFrom<u64, Error: std::fmt::Debug>
        + TryFrom<usize, Error: std::fmt::Debug>
        + Product, // + Weight,
> Counter for T
{
}

pub struct PrecomputePackage<'a, C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    tc: NovelTermCount<'a, C, L, N>,
    min_size: usize,
    max_size: usize,
    root: Id,
}

impl<'a, C, L, N> PrecomputePackage<'a, C, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    C: Counter,
{
    /// Enumerate all frontier terms from `egraph` that are NOT present in `prev_raw_egg` for the sampling process later
    #[must_use]
    pub fn precompute(
        result: &'a EqsatResult<L, N>,
        max_size: usize,
    ) -> Option<PrecomputePackage<'a, C, L, N>> {
        let tc = NovelTermCount::new(
            max_size,
            result.curr(),
            result.prev(),
            PlainTermCount::new(max_size, result.curr()),
        );

        let root = result.curr().find(result.root());
        let histogram = tc.data().get(&root)?;

        let min_size = histogram.keys().min().copied().unwrap_or(1);
        Some(PrecomputePackage {
            tc,
            min_size,
            max_size,
            root,
        })
    }

    /// Log the stats about the root
    ///
    /// # Panics
    ///
    /// Panics if there are no terms in the root
    pub fn log_root(&self) {
        let histogram = self
            .tc
            .data()
            .get(&self.root)
            .expect("Somehow the root does not contain any terms?");
        let mut sorted_hist = histogram
            .iter()
            .map(|(a, b)| (*a, b.to_owned()))
            .collect::<Vec<_>>();
        sorted_hist.sort_unstable();
        println!("Terms in frontier:");
        for (k, v) in &sorted_hist {
            println!("{v} terms of size {k}");
        }
    }

    /// Sample frontier goal terms from `egraph` that are NOT present in `prev_raw_egg`.
    #[must_use]
    pub fn sample_frontier_terms(
        &self,
        count: usize,
        distribution: TermSampleDist,
        sample_strategy: SampleStrategy,
        seed: [u64; 2],
        novel: bool,
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let histogram = self.tc.data().get(&self.root)?;

        let samples_per_size =
            distribution.samples_per_size(histogram, self.min_size, self.max_size, count);

        match (sample_strategy, novel) {
            (SampleStrategy::Naive, true) => NovelSampler::new(&self.tc, self.root, NaiveWeigher)
                .sample_batch_root(&samples_per_size, seed),
            (SampleStrategy::Count, true) => NovelSampler::new(&self.tc, self.root, CountWeigher)
                .sample_batch_root(&samples_per_size, seed),
            (SampleStrategy::Naive, false) => {
                PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, NaiveWeigher)
                    .sample_batch_root(&samples_per_size, seed)
            }
            (SampleStrategy::Count, false) => {
                PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, CountWeigher)
                    .sample_batch_root(&samples_per_size, seed)
            }
        }
    }

    #[must_use]
    pub fn smallest(&self, id: Id, novel: bool) -> RecExpr<OriginLang<L>> {
        if novel {
            NovelSampler::new(&self.tc, self.root, NaiveWeigher).smallest(id)
        } else {
            PlainSampler::new(self.tc.plain(), self.tc.curr(), self.root, NaiveWeigher).smallest(id)
        }
    }

    #[must_use]
    pub fn root(&self) -> Id {
        self.root
    }

    /// Histogram of novel root extractions by size. Guaranteed non-empty
    /// because [`Self::precompute`] returns `None` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the root histogram is somehow missing (would indicate a bug
    /// in `precompute`'s None check).
    #[must_use]
    pub fn root_histogram(&self) -> &HashMap<usize, C> {
        self.tc
            .data()
            .get(&self.root)
            .expect("root histogram present iff precompute returned Some")
    }
}

#[derive(Serialize, Debug, Clone, Copy, clap::ValueEnum, strum::Display)]
#[strum(serialize_all = "kebab-case")]
pub enum SampleStrategy {
    Naive,
    Count,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum Distribution {
    /// Uniform across term sizes
    Uniform,
    /// Normal distribution centered between min and max size
    /// value = sigma
    Normal(f64),
}

impl Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform => write!(f, "uniform"),
            Self::Normal(sigma) => write!(f, "normal:{sigma}"),
        }
    }
}

impl FromStr for Distribution {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "uniform" => Ok(Self::Uniform),
            "normal" => Ok(Self::Normal(2.6)),
            _ => {
                if let Some(rest) = s.strip_prefix("normal:") {
                    rest.parse::<f64>()
                        .map(Self::Normal)
                        .map_err(|e| format!("invalid sigma in 'normal:{rest}': {e}"))
                } else {
                    Err(format!(
                        "unknown distribution '{s}': expected 'uniform' or 'normal:<sigma>'"
                    ))
                }
            }
        }
    }
}

impl Distribution {
    /// Distribute `total_samples` across the given `sizes`, returning a size => count list.
    ///
    /// `sizes` must be sorted ascending. For `Normal`, the bell curve is centered between
    /// `sizes.first()` and `sizes.last()`, so callers that want to skip empty histogram
    /// buckets should pre-filter accordingly.
    #[expect(
        clippy::missing_panics_doc,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn samples_per_size(self, sizes: &[usize], total_samples: usize) -> Vec<(usize, u64)> {
        if sizes.is_empty() {
            return vec![];
        }
        match self {
            Self::Uniform => {
                let num_sizes = sizes.len();
                let base = (total_samples / num_sizes) as u64;
                let remainder = total_samples % num_sizes;
                sizes
                    .iter()
                    .enumerate()
                    .map(|(i, &size)| (size, base + u64::from(i < remainder)))
                    .collect()
            }
            Self::Normal(sigma) => {
                let first = *sizes.first().unwrap() as f64;
                let last = *sizes.last().unwrap() as f64;
                let normal_center = f64::midpoint(first, last);
                let weights = sizes
                    .iter()
                    .map(|&s| {
                        let z = (s as f64 - normal_center) / sigma;
                        (-0.5 * z * z).exp()
                    })
                    .collect::<Vec<_>>();
                let total_weight = weights.iter().sum::<f64>();
                // Largest remainder method: compute exact quotas, floor them, then
                // distribute the remaining counts to the sizes with the largest remainders.
                let quotas = weights
                    .iter()
                    .map(|w| w / total_weight * total_samples as f64)
                    .collect::<Vec<_>>();
                let floors = quotas.iter().map(|&q| q as u64).collect::<Vec<_>>();
                let allocated = floors.iter().sum::<u64>();
                let remainder = (total_samples as u64) - allocated;
                let mut remainders = quotas
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| (i, q - floors[i] as f64))
                    .collect::<Vec<_>>();
                remainders
                    .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                let mut counts = floors;
                for (i, _) in remainders.iter().take(remainder as usize) {
                    counts[*i] += 1;
                }
                sizes.iter().copied().zip(counts).collect()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum TermSampleDist {
    /// Proportional to the number of terms of that size with a minimum number per size
    Proportional(usize),
    /// Delegate to a histogram-free `TermDistribution`
    Statistical(Distribution),
}

impl TermSampleDist {
    pub const UNIFORM: Self = Self::Statistical(Distribution::Uniform);
}

impl Display for TermSampleDist {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Proportional(min) => write!(f, "proportional:{min}"),
            Self::Statistical(d) => write!(f, "{d}"),
        }
    }
}

impl FromStr for TermSampleDist {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "proportional" {
            return Ok(Self::Proportional(10));
        }
        if let Some(rest) = s.strip_prefix("proportional:") {
            let min = rest
                .parse::<usize>()
                .map_err(|e| format!("invalid min_per_size in 'proportional:{rest}': {e}"))?;
            return Ok(Self::Proportional(min));
        }
        Distribution::from_str(s)
            .map(Self::Statistical)
            .map_err(|_e| format!(
                "unknown distribution '{s}': expected 'uniform', 'proportional:<min>', or 'normal:<sigma>'"
            ))
    }
}

impl TermSampleDist {
    /// Build a `samples_per_size` map distributing `total_samples` across `[min_size, max_size]`.
    ///
    /// `histogram` maps size -> term count for the root e-class.
    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn samples_per_size<C: Counter>(
        self,
        histogram: &HashMap<usize, C>,
        min_size: usize,
        max_size: usize,
        total_samples: usize,
    ) -> Vec<(usize, u64)> {
        match self {
            Self::Statistical(d) => {
                let sizes = (min_size..=max_size)
                    .filter(|s| histogram.contains_key(s))
                    .collect::<Vec<_>>();
                d.samples_per_size(&sizes, total_samples)
            }
            Self::Proportional(min_per_size) => {
                let total_terms = (min_size..=max_size)
                    .filter_map(|s| histogram.get(&s))
                    .sum::<C>();
                let budget = total_samples.try_into().unwrap();
                (min_size..=max_size)
                    .map(|size| {
                        let n = histogram.get(&size).map_or(0, |count| {
                            let c = count.to_owned();
                            (c * &budget / &total_terms)
                                .try_into()
                                .unwrap_or(u64::MAX)
                                .max(min_per_size.try_into().unwrap())
                        });
                        (size, n)
                    })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(v: &[(usize, u64)]) -> u64 {
        v.iter().map(|(_, n)| n).sum()
    }

    fn sizes(v: &[(usize, u64)]) -> Vec<usize> {
        v.iter().map(|(s, _)| *s).collect()
    }

    fn range(min: usize, max: usize) -> Vec<usize> {
        (min..=max).collect()
    }

    // --- Uniform ---

    #[test]
    fn uniform_exact_total() {
        let result = Distribution::Uniform.samples_per_size(&range(5, 50), 1000);
        assert_eq!(total(&result), 1000);
    }

    #[test]
    fn uniform_divisible_total() {
        // 46 sizes, 460 samples => exactly 10 each
        let result = Distribution::Uniform.samples_per_size(&range(5, 50), 460);
        assert_eq!(total(&result), 460);
        assert!(result.iter().all(|(_, n)| *n == 10));
    }

    #[test]
    fn uniform_covers_all_sizes() {
        let result = Distribution::Uniform.samples_per_size(&range(5, 50), 1000);
        assert_eq!(sizes(&result), (5..=50).collect::<Vec<_>>());
    }

    #[test]
    fn uniform_single_size() {
        let result = Distribution::Uniform.samples_per_size(&[7], 100);
        assert_eq!(result, vec![(7, 100)]);
    }

    #[test]
    fn uniform_remainder_distributed_to_first_sizes() {
        // 3 sizes (1,2,3), 10 samples => base=3, remainder=1 => sizes get [4,3,3]
        let result = Distribution::Uniform.samples_per_size(&range(1, 3), 10);
        assert_eq!(total(&result), 10);
        assert_eq!(result[0].1, 4);
        assert_eq!(result[1].1, 3);
        assert_eq!(result[2].1, 3);
    }

    #[test]
    fn uniform_fewer_samples_than_sizes() {
        // 6 sizes, 2 samples => base=0, remainder=2 => first two get 1, rest get 0
        let result = Distribution::Uniform.samples_per_size(&range(15, 20), 2);
        assert_eq!(total(&result), 2);
        assert_eq!(result[0].1, 1);
        assert_eq!(result[1].1, 1);
        assert!(result[2..].iter().all(|(_, n)| *n == 0));
    }

    #[test]
    fn uniform_non_contiguous_sizes() {
        // Sparse populated sizes get exactly the requested total
        let result = Distribution::Uniform.samples_per_size(&[15, 17, 19], 2);
        assert_eq!(total(&result), 2);
        assert_eq!(sizes(&result), vec![15, 17, 19]);
    }

    // --- Normal ---

    #[test]
    fn normal_exact_total() {
        let result = Distribution::Normal(2.6).samples_per_size(&range(5, 50), 1000);
        assert_eq!(total(&result), 1000);
    }

    #[test]
    fn normal_exact_total_various_sizes() {
        for total_samples in [1, 7, 100, 999, 1000, 1001] {
            let result = Distribution::Normal(2.6).samples_per_size(&range(5, 50), total_samples);
            assert_eq!(
                total(&result),
                total_samples as u64,
                "failed for total_samples={total_samples}"
            );
        }
    }

    #[test]
    fn normal_covers_all_sizes() {
        let result = Distribution::Normal(2.6).samples_per_size(&range(5, 50), 1000);
        assert_eq!(sizes(&result), (5..=50).collect::<Vec<_>>());
    }

    #[test]
    fn normal_center_has_most_samples() {
        // Center is (5+50)/2 = 27 or 28; those should have the highest counts
        let result = Distribution::Normal(2.6).samples_per_size(&range(5, 50), 1000);
        let max_count = result.iter().map(|(_, n)| *n).max().unwrap();
        let center = 27usize;
        let center_count = result.iter().find(|(s, _)| *s == center).unwrap().1;
        assert_eq!(center_count, max_count);
    }

    #[test]
    fn normal_single_size() {
        let result = Distribution::Normal(2.6).samples_per_size(&[10], 50);
        assert_eq!(result, vec![(10, 50)]);
    }
}
