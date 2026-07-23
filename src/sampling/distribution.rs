use std::fmt::Display;
use std::str::FromStr;

use hashbrown::HashMap;
use serde::Serialize;

use crate::Counter;

#[derive(Serialize, serde::Deserialize, Debug, Clone, Copy, clap::ValueEnum, strum::Display)]
#[strum(serialize_all = "kebab-case")]
pub enum SampleStrategy {
    Naive,
    Count,
    Balanced,
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
            _ if let Some(rest) = s.strip_prefix("normal:") => rest
                .parse::<f64>()
                .map(Self::Normal)
                .map_err(|e| format!("invalid sigma in 'normal:{rest}': {e}")),
            _ => Err(format!(
                "unknown distribution '{s}': expected 'uniform' or 'normal:<sigma>'"
            )),
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
    /// Fill the sample budget greedily from the smallest size upward: take as
    /// many terms as each size has before moving to the next bigger one, until
    /// the goal is reached (or every size is exhausted).
    Greedy,
    /// Delegate to a histogram-free `TermDistribution`
    Statistical(Distribution),
}

impl TermSampleDist {
    pub const UNIFORM: Self = Self::Statistical(Distribution::Uniform);
    pub const GREEDY: Self = Self::Greedy;
}

impl Display for TermSampleDist {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Proportional(min) => write!(f, "proportional:{min}"),
            Self::Greedy => write!(f, "greedy"),
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
        if s == "greedy" {
            return Ok(Self::Greedy);
        }
        Distribution::from_str(s)
            .map(Self::Statistical)
            .map_err(|_e| format!(
                "unknown distribution '{s}': expected 'uniform', 'greedy', 'proportional:<min>', or 'normal:<sigma>'"
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
            Self::Greedy => {
                let mut remaining = u64::try_from(total_samples).unwrap();
                (min_size..=max_size)
                    .map(|size| {
                        let available = histogram
                            .get(&size)
                            .map_or(0, |count| count.to_u64().unwrap_or(u64::MAX));
                        let take = remaining.min(available);
                        remaining -= take;
                        (size, take)
                    })
                    .collect()
            }
            Self::Proportional(min_per_size) => {
                let total_terms = (min_size..=max_size)
                    .filter_map(|s| histogram.get(&s))
                    .sum::<C>();
                let budget = C::from_usize(total_samples).unwrap();
                let floor = u64::try_from(min_per_size).unwrap();
                (min_size..=max_size)
                    .map(|size| {
                        let n = histogram.get(&size).map_or(0, |count| {
                            (count.clone() * &budget / &total_terms)
                                .to_u64()
                                .unwrap_or(u64::MAX)
                                .max(floor)
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
    use num::BigUint;

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

    // --- Greedy ---

    fn hist(pairs: &[(usize, u64)]) -> HashMap<usize, BigUint> {
        pairs.iter().map(|&(s, c)| (s, BigUint::from(c))).collect()
    }

    #[test]
    fn greedy_fills_from_smallest_size() {
        // sizes 1,2,3 hold 5,5,5 terms; budget 8 => take 5 from size 1, 3 from
        // size 2, nothing left for size 3.
        let h = hist(&[(1, 5), (2, 5), (3, 5)]);
        let result = TermSampleDist::Greedy.samples_per_size(&h, 1, 3, 8);
        assert_eq!(result, vec![(1, 5), (2, 3), (3, 0)]);
        assert_eq!(total(&result), 8);
    }

    #[test]
    fn greedy_stops_once_budget_is_met() {
        // The first size alone covers the whole budget.
        let h = hist(&[(1, 100), (2, 100)]);
        let result = TermSampleDist::Greedy.samples_per_size(&h, 1, 2, 30);
        assert_eq!(result, vec![(1, 30), (2, 0)]);
    }

    #[test]
    fn greedy_undersupply_takes_all_available() {
        // Total available (2+3) is below the budget => take everything.
        let h = hist(&[(1, 2), (2, 3)]);
        let result = TermSampleDist::Greedy.samples_per_size(&h, 1, 2, 100);
        assert_eq!(result, vec![(1, 2), (2, 3)]);
        assert_eq!(total(&result), 5);
    }

    #[test]
    fn greedy_skips_absent_sizes() {
        // Size 2 is not in the histogram; it contributes 0 and is passed over.
        let h = hist(&[(1, 3), (3, 10)]);
        let result = TermSampleDist::Greedy.samples_per_size(&h, 1, 3, 8);
        assert_eq!(result, vec![(1, 3), (2, 0), (3, 5)]);
        assert_eq!(total(&result), 8);
    }
}
