use std::fmt::Display;
use std::str::FromStr;

use hashbrown::HashMap;
use serde::Serialize;

use crate::Counter;

#[derive(Serialize, serde::Deserialize, Debug, Clone, Copy, clap::ValueEnum, strum::Display)]
#[strum(serialize_all = "kebab-case")]
pub enum SampleStrategy {
    Independent,
    Naive,
    Balanced,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum Distribution {
    /// Proportional to the number of terms of that size with a minimum number per size
    Proportional(usize),
    /// Fill the sample budget greedily from the smallest size upward: take as
    /// many terms as each size has before moving to the next bigger one, until
    /// the goal is reached (or every size is exhausted).
    Greedy,
    /// Uniform across term sizes
    Uniform,
}

impl Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Proportional(min) => write!(f, "proportional:{min}"),
            Self::Greedy => write!(f, "greedy"),
            Self::Uniform => write!(f, "uniform"),
        }
    }
}

impl FromStr for Distribution {
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
        if s == "uniform" {
            return Ok(Self::Uniform);
        }
        Err(format!(
            "unknown distribution '{s}': expected 'uniform', 'greedy', or 'proportional:<min>'"
        ))
    }
}

/// Distribute `total_samples` uniformly across `sizes`.
///
/// Any remainder is assigned to the first sizes, so the returned counts always
/// add up to `total_samples` when at least one size is supplied.
///
/// # Panics
///
/// On platforms where `usize` is wider than `u64`, panics if the per-size
/// sample count cannot be represented as a `u64`.
#[must_use]
pub fn uniform_samples_per_size(sizes: &[usize], total_samples: usize) -> Vec<(usize, u64)> {
    if sizes.is_empty() {
        return vec![];
    }
    let num_sizes = sizes.len();
    let base = u64::try_from(total_samples / num_sizes).unwrap();
    let remainder = total_samples % num_sizes;
    sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| (size, base + u64::from(i < remainder)))
        .collect()
}

impl Distribution {
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
            Self::Uniform => {
                let sizes = (min_size..=max_size)
                    .filter(|s| histogram.contains_key(s))
                    .collect::<Vec<_>>();
                uniform_samples_per_size(&sizes, total_samples)
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

    fn hist(pairs: &[(usize, u64)]) -> HashMap<usize, BigUint> {
        pairs.iter().map(|&(s, c)| (s, BigUint::from(c))).collect()
    }

    #[test]
    fn uniform_exact_total() {
        let result = uniform_samples_per_size(&range(5, 50), 1000);
        assert_eq!(total(&result), 1000);
    }

    #[test]
    fn uniform_divisible_total() {
        let result = uniform_samples_per_size(&range(5, 50), 460);
        assert_eq!(total(&result), 460);
        assert!(result.iter().all(|(_, n)| *n == 10));
    }

    #[test]
    fn uniform_covers_all_sizes() {
        let result = uniform_samples_per_size(&range(5, 50), 1000);
        assert_eq!(sizes(&result), range(5, 50));
    }

    #[test]
    fn uniform_single_size() {
        assert_eq!(uniform_samples_per_size(&[7], 100), vec![(7, 100)]);
    }

    #[test]
    fn uniform_remainder_goes_to_first_sizes() {
        assert_eq!(
            uniform_samples_per_size(&range(1, 3), 10),
            vec![(1, 4), (2, 3), (3, 3)]
        );
    }

    #[test]
    fn uniform_fewer_samples_than_sizes() {
        assert_eq!(
            uniform_samples_per_size(&range(15, 20), 2),
            vec![(15, 1), (16, 1), (17, 0), (18, 0), (19, 0), (20, 0)]
        );
    }

    #[test]
    fn uniform_handles_no_sizes() {
        assert!(uniform_samples_per_size(&[], 100).is_empty());

        let histogram = HashMap::<usize, BigUint>::new();
        assert!(
            Distribution::Uniform
                .samples_per_size(&histogram, 1, 3, 100)
                .is_empty()
        );
    }

    #[test]
    fn uniform_skips_absent_histogram_sizes() {
        let histogram = hist(&[(15, 1), (17, 1), (19, 1)]);
        let result = Distribution::Uniform.samples_per_size(&histogram, 15, 19, 2);
        assert_eq!(result, vec![(15, 1), (17, 1), (19, 0)]);
    }

    #[test]
    fn normal_distribution_is_rejected() {
        assert!("normal".parse::<Distribution>().is_err());
        assert!("normal:2.6".parse::<Distribution>().is_err());
    }

    #[test]
    fn greedy_fills_from_smallest_size() {
        let histogram = hist(&[(1, 5), (2, 5), (3, 5)]);
        let result = Distribution::Greedy.samples_per_size(&histogram, 1, 3, 8);
        assert_eq!(result, vec![(1, 5), (2, 3), (3, 0)]);
        assert_eq!(total(&result), 8);
    }

    #[test]
    fn greedy_stops_once_budget_is_met() {
        let histogram = hist(&[(1, 100), (2, 100)]);
        let result = Distribution::Greedy.samples_per_size(&histogram, 1, 2, 30);
        assert_eq!(result, vec![(1, 30), (2, 0)]);
    }

    #[test]
    fn greedy_undersupply_takes_all_available() {
        let histogram = hist(&[(1, 2), (2, 3)]);
        let result = Distribution::Greedy.samples_per_size(&histogram, 1, 2, 100);
        assert_eq!(result, vec![(1, 2), (2, 3)]);
        assert_eq!(total(&result), 5);
    }

    #[test]
    fn greedy_skips_absent_sizes() {
        let histogram = hist(&[(1, 3), (3, 10)]);
        let result = Distribution::Greedy.samples_per_size(&histogram, 1, 3, 8);
        assert_eq!(result, vec![(1, 3), (2, 0), (3, 5)]);
        assert_eq!(total(&result), 8);
    }
}
