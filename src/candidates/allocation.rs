use std::fmt::Display;
use std::str::FromStr;

use hashbrown::HashMap;
use serde::Serialize;

use crate::Counter;

#[derive(Debug, Clone, Copy, Serialize)]
pub enum SizeAllocation {
    /// Fill the candidate budget greedily from the smallest size upward: take as
    /// many terms as each size has before moving to the next bigger one, until
    /// the goal is reached (or every size is exhausted).
    Greedy,
    /// Uniform across term sizes
    Uniform,
}

impl Display for SizeAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Greedy => write!(f, "greedy"),
            Self::Uniform => write!(f, "uniform"),
        }
    }
}

impl FromStr for SizeAllocation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "greedy" {
            return Ok(Self::Greedy);
        }
        if s == "uniform" {
            return Ok(Self::Uniform);
        }
        Err(format!(
            "unknown size allocation '{s}': expected 'uniform' or 'greedy'"
        ))
    }
}

/// Distribute `total_candidates` uniformly across `sizes`.
///
/// Any remainder is assigned to the first sizes, so the returned counts always
/// add up to `total_candidates` when at least one size is supplied.
///
/// # Panics
///
/// On platforms where `usize` is wider than `u64`, panics if the per-size
/// candidate count cannot be represented as a `u64`.
#[must_use]
pub fn uniform_candidate_allocation(sizes: &[usize], total_candidates: usize) -> Vec<(usize, u64)> {
    if sizes.is_empty() {
        return vec![];
    }
    let size_count = sizes.len();
    let base = u64::try_from(total_candidates / size_count).unwrap();
    let remainder = total_candidates % size_count;
    sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| (size, base + u64::from(i < remainder)))
        .collect()
}

impl SizeAllocation {
    /// Allocate candidates distributing `total_candidates` across `[min_size, max_size]`.
    ///
    /// `histogram` maps size -> term count for the root e-class.
    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn allocate<C: Counter>(
        self,
        histogram: &HashMap<usize, C>,
        min_size: usize,
        max_size: usize,
        total_candidates: usize,
    ) -> Vec<(usize, u64)> {
        match self {
            Self::Uniform => {
                let sizes = (min_size..=max_size)
                    .filter(|s| histogram.contains_key(s))
                    .collect::<Vec<_>>();
                uniform_candidate_allocation(&sizes, total_candidates)
            }
            Self::Greedy => {
                let mut remaining = u64::try_from(total_candidates).unwrap();
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
        let result = uniform_candidate_allocation(&range(5, 50), 1000);
        assert_eq!(total(&result), 1000);
    }

    #[test]
    fn uniform_divisible_total() {
        let result = uniform_candidate_allocation(&range(5, 50), 460);
        assert_eq!(total(&result), 460);
        assert!(result.iter().all(|(_, n)| *n == 10));
    }

    #[test]
    fn uniform_covers_all_sizes() {
        let result = uniform_candidate_allocation(&range(5, 50), 1000);
        assert_eq!(sizes(&result), range(5, 50));
    }

    #[test]
    fn uniform_single_size() {
        assert_eq!(uniform_candidate_allocation(&[7], 100), vec![(7, 100)]);
    }

    #[test]
    fn uniform_remainder_goes_to_first_sizes() {
        assert_eq!(
            uniform_candidate_allocation(&range(1, 3), 10),
            vec![(1, 4), (2, 3), (3, 3)]
        );
    }

    #[test]
    fn uniform_fewer_candidates_than_sizes() {
        assert_eq!(
            uniform_candidate_allocation(&range(15, 20), 2),
            vec![(15, 1), (16, 1), (17, 0), (18, 0), (19, 0), (20, 0)]
        );
    }

    #[test]
    fn uniform_handles_no_sizes() {
        assert!(uniform_candidate_allocation(&[], 100).is_empty());

        let histogram = HashMap::<usize, BigUint>::new();
        assert!(
            SizeAllocation::Uniform
                .allocate(&histogram, 1, 3, 100)
                .is_empty()
        );
    }

    #[test]
    fn uniform_skips_absent_histogram_sizes() {
        let histogram = hist(&[(15, 1), (17, 1), (19, 1)]);
        let result = SizeAllocation::Uniform.allocate(&histogram, 15, 19, 2);
        assert_eq!(result, vec![(15, 1), (17, 1), (19, 0)]);
    }

    #[test]
    fn unknown_allocation_is_rejected() {
        assert!("normal".parse::<SizeAllocation>().is_err());
        assert!("normal:2.6".parse::<SizeAllocation>().is_err());
    }

    #[test]
    fn greedy_fills_from_smallest_size() {
        let histogram = hist(&[(1, 5), (2, 5), (3, 5)]);
        let result = SizeAllocation::Greedy.allocate(&histogram, 1, 3, 8);
        assert_eq!(result, vec![(1, 5), (2, 3), (3, 0)]);
        assert_eq!(total(&result), 8);
    }

    #[test]
    fn greedy_stops_once_budget_is_met() {
        let histogram = hist(&[(1, 100), (2, 100)]);
        let result = SizeAllocation::Greedy.allocate(&histogram, 1, 2, 30);
        assert_eq!(result, vec![(1, 30), (2, 0)]);
    }

    #[test]
    fn greedy_undersupply_takes_all_available() {
        let histogram = hist(&[(1, 2), (2, 3)]);
        let result = SizeAllocation::Greedy.allocate(&histogram, 1, 2, 100);
        assert_eq!(result, vec![(1, 2), (2, 3)]);
        assert_eq!(total(&result), 5);
    }

    #[test]
    fn greedy_skips_absent_sizes() {
        let histogram = hist(&[(1, 3), (3, 10)]);
        let result = SizeAllocation::Greedy.allocate(&histogram, 1, 3, 8);
        assert_eq!(result, vec![(1, 3), (2, 0), (3, 5)]);
        assert_eq!(total(&result), 8);
    }
}
