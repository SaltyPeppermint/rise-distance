use std::fmt::Display;
use std::fs::File;
use std::path::PathBuf;
use std::str::FromStr;

use egg::RecExpr;
use hashbrown::HashMap;
use num::ToPrimitive;
use serde::Serialize;

use crate::count::Counter;
use crate::egg::math::Math;

/// Either a single seed s-expression or a path to a JSON file with objects containing `size` and `term` fields.
///
/// Pass as `--seed '(d x ...)'` or `--seed-json path/to/file.json`.
#[derive(Debug, Clone)]
pub enum SeedInput {
    Single { term: String, max_size: usize },
    JSON(PathBuf),
}

#[derive(Serialize, Debug, Clone, Copy, clap::ValueEnum, strum::Display)]
#[strum(serialize_all = "kebab-case")]
pub enum DistanceMetric {
    ZhangShasha,
    Structural,
}

#[derive(Serialize, Debug, Clone, Copy, clap::ValueEnum, strum::Display)]
#[strum(serialize_all = "kebab-case")]
pub enum SampleStrategy {
    Naive,
    CountBased,
    ZSDiverseNaive,
    ZSDiverseCountBased,
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
        if s == "uniform" {
            return Ok(Self::Uniform);
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
            "unknown distribution '{s}': expected 'uniform' or 'normal:<sigma>'"
        ))
    }
}

impl Distribution {
    /// Distribute `total_samples` across `[min_size, max_size]`, returning a size => count map.
    #[expect(
        clippy::missing_panics_doc,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn samples_per_size(
        self,
        min_size: usize,
        max_size: usize,
        total_samples: usize,
    ) -> Vec<(usize, u64)> {
        match self {
            Self::Uniform => {
                let num_sizes = (max_size - min_size + 1).max(1);
                let base = (total_samples / num_sizes).max(1) as u64;
                let remainder = total_samples % num_sizes;
                (min_size..=max_size)
                    .enumerate()
                    .map(|(i, size)| (size, base + u64::from(i < remainder)))
                    .collect()
            }
            Self::Normal(sigma) => {
                let normal_center = (min_size + max_size) as f64 / 2.0;
                let weights = (min_size..=max_size).map(|s| {
                    let z = (s as f64 - normal_center) / sigma;
                    (-0.5 * z * z).exp()
                });
                let total_weight = weights.clone().sum::<f64>();
                // Largest remainder method: compute exact quotas, floor them, then
                // distribute the remaining counts to the sizes with the largest remainders.
                let quotas = weights.map(|w| w / total_weight * total_samples as f64);
                let floors = quotas.clone().map(|q| q as u64).collect::<Vec<_>>();
                let allocated = floors.iter().sum::<u64>();
                let remainder = (total_samples as u64) - allocated;
                let mut remainders = quotas
                    .enumerate()
                    .map(|(i, q)| (i, q - floors[i] as f64))
                    .collect::<Vec<_>>();
                remainders
                    .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                let mut counts = floors;
                #[allow(clippy::cast_possible_truncation)]
                for (i, _) in remainders.iter().take(remainder as usize) {
                    counts[*i] += 1;
                }
                (min_size..=max_size).zip(counts).collect()
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
            Self::Statistical(d) => d.samples_per_size(min_size, max_size, total_samples),
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

/// Parse a `SeedInput` into a list of `(seed_string, parsed_expr, max_size)` triples.
///
/// # Panics
///
/// Panics on malformed seed expressions or unreadable JSON files.
#[must_use]
pub fn parse_seeds(input: SeedInput) -> Vec<(String, RecExpr<Math>, usize)> {
    match input {
        SeedInput::Single { term, max_size } => {
            let expr = term
                .parse::<RecExpr<Math>>()
                .unwrap_or_else(|e| panic!("Failed to parse seed: {e}"));
            vec![(term, expr, max_size)]
        }
        SeedInput::JSON(path) => {
            let reader = File::open(&path)
                .unwrap_or_else(|e| panic!("Failed to open JSON {}: {e}", path.display()));
            serde_json::from_reader::<_, serde_json::Value>(reader)
                .unwrap()
                .as_array()
                .unwrap_or_else(|| panic!("Expected top-level JSON array in {}", path.display()))
                .iter()
                .flat_map(|group| {
                    let pair = group.as_array().expect("Expected [size, {{terms}}]");
                    let max_size = 2 * pair[0]
                        .as_u64()
                        .expect("Expected size as u64 in")
                        .to_usize()
                        .unwrap();

                    pair[1]
                        .as_object()
                        .expect("Expected term object as second element")
                        .keys()
                        .map(move |term| {
                            let expr = term
                                .parse::<RecExpr<Math>>()
                                .unwrap_or_else(|e| panic!("Failed to parse seed '{term}': {e}"));
                            (term.clone(), expr, max_size)
                        })
                })
                .collect()
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

    // --- Uniform ---

    #[test]
    fn uniform_exact_total() {
        let result = Distribution::Uniform.samples_per_size(5, 50, 1000);
        assert_eq!(total(&result), 1000);
    }

    #[test]
    fn uniform_divisible_total() {
        // 46 sizes, 460 samples => exactly 10 each
        let result = Distribution::Uniform.samples_per_size(5, 50, 460);
        assert_eq!(total(&result), 460);
        assert!(result.iter().all(|(_, n)| *n == 10));
    }

    #[test]
    fn uniform_covers_all_sizes() {
        let result = Distribution::Uniform.samples_per_size(5, 50, 1000);
        assert_eq!(sizes(&result), (5..=50).collect::<Vec<_>>());
    }

    #[test]
    fn uniform_single_size() {
        let result = Distribution::Uniform.samples_per_size(7, 7, 100);
        assert_eq!(result, vec![(7, 100)]);
    }

    #[test]
    fn uniform_remainder_distributed_to_first_sizes() {
        // 3 sizes (1,2,3), 10 samples => base=3, remainder=1 => sizes get [4,3,3]
        let result = Distribution::Uniform.samples_per_size(1, 3, 10);
        assert_eq!(total(&result), 10);
        assert_eq!(result[0].1, 4);
        assert_eq!(result[1].1, 3);
        assert_eq!(result[2].1, 3);
    }

    // --- Normal ---

    #[test]
    fn normal_exact_total() {
        let result = Distribution::Normal(2.6).samples_per_size(5, 50, 1000);
        assert_eq!(total(&result), 1000);
    }

    #[test]
    fn normal_exact_total_various_sizes() {
        for total_samples in [1, 7, 100, 999, 1000, 1001] {
            let result = Distribution::Normal(2.6).samples_per_size(5, 50, total_samples);
            assert_eq!(
                total(&result),
                total_samples as u64,
                "failed for total_samples={total_samples}"
            );
        }
    }

    #[test]
    fn normal_covers_all_sizes() {
        let result = Distribution::Normal(2.6).samples_per_size(5, 50, 1000);
        assert_eq!(sizes(&result), (5..=50).collect::<Vec<_>>());
    }

    #[test]
    fn normal_center_has_most_samples() {
        // Center is (5+50)/2 = 27 or 28; those should have the highest counts
        let result = Distribution::Normal(2.6).samples_per_size(5, 50, 1000);
        let max_count = result.iter().map(|(_, n)| *n).max().unwrap();
        let center = 27usize;
        let center_count = result.iter().find(|(s, _)| *s == center).unwrap().1;
        assert_eq!(center_count, max_count);
    }

    #[test]
    fn normal_single_size() {
        let result = Distribution::Normal(2.6).samples_per_size(10, 10, 50);
        assert_eq!(result, vec![(10, 50)]);
    }
}
