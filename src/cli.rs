use std::fmt::Display;
use std::str::FromStr;

use hashbrown::HashMap;
use num::{BigUint, ToPrimitive};

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

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum SampleStrategy {
    /// Use the `sample_with_overlap`
    Overlap,
    /// Sample fully random
    Random,
    /// Enumerate all terms up to the limit
    Enumerate,
}

impl Display for SampleStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overlap => write!(f, "overlap"),
            Self::Random => write!(f, "random"),
            Self::Enumerate => write!(f, "enumerate"),
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
