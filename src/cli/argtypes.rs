use std::fmt::Display;
use std::str::FromStr;

use hashbrown::HashMap;
use num::ToPrimitive;
use serde::Serialize;

use crate::count::Counter;

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
    /// Distribute `total_samples` across `[min_size, max_size]`, returning a size → count map.
    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn samples_per_size(
        self,
        min_size: usize,
        max_size: usize,
        total_samples: usize,
    ) -> HashMap<usize, u64> {
        match self {
            Self::Uniform => {
                let num_sizes = (max_size - min_size).max(1);
                let s = (total_samples / num_sizes).max(1).try_into().unwrap();
                (min_size..=max_size).map(|size| (size, s)).collect()
            }
            Self::Normal(sigma) => {
                #[expect(clippy::cast_precision_loss)]
                let normal_center = (min_size + max_size) as f64 / 2.0;
                let weights = (min_size..=max_size)
                    .map(|s| {
                        #[expect(clippy::cast_precision_loss)]
                        let z = (s as f64 - normal_center) / sigma;
                        (s, (-0.5 * z * z).exp())
                    })
                    .collect::<HashMap<_, _>>();
                let total_weight: f64 = weights.values().sum();
                (min_size..=max_size)
                    .map(|size| {
                        let w = *weights.get(&size).unwrap_or(&0.0);
                        #[expect(clippy::cast_precision_loss)]
                        let n = (w / total_weight * total_samples as f64)
                            .round()
                            .to_u64()
                            .unwrap();
                        (size, n)
                    })
                    .collect()
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
    ) -> HashMap<usize, u64> {
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
