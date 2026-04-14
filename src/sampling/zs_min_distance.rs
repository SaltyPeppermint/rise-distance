use hashbrown::HashSet;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use crate::sampling::Sampler;
use crate::tree::OriginTree;
use crate::utils::combined_rng;
use crate::zs::{EditCosts, PreprocessedTree, tree_distance_preprocessed};
use crate::{EClassId, TreeShaped, tree_distance};

/// Greedily only accepts new terms that have a bigger or equal zs distance
pub struct ZSDistanceSampler<E: EditCosts<S::Label>, S: Sampler> {
    inner: S,
    cost_fn: E,
    percentile: f64,
    with_types: bool,
}

impl<E: EditCosts<S::Label>, S: Sampler> ZSDistanceSampler<E, S> {
    #[must_use]
    /// Wrap an existing sampler in one that filters by zs
    ///
    /// # Panics
    ///
    /// Panics if `percentile` is not in (0, 1]
    pub fn new(inner: S, cost_fn: E, percentile: f64, with_types: bool) -> Self {
        assert!(
            percentile > 0.0 && percentile <= 1.0,
            "percentile must be in (0, 1]"
        );
        Self {
            inner,
            cost_fn,
            percentile,
            with_types,
        }
    }

    /// Sample `n` trees of the given `size` from `id`, then compute all pairwise
    /// tree-edit distances.  Returns the distance at the `percentile` position
    /// (e.g. `percentile = 0.2` => 20th percentile of distances).
    ///
    /// # Panics
    ///
    /// Panics if fewer than 2 unique samples are drawn.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn average_distance<const PARALLEL: bool>(
        &self,
        id: EClassId,
        size: usize,
        n: u64,
        seed: [u64; 2],
    ) -> usize {
        // Sample n unique trees of the given size.
        let trees = self.inner.sample_batch::<PARALLEL>(id, &[(size, n)], seed);
        assert!(
            trees.len() >= 2,
            "need at least 2 unique samples, got {}",
            trees.len()
        );

        // Flatten once.
        let flat = trees
            .iter()
            .map(|t| t.flatten(self.with_types))
            .collect::<Vec<_>>();
        // Preprocess once for reuse.
        let preprocessed = flat.iter().map(PreprocessedTree::new).collect::<Vec<_>>();

        // Compute all pairwise distances.
        let mut distances = (0..flat.len())
            .flat_map(|i| ((i + 1)..flat.len()).map(move |j| (i, j)))
            .map(|(i, j)| {
                tree_distance_preprocessed(&preprocessed[i], &preprocessed[j], &self.cost_fn)
            })
            .collect::<Vec<_>>();

        // Sort ascending and take the value at the configured percentile.
        distances.sort_unstable();
        let idx = (self.percentile * distances.len() as f64).round() as usize;
        distances[idx.min(distances.len() - 1)]
    }
}

impl<E: EditCosts<S::Label>, S: Sampler> ZSDistanceSampler<E, S> {
    fn sample_for_size<const PARALLEL: bool>(
        &self,
        id: EClassId,
        size: usize,
        samples: u64,
        seed: [u64; 2],
    ) -> impl Iterator<Item = OriginTree<S::Label>> {
        let mut samples_to_take = samples;
        let mut existing_flat = HashSet::new();
        let mut existing = HashSet::new();

        let mut rejected = 0;
        let cut_off = self.average_distance::<PARALLEL>(id, size, 1000, seed);

        let mut candidate_idx: u64 = 0;
        while samples_to_take > 0 {
            let mut rng = combined_rng([size as u64, candidate_idx, seed[0], seed[1]]);
            let new_candidate = self.sample(id, size, &mut rng);
            candidate_idx += 1;

            if existing.contains(&new_candidate) {
                continue;
            }
            let candidate_flat = new_candidate.flatten(self.with_types);
            if existing_flat
                .iter()
                .any(|e| tree_distance(e, &candidate_flat, &self.cost_fn) < cut_off)
            {
                rejected += 1;
                if rejected % 100 == 0 {
                    eprintln!("REJECTED: {rejected}");
                    eprintln!("SO FAR THIS MANY {}", existing.len());
                }
                continue;
            }
            existing_flat.insert(candidate_flat);
            existing.insert(new_candidate);
            samples_to_take -= 1;
        }
        existing.into_iter()
    }
}

impl<E: EditCosts<S::Label>, S: Sampler> Sampler for ZSDistanceSampler<E, S> {
    type Label = S::Label;

    fn root(&self) -> EClassId {
        self.inner.root()
    }

    fn possible_size(&self, id: EClassId, size: usize, samples: u64) -> bool {
        self.inner.possible_size(id, size, samples)
    }

    fn sample_batch<const PARALLEL: bool>(
        &self,
        id: EClassId,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
    ) -> HashSet<OriginTree<S::Label>> {
        if PARALLEL {
            samples_per_size
                .par_iter()
                .filter(|(size, samples)| self.possible_size(id, *size, *samples))
                .flat_map_iter(|(size, samples)| {
                    self.sample_for_size::<PARALLEL>(id, *size, *samples, seed)
                })
                .collect()
        } else {
            samples_per_size
                .iter()
                .filter(|(size, samples)| self.possible_size(id, *size, *samples))
                .flat_map(|(size, samples)| {
                    self.sample_for_size::<PARALLEL>(id, *size, *samples, seed)
                })
                .collect()
        }
    }

    /// Sample uniformly: each feasible choice gets equal weight.
    fn sample(&self, id: EClassId, size: usize, rng: &mut ChaCha12Rng) -> OriginTree<S::Label> {
        self.inner.sample(id, size, rng)
    }
}
