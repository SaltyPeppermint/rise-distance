mod count;
mod naive;

use hashbrown::{HashMap, HashSet};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

pub use count::CountSampler;
pub use naive::NaiveSampler;

use crate::{EClassId, Label, tree::OriginTree};

pub trait Sampler<L: Label>: Sync + Send {
    #[must_use]
    fn root(&self) -> EClassId;

    #[must_use]
    fn possible_size(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
    ) -> impl Iterator<Item = usize> + Send;

    #[must_use]
    fn sample_many_root(
        &self,
        size: usize,
        samples: u64,
        seed: u64,
    ) -> impl ParallelIterator<Item = OriginTree<L>> {
        self.sample_many(self.root(), size, samples, seed)
    }

    #[must_use]
    fn sample_many(
        &self,
        id: EClassId,
        size: usize,
        samples: u64,
        seed: u64,
    ) -> impl ParallelIterator<Item = OriginTree<L>> {
        (0..samples).into_par_iter().map(move |sample| {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            rng.set_stream(sample);
            self.sample(id, size, &mut rng)
        })
    }

    /// Sample unique terms across a range of sizes from root.
    ///
    /// See `sample_unique` for more info
    fn sample_batch_root(
        &self,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<OriginTree<L>> {
        self.sample_batch(self.root(), min_size, max_size, samples_per_size)
    }

    /// Sample unique terms across a range of sizes.
    ///
    /// For each size in `[min_size, max_size]` that the root e-class actually has
    /// terms for, samples `samples_fn` terms and deduplicates them.
    ///
    /// Only sizes present in the root's histogram are sampled. The root e-class
    /// may have gaps in its reachable sizes (e.g. terms only at sizes 5, 7, 9),
    /// and calling `sample` with a size that has no terms would cause all node
    /// weights to be zero, panicking with `AllWeightsZero`.
    fn sample_batch(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<OriginTree<L>> {
        self.possible_size(id, min_size, max_size)
            .par_bridge()
            .flat_map(|size| self.sample_many(id, size, samples_per_size[&size], size as u64))
            .collect()
    }

    #[must_use]
    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> OriginTree<L>;
}
