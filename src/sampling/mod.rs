pub mod count;
pub mod naive;

use hashbrown::{HashMap, HashSet};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use crate::{EClassId, Label, TreeNode};

pub trait Sampler<L: Label>: Sync + Send {
    #[must_use]
    fn root(&self) -> EClassId;

    #[must_use]
    fn sample_root(
        &self,
        size: usize,
        samples: u64,
        seed: u64,
    ) -> impl ParallelIterator<Item = TreeNode<L>> {
        self.sample_class(self.root(), size, samples, seed)
    }

    #[must_use]
    fn sample_class(
        &self,
        id: EClassId,
        size: usize,
        samples: u64,
        seed: u64,
    ) -> impl ParallelIterator<Item = TreeNode<L>> {
        (0..samples).into_par_iter().map(move |sample| {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            rng.set_stream(sample);
            self.sample(id, size, &mut rng)
        })
    }

    /// Sample unique terms across a range of sizes from root.
    ///
    /// See `sample_unique` for more info
    fn sample_unique_root(
        &self,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<TreeNode<L>> {
        self.sample_unique(self.root(), min_size, max_size, samples_per_size)
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
    #[must_use]
    fn sample_unique(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<TreeNode<L>>;

    #[must_use]
    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> TreeNode<L>;
}
