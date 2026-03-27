mod count;
mod naive;
mod zs_min_distance;

use hashbrown::{HashMap, HashSet};
use rand::prelude::*;

pub use count::CountSampler;
pub use naive::NaiveSampler;

use crate::{EClassId, Label, tree::OriginTree};

pub trait Sampler: Sync + Send {
    type Label: Label;

    #[must_use]
    fn root(&self) -> EClassId;

    #[must_use]
    fn possible_size(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
    ) -> impl Iterator<Item = usize> + Send;

    /// Sample unique terms across a range of sizes from root.
    ///
    /// See `sample_unique` for more info
    fn sample_batch_root(
        &self,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<OriginTree<Self::Label>> {
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
    ) -> HashSet<OriginTree<Self::Label>>;

    #[must_use]
    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> OriginTree<Self::Label>;
}
