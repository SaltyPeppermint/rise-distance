mod common;
mod count;
mod naive;
mod zs_min_distance;

use hashbrown::HashSet;
use rand::prelude::*;

pub use count::CountSampler;
pub use naive::NaiveSampler;
pub use zs_min_distance::ZSDistanceSampler;

use crate::{EClassId, Label, tree::OriginTree};

pub trait Sampler: Sync + Send {
    type Label: Label;

    #[must_use]
    fn root(&self) -> EClassId;

    #[must_use]
    fn possible_size(&self, id: EClassId, size: usize, samples: u64) -> bool;

    /// Sample unique terms across a range of sizes from root.
    ///
    /// See `sample_unique` for more info
    fn sample_batch_root(
        &self,
        samples_per_size: &[(usize, u64)],
    ) -> HashSet<OriginTree<Self::Label>> {
        self.sample_batch(self.root(), samples_per_size)
    }

    /// Sample unique terms across a range of sizes.
    ///
    ///
    /// Only sizes present in the root's histogram are sampled. The root e-class
    /// may have gaps in its reachable sizes (e.g. terms only at sizes 5, 7, 9),
    /// and calling `sample` with a size that has no terms would cause all node
    /// weights to be zero, panicking with `AllWeightsZero`.
    fn sample_batch(
        &self,
        id: EClassId,
        samples_per_size: &[(usize, u64)],
    ) -> HashSet<OriginTree<Self::Label>>;

    #[must_use]
    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> OriginTree<Self::Label>;
}
