mod common;
mod count;
mod naive;
mod zs_min_distance;

use egg::{Id, Language, RecExpr};
use hashbrown::HashSet;

pub use count::CountSampler;
pub use naive::NaiveSampler;
use rand_chacha::ChaCha12Rng;
pub use zs_min_distance::ZSDistanceSampler;

use crate::origin::OriginNode;

pub trait Sampler: Sync + Send {
    type Lang: Language + Send + Sync;

    #[must_use]
    fn root(&self) -> Id;

    #[must_use]
    fn possible_size(&self, id: Id, size: usize, samples: u64) -> bool;

    /// Sample unique terms across a range of sizes from root.
    ///
    /// See `sample_unique` for more info
    fn sample_batch_root<const PARALLEL: bool, F>(
        &self,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
        check: &F,
    ) -> HashSet<RecExpr<OriginNode<Self::Lang>>>
    where
        F: Fn(&RecExpr<OriginNode<Self::Lang>>) -> bool + Sync,
    {
        self.sample_batch::<PARALLEL, _>(self.root(), samples_per_size, seed, check)
    }

    /// Sample unique terms across a range of sizes.
    ///
    ///
    /// Only sizes present in the root's histogram are sampled. The root e-class
    /// may have gaps in its reachable sizes (e.g. terms only at sizes 5, 7, 9),
    /// and calling `sample` with a size that has no terms would cause all node
    /// weights to be zero, panicking with `AllWeightsZero`.
    fn sample_batch<const PARALLEL: bool, F>(
        &self,
        id: Id,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
        check: &F,
    ) -> HashSet<RecExpr<OriginNode<Self::Lang>>>
    where
        F: Fn(&RecExpr<OriginNode<Self::Lang>>) -> bool + Sync;

    #[must_use]
    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng)
    -> RecExpr<OriginNode<Self::Lang>>;
}
