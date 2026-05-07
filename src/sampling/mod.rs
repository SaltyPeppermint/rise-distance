mod novel;
mod plain;
mod weigher;
// mod zs_min_distance;

use egg::{Id, RecExpr};
use hashbrown::HashSet;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

pub use novel::NovelSampler;
pub use plain::PlainSampler;
pub use weigher::{CountWeigher, NaiveWeigher, Weigher};
// TODO: reenable zs_min_distance sampler
// pub use zs_min_distance::ZSDistanceSampler;

use crate::{Counter, MyAnalysis, MyLanguage, OriginLang};

/// Common interface for samplers that draw size-targeted terms from an e-graph.
///
/// `sample_batch` and `sample_batch_root` are provided as default implementations
/// in terms of [`Sampler::sample`] and [`Sampler::possible_size`].
pub trait Sampler<C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn root(&self) -> Id;

    /// True iff at least `samples + 1` distinct terms of `size` are reachable
    /// from `id` under this sampler's constraints.
    fn possible_size(&self, id: Id, size: usize, samples: u64) -> bool;

    /// Precondition: `possible_size(id, size, 0)`.
    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>>;

    fn sample_batch<const PARALLEL: bool, F>(
        &self,
        id: Id,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
        check: &F,
    ) -> HashSet<RecExpr<OriginLang<L>>>
    where
        F: Fn(&RecExpr<OriginLang<L>>) -> bool + Sync,
        Self: Sync,
    {
        if PARALLEL {
            samples_per_size
                .par_iter()
                .filter(|(size, samples)| self.possible_size(id, *size, *samples))
                .flat_map(|(size, samples)| {
                    (0..*samples).into_par_iter().filter_map(|s| {
                        let candidate = self.sample(
                            id,
                            *size,
                            &mut crate::utils::combined_rng([*size as u64, s, seed[0], seed[1]]),
                        );
                        check(&candidate).then_some(candidate)
                    })
                })
                .collect()
        } else {
            samples_per_size
                .iter()
                .filter(|(size, samples)| self.possible_size(id, *size, *samples))
                .flat_map(|(size, samples)| {
                    (0..*samples).filter_map(|s| {
                        let candidate = self.sample(
                            id,
                            *size,
                            &mut crate::utils::combined_rng([*size as u64, s, seed[0], seed[1]]),
                        );
                        check(&candidate).then_some(candidate)
                    })
                })
                .collect()
        }
    }

    fn sample_batch_root<const PARALLEL: bool, F>(
        &self,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
        check: &F,
    ) -> HashSet<RecExpr<OriginLang<L>>>
    where
        F: Fn(&RecExpr<OriginLang<L>>) -> bool + Sync,
        Self: Sync,
    {
        self.sample_batch::<PARALLEL, _>(self.root(), samples_per_size, seed, check)
    }
}
