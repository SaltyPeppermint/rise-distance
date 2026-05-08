mod novel;
mod plain;
mod weigher;
// mod zs_min_distance;

use egg::{Id, RecExpr};
use hashbrown::HashSet;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use crate::{Counter, MyAnalysis, MyLanguage, OriginLang, utils};

// TODO: reenable zs_min_distance sampler
// pub use zs_min_distance::ZSDistanceSampler;
pub use novel::NovelSampler;
pub use plain::PlainSampler;
pub use weigher::{CountWeigher, NaiveWeigher, Weigher};

/// Common interface for samplers that draw size-targeted terms from an e-graph.
///
/// `sample_batch` and `sample_batch_root` are provided as default implementations
/// in terms of [`Sampler::sample`] and [`Sampler::possible_size`].
pub trait Sampler<C, L, N>: Sync
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn root(&self) -> Id;

    /// True iff at least `samples + 1` distinct terms of `size` are reachable
    /// from `id` under this sampler's constraints.
    fn possible_size(&self, id: Id, size: usize, samples: u64) -> bool;

    fn min_size(&self, id: Id) -> usize {
        (0..)
            .into_iter()
            .find(|size| self.possible_size(id, *size, 1))
            .unwrap()
    }

    /// Returns a random (but stable) smallest term
    fn smallest(&self, id: Id) -> RecExpr<OriginLang<L>> {
        let size = self.min_size(id);
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        self.sample(id, size, &mut rng)
    }

    /// Precondition: `possible_size(id, size, 0)`.
    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>>;

    fn sample_batch<const PARALLEL: bool>(
        &self,
        id: Id,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
    ) -> HashSet<RecExpr<OriginLang<L>>> {
        if PARALLEL {
            samples_per_size
                .par_iter()
                .filter(|(size, samples)| self.possible_size(id, *size, *samples))
                .flat_map(|(size, samples)| {
                    (0..*samples).into_par_iter().map(|s| {
                        self.sample(
                            id,
                            *size,
                            &mut utils::combined_rng([*size as u64, s, seed[0], seed[1]]),
                        )
                    })
                })
                .collect()
        } else {
            samples_per_size
                .iter()
                .filter(|(size, samples)| self.possible_size(id, *size, *samples))
                .flat_map(|(size, samples)| {
                    (0..*samples).map(|s| {
                        self.sample(
                            id,
                            *size,
                            &mut utils::combined_rng([*size as u64, s, seed[0], seed[1]]),
                        )
                    })
                })
                .collect()
        }
    }

    fn sample_batch_root<const PARALLEL: bool>(
        &self,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
    ) -> HashSet<RecExpr<OriginLang<L>>> {
        self.sample_batch::<PARALLEL>(self.root(), samples_per_size, seed)
    }
}
