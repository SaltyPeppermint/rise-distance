mod novel;
mod plain;
mod weigher;
// mod zs_min_distance;

use egg::{Id, RecExpr};
use hashbrown::HashSet;
use num::ToPrimitive;
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

    /// All sizes for which `id` has at least one extractable term under this
    /// sampler's constraints. Iteration order is unspecified.
    fn term_sizes(&self, id: Id) -> Vec<usize>;

    /// Enumerate every distinct term of exactly `size` reachable from `id`
    /// under this sampler's constraints. Returned order is unspecified.
    fn enumerate_size(&self, id: Id, size: usize) -> Vec<RecExpr<OriginLang<L>>>;

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

    /// Returns the `n` distinct terms of smallest total size reachable from
    /// `id`, or `None` if fewer than `n` distinct terms exist.
    ///
    /// The result is monotone: for `k <= n`, `n_smallest(id, k)` is a subset
    /// of `n_smallest(id, n)`. Within a single size, terms are ordered by
    /// their structural `Ord` impl (i.e., lex order on the underlying node
    /// sequence), so repeated calls return the same set.
    fn n_smallest(&self, id: Id, n: u64) -> Option<HashSet<RecExpr<OriginLang<L>>>> {
        let mut sizes = self.term_sizes(id);
        sizes.sort_unstable();

        let mut result = HashSet::new();

        for size in sizes {
            if (result.len() as u64) >= n {
                break;
            }
            let need = n.to_usize().unwrap() - result.len();
            let mut terms = self.enumerate_size(id, size);
            // Total order on terms gives a stable, monotone selection: taking
            // any prefix of `n` always yields a subset of taking `n+k`.
            terms.sort_unstable();
            terms.truncate(need);
            for t in terms {
                result.insert(t);
            }
        }

        ((result.len() as u64) >= n).then_some(result)
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
