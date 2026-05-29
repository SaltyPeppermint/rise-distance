mod novel;
mod plain;

use egg::{Id, RecExpr};
use foldhash::fast::FixedState;
use hashbrown::{HashMap, HashSet};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use crate::sampling::Counter;
use crate::{MyAnalysis, MyLanguage, OriginLang, utils};

pub use novel::NovelSampler;
pub use plain::PlainSampler;

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

    /// Canonicalize `id` in the underlying e-graph.
    fn find(&self, id: Id) -> Id;

    /// Histogram of extractable term sizes for the canonical class of `id`,
    /// or `None` if the class has no entries under this sampler's constraints.
    fn size_histogram(&self, id: Id) -> Option<&HashMap<usize, C>>;

    /// True iff at least `samples + 1` distinct terms of `size` are reachable
    /// from `id` under this sampler's constraints.
    fn possible_size(&self, id: Id, size: usize, samples: u64) -> bool {
        let canon_id = self.find(id);
        let Some(count) = self.size_histogram(canon_id).and_then(|h| h.get(&size)) else {
            return false;
        };
        samples.try_into().is_ok_and(|s: C| count > &s)
    }

    /// All sizes for which `id` has at least one extractable term under this
    /// sampler's constraints. Iteration order is unspecified.
    fn term_sizes(&self, id: Id) -> Vec<usize> {
        let canon_id = self.find(id);
        self.size_histogram(canon_id)
            .map(|h| h.keys().copied().collect())
            .unwrap_or_default()
    }

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

    /// Precondition: `possible_size(id, size, 0)`.
    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>>;

    /// Sample exactly n number of terms for each size from an eclass
    ///
    /// # Errors
    ///
    /// Errors if even with oversampling the sampler does not find enough terms
    fn sample_batch(
        &self,
        id: Id,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let iters = samples_per_size
            .par_iter()
            .map(|&(size, samples)| self.sample_size(id, size, samples, seed))
            .collect::<Option<Vec<_>>>()?;
        Some(iters.into_iter().flatten().collect())
    }

    /// Draw `samples` distinct terms of exactly `size` from `id`, doubling the
    /// oversample factor up to 2^5 until enough unique terms are found.
    ///
    /// # Errors
    ///
    /// Errors if even the largest oversample factor does not yield `samples`
    /// distinct terms.
    fn sample_size(
        &self,
        id: Id,
        size: usize,
        samples: u64,
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let target = usize::try_from(samples).unwrap();
        let mut drawn = HashSet::with_hasher(FixedState::default());
        let mut prev_end = 0;
        for oversample in powers_of_two::<6>() {
            let end = samples * oversample as u64;
            drawn.par_extend((prev_end..end).into_par_iter().map(|s| {
                let mut rng = utils::combined_rng([size as u64, s, seed[0], seed[1]]);
                self.sample(id, size, &mut rng)
            }));
            if drawn.len() >= target {
                let mut v = drawn.into_iter().collect::<Vec<_>>();
                v.sort_unstable();
                v.truncate(target);
                return Some(v);
            }
            prev_end = end;
        }
        None
    }

    /// Sample exactly n number of terms for each size from the root
    ///
    /// # Errors
    ///
    /// Errors if even with oversampling the sampler does not find enough terms
    fn sample_batch_root(
        &self,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        self.sample_batch(self.root(), samples_per_size, seed)
    }
}

#[must_use]
pub const fn powers_of_two<const N: usize>() -> [usize; N] {
    const { assert!(N <= usize::BITS as usize, "N exceeds usize bit width") };
    let mut out = [0usize; N];
    let mut i = 0;
    while i < N {
        out[i] = 1 << i;
        i += 1;
    }
    out
}
