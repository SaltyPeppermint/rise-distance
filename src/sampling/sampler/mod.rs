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

    /// Sample up to `samples` distinct terms for each size from an eclass and
    /// concatenate them.
    ///
    /// Each size contributes as many distinct terms as it can, capped at its
    /// requested count: a size whose frontier holds fewer than the requested
    /// number of distinct terms simply contributes all of them rather than
    /// failing the whole batch. Returns `None` only if *every* requested size
    /// is empty (nothing at all could be drawn), so callers still see a clean
    /// failure for a wholly empty frontier.
    fn sample_batch(
        &self,
        id: Id,
        samples_per_size: &[(usize, u64)],
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let terms = samples_per_size
            .par_iter()
            .flat_map(|&(size, samples)| self.sample_size(id, size, samples, seed))
            .flatten()
            .collect::<Vec<_>>();
        (!terms.is_empty()).then_some(terms)
    }

    /// Draw up to `samples` distinct terms of exactly `size` from `id`, doubling
    /// the oversample factor up to 2^5 to hit the target. A size holds a fixed
    /// number of distinct terms; if that is below `samples` no amount of
    /// oversampling can reach the target, so the draw is capped at the size's
    /// known distinct-term count (from the histogram) and returns everything it
    /// finds instead of failing.
    ///
    /// # Errors
    ///
    /// Returns `None` only if the size has no extractable terms at all, so
    /// nothing could be drawn.
    fn sample_size(
        &self,
        id: Id,
        size: usize,
        samples: u64,
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let requested = usize::try_from(samples).unwrap();
        // Never chase more distinct terms than the size actually has: the
        // histogram count is exact, so cap the target at `min(requested,
        // available)`. Without this, a size with fewer distinct terms than
        // `requested` would spin through every oversample factor and still fail.
        // A count that overflows `u64`/`usize` far exceeds any `requested`, so
        // saturate to `requested` there rather than capping low.
        let available = self
            .size_histogram(self.find(id))
            .and_then(|h| h.get(&size))
            .map_or(0, |c| {
                TryInto::<u64>::try_into(c.clone())
                    .ok()
                    .and_then(|c| usize::try_from(c).ok())
                    .unwrap_or(requested)
            });
        let target = requested.min(available);
        if target == 0 {
            return None;
        }
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
