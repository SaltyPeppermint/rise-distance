mod frontier;
mod plain;
mod weigher;

use egg::{Id, RecExpr};
use hashbrown::{HashMap, HashSet};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::Counter;
use crate::{MyAnalysis, MyLanguage, OriginLang, utils};

pub use frontier::{BalanceConfig, BalancedFrontierSampler, IndependentFrontierSampler};
pub use plain::PlainSampler;
pub use weigher::{CountWeigher, NaiveWeigher, Weigher};

/// Common interface for samplers that draw size-targeted terms from an e-graph.
///
/// `sample_batch` and `sample_batch_root` are provided as default implementations
/// in terms of [`Sampler::sample`].
pub trait Sampler<C, L, N>
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
    #[cfg(test)]
    fn possible_size(&self, id: Id, size: usize, samples: u64) -> bool {
        let canon_id = self.find(id);
        let Some(count) = self.size_histogram(canon_id).and_then(|h| h.get(&size)) else {
            return false;
        };
        C::from_u64(samples).is_some_and(|s| count > &s)
    }

    /// All sizes for which `id` has at least one extractable term under this
    /// sampler's constraints. Iteration order is unspecified.
    fn term_sizes(&self, id: Id) -> Vec<usize> {
        let canon_id = self.find(id);
        self.size_histogram(canon_id)
            .map(|h| h.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Smallest size with at least one extractable term.
    ///
    /// # Panics
    ///
    /// Panics if the class has no extractable terms under this sampler's
    /// constraints.
    fn min_size(&self, id: Id) -> usize {
        self.term_sizes(id).into_iter().min().unwrap()
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
            .iter()
            .filter_map(|&(size, samples)| self.sample_size(id, size, samples, seed))
            .flatten()
            .collect::<Vec<_>>();
        (!terms.is_empty()).then_some(terms)
    }

    /// Draw up to `samples` distinct terms of exactly `size` from `id`,
    /// oversampling up to a factor of 2^5 to hit the target. A size holds a
    /// fixed number of distinct terms; if that is below `samples` no amount of
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
        // `requested` would draw through the whole oversample budget and still
        // fail. A count that overflows `u64`/`usize` far exceeds any
        // `requested`, so saturate to `requested` there rather than capping low.
        let count = self.size_histogram(self.find(id))?.get(&size)?;
        let target = requested.min(count.to_usize().unwrap_or(requested));
        if target == 0 {
            return None;
        }

        // Deterministic draw stream: `size` and `seed` seed a single RNG, so
        // for a fixed seed the result set is reproducible (and distinct sizes
        // never share a stream). Draw with replacement into a set until either
        // `target` distinct terms are seen or the oversample budget runs out;
        // return whatever was collected rather than failing on a short draw.
        let mut rng = utils::combined_rng([size as u64, seed[0], seed[1]]);
        let mut drawn = HashSet::new();
        let mut budget = samples * MAX_OVERSAMPLE;
        while drawn.len() < target && budget > 0 {
            drawn.insert(self.sample(id, size, &mut rng));
            budget -= 1;
        }

        if drawn.is_empty() {
            return None;
        }
        let mut v = drawn.into_iter().collect::<Vec<_>>();
        v.sort_unstable();
        Some(v)
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

/// Cap on how many draws `sample_size` attempts per requested sample.
const MAX_OVERSAMPLE: u64 = 32;
