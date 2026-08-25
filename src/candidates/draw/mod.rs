mod frontier;
mod plain;
mod weigher;

use egg::{Id, RecExpr};
use hashbrown::{HashMap, HashSet};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::Counter;
use crate::{MyAnalysis, MyLanguage, OriginLang, utils};

pub use frontier::{BalanceConfig, BalancedFrontierDrawer, IndependentFrontierDrawer};
pub use plain::PlainDrawer;
pub use weigher::{CountWeigher, NaiveWeigher, Weigher};

/// Draws size-targeted candidates from an e-graph.
pub trait Drawer<C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn root(&self) -> Id;

    /// Canonicalize `id` in the underlying e-graph.
    fn find(&self, id: Id) -> Id;

    /// Histogram of extractable term sizes for the canonical class of `id`,
    /// or `None` if the class has no entries under this drawer's constraints.
    fn size_histogram(&self, id: Id) -> Option<&HashMap<usize, C>>;

    /// True iff at least `requested_count + 1` distinct terms of `size` are reachable
    /// from `id` under this drawer's constraints.
    #[cfg(test)]
    fn possible_size(&self, id: Id, size: usize, requested_count: u64) -> bool {
        let canon_id = self.find(id);
        let Some(count) = self.size_histogram(canon_id).and_then(|h| h.get(&size)) else {
            return false;
        };
        C::from_u64(requested_count).is_some_and(|s| count > &s)
    }

    /// All sizes for which `id` has at least one extractable term under this
    /// drawer's constraints. Iteration order is unspecified.
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
    /// Panics if the class has no extractable terms under this drawer's
    /// constraints.
    fn min_size(&self, id: Id) -> usize {
        self.term_sizes(id).into_iter().min().unwrap()
    }

    /// Returns a random (but stable) smallest term
    fn smallest(&self, id: Id) -> RecExpr<OriginLang<L>> {
        let size = self.min_size(id);
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        self.draw(id, size, &mut rng)
    }

    /// Precondition: `possible_size(id, size, 0)`.
    fn draw(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>>;

    /// Draw up to the requested number of distinct terms at each size.
    /// Returns `None` only when every requested size is empty.
    fn draw_batch(
        &self,
        id: Id,
        requests: &[(usize, u64)],
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let terms = requests
            .iter()
            .filter_map(|&(size, requested_count)| self.draw_size(id, size, requested_count, seed))
            .flatten()
            .collect::<Vec<_>>();
        (!terms.is_empty()).then_some(terms)
    }

    /// Draw up to `requested_count` distinct terms of exactly `size`, capped by the
    /// known term count and the 32x draw budget.
    ///
    /// # Errors
    ///
    /// Returns `None` if the size has no extractable terms.
    fn draw_size(
        &self,
        id: Id,
        size: usize,
        requested_count: u64,
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let requested = usize::try_from(requested_count).unwrap();
        // Never chase more distinct terms than the size actually has: the
        // histogram count is exact, so cap the target at `min(requested,
        // available)`. Without this, a size with fewer distinct terms than
        // `requested` would exhaust the whole retry budget and still
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
        // `target` distinct terms are seen or the retry budget runs out;
        // return whatever was collected rather than failing on a short draw.
        let mut rng = utils::combined_rng([size as u64, seed[0], seed[1]]);
        let mut drawn = HashSet::new();
        let mut budget = requested_count * MAX_DRAW_ATTEMPTS_PER_CANDIDATE;
        while drawn.len() < target && budget > 0 {
            drawn.insert(self.draw(id, size, &mut rng));
            budget -= 1;
        }

        if drawn.is_empty() {
            return None;
        }
        let mut v = drawn.into_iter().collect::<Vec<_>>();
        v.sort_unstable();
        Some(v)
    }

    /// Draw up to the requested number of terms at each size from the root.
    fn draw_root_batch(
        &self,
        requests: &[(usize, u64)],
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        self.draw_batch(self.root(), requests, seed)
    }
}

/// Cap on how many draws `draw_size` attempts per requested candidate.
pub(super) const MAX_DRAW_ATTEMPTS_PER_CANDIDATE: u64 = 32;
