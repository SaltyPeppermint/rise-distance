mod frontier;
mod trie;
mod whole;

use std::borrow::Borrow;

use egg::{Language, RecExpr};
use num::{BigUint, ToPrimitive, Zero};
use rand_chacha::ChaCha12Rng;
use thiserror::Error;

use crate::cli::Policy;
use crate::origin::OriginLang;
use crate::sampling;
use crate::sampling::weigher::{Choice, Weigher};
use crate::utils::{self, HashMap};

pub use frontier::FrontierPackage;
pub(super) use trie::DrawCursor;
use trie::SampledTrie;
pub use whole::WholePackage;

/// Draws size-targeted root samples from an e-graph.
trait Drawer<L: Language> {
    /// Histogram of extractable term sizes for the drawer's root class,
    /// or `None` if the root has no entries under this drawer's constraints.
    fn root_histogram(&self) -> Option<&HashMap<usize, BigUint>>;

    /// Draw one root term of `size` not yet present in `sampled`, recording its
    /// derivation path in `sampled`.
    ///
    /// Precondition: fewer than `root_histogram()[size]` terms have already
    /// been recorded in `sampled`.
    fn draw_next(
        &self,
        size: usize,
        sampled: &mut SampledTrie,
        rng: &mut ChaCha12Rng,
    ) -> RecExpr<OriginLang<L>>;

    /// Draw the requested number of distinct root terms at each size, without
    /// replacement.
    ///
    /// # Errors
    ///
    /// Errors if any requested size has fewer extractable terms than requested.
    fn draw_batch(
        &self,
        requests: &[(usize, u64)],
        seed: [u64; 2],
    ) -> Result<Vec<RecExpr<OriginLang<L>>>, DrawingError> {
        let hist = self.root_histogram();
        let feasible = requests.iter().all(|&(size, req)| {
            req == 0
                || hist
                    .and_then(|h| h.get(&size))
                    .is_some_and(|avail| avail.to_u64().is_none_or(|a| a >= req))
        });
        if !feasible {
            return Err(DrawingError::InsufficientTerms);
        }

        let mut terms = Vec::new();
        for &(size, requested_count) in requests {
            if requested_count == 0 {
                continue;
            }
            let target = usize::try_from(requested_count).unwrap();
            let mut rng = utils::combined_rng([size as u64, seed[0], seed[1]]);
            let mut sampled = SampledTrie::new();
            let mut drawn = Vec::with_capacity(target);
            for _ in 0..target {
                drawn.push(self.draw_next(size, &mut sampled, &mut rng));
            }
            drawn.sort_unstable();
            debug_assert!(
                drawn.windows(2).all(|pair| pair[0] != pair[1]),
                "sampled terms must be pairwise distinct"
            );
            terms.extend(drawn);
        }
        Ok(terms)
    }

    /// Returns a random (but stable) smallest root term.
    ///
    /// # Panics
    ///
    /// Panics if the root has no extractable terms under this drawer's
    /// constraints.
    fn smallest(&self) -> RecExpr<OriginLang<L>> {
        const SMALLEST_SEED: [u64; 2] = [0, 0];

        let size = *self
            .root_histogram()
            .and_then(|h| h.keys().min())
            .expect("root has at least one extractable term");
        self.draw_batch(&[(size, 1)], SMALLEST_SEED)
            .expect("smallest size has at least one term")
            .pop()
            .expect("batch of one returns one term")
    }

    /// Draw `count` samples according to `policy` from this drawer's root.
    fn draw_for_policy(
        &self,
        count: usize,
        policy: Policy,
        seed: [u64; 2],
    ) -> Result<Vec<RecExpr<OriginLang<L>>>, DrawingError> {
        if matches!(policy, Policy::Smallest) {
            return Ok(vec![self.smallest()]);
        }
        let histogram = self
            .root_histogram()
            .expect("root histogram present iff package built");
        let requests = sampling::greedy_distribute_alloc(count, histogram);
        self.draw_batch(&requests, seed)
    }
}

#[derive(Error, Debug)]
pub enum DrawingError {
    #[error("Insufficient terms")]
    InsufficientTerms,
}

pub trait SamplePackage<L: Language> {
    /// Log the stats about the root into stderr.
    fn log_root_counts(&self) {
        let mut sorted_hist = self.root_histogram().iter().collect::<Vec<_>>();
        sorted_hist.sort_unstable_by_key(|&(&size, _)| size);
        eprintln!("Terms in package:");
        for (k, v) in sorted_hist {
            eprintln!("{v} terms of size {k}");
        }
    }

    /// Draw `count` samples for specific size under a specific policy with a specific seed.
    ///
    /// # Errors
    ///
    /// Errors if the drawing fails and not enough terms could be sampled.
    fn draw_samples(
        &self,
        count: usize,
        policy: Policy,
        seed: [u64; 2],
    ) -> Result<Vec<RecExpr<OriginLang<L>>>, DrawingError>;

    fn root_histogram(&self) -> &HashMap<usize, BigUint>;
}

/// Draw each child subterm for a chosen production, splitting `child_budget`
/// across children `0..arity - 1` via suffix convolutions and propagating
/// residual trie capacity into each recursive call.
pub(super) fn draw_children<L, H, F>(
    weigher: Weigher,
    child_hists: &[H],
    child_budget: usize,
    active_rest: Option<&BigUint>,
    cursor: &mut DrawCursor<'_>,
    mut draw_child: F,
) -> Vec<RecExpr<OriginLang<L>>>
where
    L: Language,
    H: Borrow<HashMap<usize, BigUint>>,
    F: FnMut(usize, usize, Option<&BigUint>, &mut DrawCursor<'_>) -> RecExpr<OriginLang<L>>,
{
    let Some((_, init)) = child_hists.split_last() else {
        return Vec::new();
    };
    let mut drawn = Vec::with_capacity(child_hists.len());
    let mut remaining = child_budget;

    if !init.is_empty() {
        let suffix = sampling::suffix_convolutions(child_hists, child_budget);
        for (i, hist) in init.iter().enumerate() {
            let size_rest = cursor.active_rest(active_rest);
            let mut choices = hist
                .borrow()
                .iter()
                .filter_map(|(&child_size, child_count)| {
                    let rest_size = remaining.checked_sub(child_size)?;
                    let rest_count = suffix[i + 1].get(&rest_size)?;
                    (!rest_count.is_zero())
                        .then(|| Choice::child_size(child_size, child_count, rest_count))
                })
                .collect::<Vec<_>>();
            choices.sort_unstable_by_key(|choice| choice.key);
            assert!(
                !choices.is_empty(),
                "chosen production has a feasible child-size split"
            );

            let chosen = choices[weigher.pick(&choices, size_rest, cursor)];
            cursor.step(chosen.key);
            remaining -= chosen.key;

            let rest_count = chosen.rest.expect("child-size choice carries suffix count");
            let child_rest = cursor.active_rest(size_rest).map(|rt| rest_count * rt);
            drawn.push(draw_child(i, chosen.key, child_rest.as_ref(), cursor));
        }
    }

    let next_rest = cursor.active_rest(active_rest);
    drawn.push(draw_child(init.len(), remaining, next_rest, cursor));
    drawn
}
