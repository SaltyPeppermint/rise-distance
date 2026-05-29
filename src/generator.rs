use egg::{AstSize, CostFunction, RecExpr};
use rand::Rng;

use crate::MyLanguage;

/// Shared interface for Boltzmann samplers of random terms.
///
/// Each language provides its own implementation with language-specific grammar,
/// generating-function formulas, and post-validation (via [`extra_valid`]). The
/// rejection-sampling loop in [`sample`] / [`sample_many`] is shared.
///
/// [`extra_valid`]: BoltzmannSampler::extra_valid
/// [`sample`]: BoltzmannSampler::sample
/// [`sample_many`]: BoltzmannSampler::sample_many
pub trait BoltzmannSampler: Sized {
    type Lang: MyLanguage;

    /// Construct a sampler targeting terms of `target` size (± `tolerance`).
    /// `leaf_symbols` overrides the default leaf pool when `Some`.
    fn new(target: usize, tolerance: usize, leaf_symbols: Option<Vec<Self::Lang>>) -> Self;

    fn target(&self) -> usize;
    fn tolerance(&self) -> usize;

    /// Generate one random expression. Called by the rejection-sampling loop.
    fn gen_node<R: Rng>(&self, rng: &mut R, depth: usize) -> RecExpr<Self::Lang>;

    /// Language-specific post-validation (e.g. binder scoping). Default: always valid.
    #[must_use]
    fn extra_valid(_expr: &RecExpr<Self::Lang>) -> bool {
        true
    }

    /// Sample a single expression whose size is in `[target - tolerance, target + tolerance]`,
    /// passes [`extra_valid`], and is accepted by `filter_hook`.
    /// Returns `None` if no valid expression is found within `100_000` attempts.
    ///
    /// [`extra_valid`]: BoltzmannSampler::extra_valid
    fn sample<R: Rng, T, F: Fn(&RecExpr<Self::Lang>) -> Option<T>>(
        &self,
        rng: &mut R,
        filter_hook: &F,
    ) -> Option<(RecExpr<Self::Lang>, T, usize)> {
        let lo = self.target().saturating_sub(self.tolerance());
        let hi = self.target() + self.tolerance();
        (0..100_000).find_map(|n| {
            let candidate = self.gen_node(rng, 0);
            if (lo..=hi).contains(&AstSize.cost_rec(&candidate))
                && Self::extra_valid(&candidate)
                && let Some(reason) = filter_hook(&candidate)
            {
                return Some((candidate, reason, n));
            }
            None
        })
    }

    /// Generate `count` random terms within the size window.
    fn sample_many<R: Rng, T, F: Fn(&RecExpr<Self::Lang>) -> Option<T>>(
        &self,
        rng: &mut R,
        count: usize,
        filter_hook: &F,
    ) -> Vec<(RecExpr<Self::Lang>, T)> {
        let (exprs, total_attempts, failed) =
            (0..count).map(|_| self.sample(rng, filter_hook)).fold(
                (Vec::with_capacity(count), 0, 0),
                |(mut exprs, attempts, failed), result| match result {
                    Some((expr, reason, a)) => {
                        exprs.push((expr, reason));
                        (exprs, attempts + a, failed)
                    }
                    None => (exprs, attempts, failed + 1),
                },
            );
        println!(
            "Took a total of {total_attempts} attempts for {} terms. {failed} failed!",
            exprs.len()
        );
        exprs
    }
}
