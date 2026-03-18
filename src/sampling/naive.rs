use crate::count::{Counter, TermCount};
use crate::sampling::Sampler;
use crate::{EClassId, Label, TreeNode};

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

pub struct NaiveSampler<'a, C: Counter, L: Label>(&'a TermCount<'a, C, L>);

impl<'a, C: Counter, L: Label> NaiveSampler<'a, C, L> {
    #[must_use]
    pub fn new(term_count: &'a TermCount<'a, C, L>) -> Self {
        Self(term_count)
    }
}

impl<C: Counter, L: Label> Sampler<L> for NaiveSampler<'_, C, L> {
    fn root(&self) -> EClassId {
        self.0.graph.root()
    }

    /// Sample unique terms across a range of sizes.
    ///
    /// For each size in `[min_size, max_size]` that the root e-class actually has
    /// terms for, samples `samples_fn` terms and deduplicates them.
    ///
    /// Only sizes present in the root's histogram are sampled. The root e-class
    /// may have gaps in its reachable sizes (e.g. terms only at sizes 5, 7, 9),
    /// and calling `sample` with a size that has no terms would cause all node
    /// weights to be zero, panicking with `AllWeightsZero`.
    fn sample_unique(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<TreeNode<L>> {
        let canon_id = self.0.graph.canonicalize(id);
        self.0
            .data
            .get(&canon_id)
            .into_iter()
            .flat_map(|h| h.keys().filter(|&&s| s >= min_size && s <= max_size))
            .par_bridge()
            .flat_map(|&size| self.sample_root(size, samples_per_size[&size], size as u64))
            .collect()
    }

    fn sample<R: rand::Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> TreeNode<L> {
        todo!()
    }
}
