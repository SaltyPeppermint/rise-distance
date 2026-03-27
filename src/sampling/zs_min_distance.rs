use hashbrown::{HashMap, HashSet};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use crate::sampling::Sampler;
use crate::tree::OriginTree;
use crate::zs::EditCosts;
use crate::{EClassId, TreeShaped, tree_distance};

/// Greedily only accepts new terms that have a bigger or equal zs distance
pub struct ZSDistanceSampler<E: EditCosts<S::Label>, S: Sampler> {
    inner: S,
    cost_fn: E,
    with_types: bool,
}

impl<E: EditCosts<S::Label>, S: Sampler> ZSDistanceSampler<E, S> {
    #[must_use]
    pub fn new(inner: S, cost_fn: E, with_types: bool) -> Self {
        Self {
            inner,
            cost_fn,
            with_types,
        }
    }
}

impl<E: EditCosts<S::Label>, S: Sampler> Sampler for ZSDistanceSampler<E, S> {
    type Label = S::Label;

    fn root(&self) -> EClassId {
        self.inner.root()
    }

    fn possible_size(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
    ) -> impl Iterator<Item = usize> + Send {
        self.inner.possible_size(id, min_size, max_size)
    }

    fn sample_batch(
        &self,
        id: EClassId,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<OriginTree<S::Label>> {
        self.possible_size(id, min_size, max_size)
            .par_bridge()
            .flat_map(|size| {
                let mut samples_to_take = samples_per_size[&size];
                let mut existing_flat = HashSet::new();
                let mut existing = HashSet::new();
                let mut rng = ChaCha12Rng::seed_from_u64(size as u64);
                let mut current_max = 0;
                while samples_to_take > 0 {
                    let new_candidate = self.sample(id, size, &mut rng);
                    if existing.contains(&new_candidate) {
                        continue;
                    }
                    let candidate_flat = new_candidate.flatten(self.with_types);
                    if let Some(new_max) = existing_flat.iter().try_fold(current_max, |acc, e| {
                        let td = tree_distance(e, &candidate_flat, &self.cost_fn);
                        (td > current_max).then_some(acc.max(td))
                    }) {
                        existing_flat.insert(candidate_flat);
                        existing.insert(new_candidate);
                        samples_to_take -= 1;
                        current_max = new_max;
                    }
                }
                existing.into_par_iter()
            })
            .collect()
    }

    /// Sample uniformly: each feasible choice gets equal weight.
    fn sample<R: Rng>(&self, id: EClassId, size: usize, rng: &mut R) -> OriginTree<S::Label> {
        self.inner.sample(id, size, rng)
    }
}
