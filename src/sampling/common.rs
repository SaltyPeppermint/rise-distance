use hashbrown::HashSet;
use rayon::prelude::*;

use crate::Graph;
use crate::count::{Counter, TermCount};
use crate::ids::EClassId;
use crate::nodes::Label;
use crate::tree::OriginTree;
use crate::utils::combined_rng;

use super::Sampler;

pub(super) fn possible_size<C: Counter>(
    term_count: &TermCount<C>,
    graph: &Graph<impl Label>,
    id: EClassId,
    size: usize,
    samples: u64,
) -> bool {
    let canon_id = graph.canonicalize(id);
    let Some(count) = term_count.data.get(&canon_id).and_then(|h| h.get(&size)) else {
        return false;
    };
    samples.try_into().is_ok_and(|s: C| count > &s)
}

pub(super) fn sample_batch<const PARALLEL: bool, S, F>(
    sampler: &S,
    id: EClassId,
    samples_per_size: &[(usize, u64)],
    seed: [u64; 2],
    check: F,
) -> HashSet<OriginTree<S::Label>>
where
    S: Sampler,
    F: for<'a> Fn(&'a OriginTree<S::Label>) -> bool + Sync,
{
    if PARALLEL {
        samples_per_size
            .par_iter()
            .filter(|(size, samples)| sampler.possible_size(id, *size, *samples))
            .flat_map(|(size, samples)| {
                (0..*samples).into_par_iter().filter_map(|s| {
                    let candidate = sampler.sample(
                        id,
                        *size,
                        &mut combined_rng([*size as u64, s, seed[0], seed[1]]),
                    );
                    check(&candidate).then_some(candidate)
                })
            })
            .collect()
    } else {
        samples_per_size
            .iter()
            .filter(|(size, samples)| sampler.possible_size(id, *size, *samples))
            .flat_map(|(size, samples)| {
                (0..*samples).filter_map(|s| {
                    let candidate = sampler.sample(
                        id,
                        *size,
                        &mut combined_rng([*size as u64, s, seed[0], seed[1]]),
                    );
                    check(&candidate).then_some(candidate)
                })
            })
            .collect()
    }
}
