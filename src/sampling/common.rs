use egg::{Analysis, EGraph, Id, Language, RecExpr};
use hashbrown::HashSet;
use rayon::prelude::*;

use crate::count::{Counter, TermCount};
use crate::egg::TypeAnalysisWrapper;
use crate::origin::OriginNode;
use crate::utils::combined_rng;

use super::Sampler;

pub(super) fn possible_size<C: Counter, L: Language, N: Analysis<L>>(
    term_count: &TermCount<C>,
    graph: &EGraph<L, TypeAnalysisWrapper<N>>,
    id: Id,
    size: usize,
    samples: u64,
) -> bool {
    let canon_id = graph.find(id);
    let Some(count) = term_count.data.get(&canon_id).and_then(|h| h.get(&size)) else {
        return false;
    };
    samples.try_into().is_ok_and(|s: C| count > &s)
}

pub(super) fn sample_batch<const PARALLEL: bool, S, F>(
    sampler: &S,
    id: Id,
    samples_per_size: &[(usize, u64)],
    seed: [u64; 2],
    check: F,
) -> HashSet<RecExpr<OriginNode<S::Lang>>>
where
    S: Sampler,
    F: for<'a> Fn(&'a RecExpr<OriginNode<S::Lang>>) -> bool + Sync,
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
