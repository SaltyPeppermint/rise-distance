use egg::{EGraph, Id, RecExpr};
use hashbrown::HashSet;
use rayon::prelude::*;

use crate::count::{Counter, TermCount};
use crate::utils::combined_rng;
use crate::{MyAnalysis, MyLanguage, OriginLang};

use crate::sampling::Sampler;

pub(super) fn possible_size<C, L, N>(
    term_count: &TermCount<C>,
    graph: &EGraph<L, N>,
    id: Id,
    size: usize,
    samples: u64,
) -> bool
where
    L: MyLanguage,
    C: Counter,
    N: MyAnalysis<L>,
{
    let canon_id = graph.find(id);
    let Some(count) = term_count.data().get(&canon_id).and_then(|h| h.get(&size)) else {
        return false;
    };
    samples.try_into().is_ok_and(|s: C| count > &s)
}

pub(super) fn sample_batch<const PARALLEL: bool, L, S, F>(
    sampler: &S,
    id: Id,
    samples_per_size: &[(usize, u64)],
    seed: [u64; 2],
    check: F,
) -> HashSet<RecExpr<OriginLang<L>>>
where
    L: MyLanguage,
    S: Sampler<L>,
    F: for<'a> Fn(&'a RecExpr<OriginLang<L>>) -> bool + Sync,
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
