use hashbrown::HashSet;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use crate::Graph;
use crate::count::{Counter, TermCount};
use crate::ids::EClassId;
use crate::nodes::Label;
use crate::tree::OriginTree;

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

pub(super) fn sample_batch<S: Sampler>(
    sampler: &S,
    id: EClassId,
    samples_per_size: &[(usize, u64)],
) -> HashSet<OriginTree<S::Label>> {
    samples_per_size
        .par_iter()
        .filter(|(size, samples)| sampler.possible_size(id, *size, *samples))
        .flat_map_iter(|(size, samples)| {
            let mut rng = ChaCha12Rng::seed_from_u64(*size as u64);
            (0..*samples).map(move |sample| {
                rng.set_stream(sample);
                sampler.sample(id, *size, &mut rng)
            })
        })
        .collect()
}
