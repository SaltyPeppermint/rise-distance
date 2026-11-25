use num::{BigUint, ToPrimitive, Zero};
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

use crate::sampling::draw::DrawCursor;

/// One feasible decision candidate with trie edge `key` and local derivation
/// count `count * rest.unwrap_or(1)`.
#[derive(Clone, Copy)]
pub(super) struct Choice<'a> {
    pub(super) key: usize,
    pub(super) count: &'a BigUint,
    pub(super) rest: Option<&'a BigUint>,
}

impl<'a> Choice<'a> {
    #[must_use]
    pub(super) const fn branch(key: usize, count: &'a BigUint) -> Self {
        Self {
            key,
            count,
            rest: None,
        }
    }

    #[must_use]
    pub(super) const fn child_size(
        key: usize,
        child_count: &'a BigUint,
        rest_count: &'a BigUint,
    ) -> Self {
        Self {
            key,
            count: child_count,
            rest: Some(rest_count),
        }
    }
}

pub(super) trait Weigher {
    /// Pick an index into `candidates`, each having strictly positive counts.
    ///
    /// When `rest_total` is `Some(rt)`, choices whose capacity
    /// `local_count * rt` is already exhausted at `cursor` are excluded.
    /// When `rest_total` is `None`, `cursor` has no prior draws and every
    /// candidate has `used == 0`.
    fn pick(
        &self,
        candidates: &[Choice<'_>],
        rest_total: Option<&BigUint>,
        cursor: &mut DrawCursor<'_>,
    ) -> usize;
}

pub(super) struct Uniform;

/// True iff the product of positive `factors` strictly exceeds `used`.
fn product_exceeds(factors: &[&BigUint], used: u64) -> bool {
    if used == 0 {
        return true;
    }
    let limit = u128::from(used);
    let mut prod: u128 = 1;
    for factor in factors {
        debug_assert!(!factor.is_zero(), "candidate count must be positive");
        let Some(value) = factor.to_u64() else {
            return true;
        };
        if value > used {
            return true;
        }
        prod = prod.saturating_mul(u128::from(value));
        if prod > limit {
            return true;
        }
    }
    false
}

impl Weigher for Uniform {
    fn pick(
        &self,
        candidates: &[Choice<'_>],
        rest_total: Option<&BigUint>,
        cursor: &mut DrawCursor<'_>,
    ) -> usize {
        let Some(rt) = rest_total else {
            return cursor.rng.gen_range(0..candidates.len());
        };
        let feasible = candidates
            .iter()
            .enumerate()
            .filter_map(|(idx, choice)| {
                let used = cursor.used(choice.key);
                let has_room = match choice.rest {
                    Some(rest) => product_exceeds(&[choice.count, rest, rt], used),
                    None => product_exceeds(&[choice.count, rt], used),
                };
                has_room.then_some(idx)
            })
            .collect::<Vec<_>>();
        assert!(
            !feasible.is_empty(),
            "at least one choice has remaining unsampled terms"
        );
        feasible[cursor.rng.gen_range(0..feasible.len())]
    }
}

pub(super) struct Count;

fn residual_weight(mut cap: BigUint, used: u64) -> Option<BigUint> {
    if used == 0 {
        return (!cap.is_zero()).then_some(cap);
    }
    debug_assert!(
        cap.to_u64().is_none_or(|v| v >= used),
        "used ({used}) exceeded capacity ({cap})"
    );
    cap -= used;
    (!cap.is_zero()).then_some(cap)
}

impl Weigher for Count {
    fn pick(
        &self,
        candidates: &[Choice<'_>],
        rest_total: Option<&BigUint>,
        cursor: &mut DrawCursor<'_>,
    ) -> usize {
        let Some(rt) = rest_total else {
            return WeightedIndex::new(
                candidates
                    .iter()
                    .map(|choice| choice.rest.map_or_else(|| choice.count.clone(), |r| choice.count * r)),
            )
            .expect("weights contain a positive choice")
            .sample(cursor.rng);
        };
        let mut indices = Vec::with_capacity(candidates.len());
        let mut weights = Vec::with_capacity(candidates.len());
        for (idx, choice) in candidates.iter().enumerate() {
            let used = cursor.used(choice.key);
            let cap = match choice.rest {
                Some(rest) => choice.count * rest * rt,
                None => choice.count * rt,
            };
            if let Some(rem) = residual_weight(cap, used) {
                indices.push(idx);
                weights.push(rem);
            }
        }
        let pick = WeightedIndex::new(&weights)
            .expect("residual weights contain a positive choice")
            .sample(cursor.rng);
        indices[pick]
    }
}
