use num::{BigUint, ToPrimitive, Zero};
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

use crate::cli::Policy;
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

/// Local selection distribution over feasible derivation choices.
#[derive(Clone, Copy)]
pub(super) enum Weigher {
    Uniform,
    Count,
}

impl From<Policy> for Weigher {
    fn from(policy: Policy) -> Self {
        match policy {
            Policy::Count => Self::Count,
            Policy::Uniform | Policy::Smallest => Self::Uniform,
        }
    }
}

impl Weigher {
    /// Pick an index into `candidates`, each having strictly positive counts.
    ///
    /// When `rest_total` is `Some(rt)`, choices whose capacity
    /// `local_count * rt` is already exhausted at `cursor` are excluded.
    /// When `rest_total` is `None`, `cursor` has no prior draws and every
    /// candidate has `used == 0`.
    pub(super) fn pick(
        self,
        candidates: &[Choice<'_>],
        rest_total: Option<&BigUint>,
        cursor: &mut DrawCursor<'_>,
    ) -> usize {
        match self {
            Self::Uniform => Self::pick_uniform(candidates, rest_total, cursor),
            Self::Count => Self::pick_count(candidates, rest_total, cursor),
        }
    }

    fn pick_uniform(
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

    fn pick_count(
        candidates: &[Choice<'_>],
        rest_total: Option<&BigUint>,
        cursor: &mut DrawCursor<'_>,
    ) -> usize {
        let Some(rt) = rest_total else {
            return WeightedIndex::new(candidates.iter().map(|choice| {
                choice
                    .rest
                    .map_or_else(|| choice.count.clone(), |r| choice.count * r)
            }))
            .expect("weights contain a positive choice")
            .sample(cursor.rng);
        };
        WeightedIndex::new(candidates.iter().map(|choice| {
            let used = cursor.used(choice.key);
            let mut cap = match choice.rest {
                Some(rest) => choice.count * rest * rt,
                None => choice.count * rt,
            };
            debug_assert!(
                cap.to_u64().is_none_or(|v| v >= used),
                "used ({used}) exceeded capacity ({cap})"
            );
            cap -= used;
            cap
        }))
        .expect("residual weights contain a positive choice")
        .sample(cursor.rng)
    }
}

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
