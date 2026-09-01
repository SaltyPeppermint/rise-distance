//! Guide-candidate construction from an e-graph.

pub mod count;
pub mod draw;

pub use draw::{DrawerPackage, FrontierPackage, PlainPackage};

use std::borrow::Borrow;

use hashbrown::HashMap;
use num::{BigUint, ToPrimitive};

/// Convolve all child histograms into a single result (left-to-right).
pub fn convolve<H: Borrow<HashMap<usize, BigUint>>>(
    histograms: &[H],
    budget: usize,
) -> HashMap<usize, BigUint> {
    let mut acc = HashMap::from([(0, BigUint::ONE)]);
    let mut prev = HashMap::new();

    for h in histograms {
        std::mem::swap(&mut acc, &mut prev);
        for (&s_acc, c_acc) in &prev {
            for (&s_h, c_h) in h.borrow() {
                let total = s_acc + s_h;
                if total > budget {
                    continue;
                }
                let product = c_acc.to_owned() * c_h;
                acc.entry(total)
                    .and_modify(|c| *c += &product)
                    .or_insert(product);
            }
        }
        prev.clear();
    }

    acc
}

fn convolve_at(histograms: &[&HashMap<usize, BigUint>], budget: usize) -> Option<BigUint> {
    if histograms.iter().any(|h| h.is_empty()) {
        return None;
    }
    convolve(histograms, budget).get(&budget).cloned()
}

/// Convolve child histograms right-to-left, returning suffix intermediates.
/// `suffix[i]` = convolution of children `i..n`, mapping budget -> count.
pub fn suffix_convolutions<H: Borrow<HashMap<usize, BigUint>>>(
    histograms: &[H],
    budget: usize,
) -> Vec<HashMap<usize, BigUint>> {
    let n = histograms.len();
    let mut suffix = vec![HashMap::new(); n + 1];
    suffix[n] = HashMap::from([(0, BigUint::ONE)]);

    for i in (0..n).rev() {
        let (left, right) = suffix.split_at_mut(i + 1);
        for (&s_i, c_i) in histograms[i].borrow() {
            for (&s_rest, c_rest) in &right[0] {
                let total = s_i + s_rest;
                if total > budget {
                    continue;
                }
                let product = c_i.to_owned() * c_rest;
                left[i]
                    .entry(total)
                    .and_modify(|c| *c += &product)
                    .or_insert(product);
            }
        }
    }

    suffix
}

/// Distribute `total_candidates` uniformly across `sizes`.
///
/// Any remainder is assigned to the first sizes, so the returned counts always
/// add up to `total_candidates` when at least one size is supplied.
///
/// # Panics
///
/// On platforms where `usize` is wider than `u64`, panics if the per-size
/// candidate count cannot be represented as a `u64`.
#[must_use]
pub fn uniform_candidate_allocation(sizes: &[usize], total_candidates: usize) -> Vec<(usize, u64)> {
    if sizes.is_empty() {
        return vec![];
    }
    let size_count = sizes.len();
    let base = u64::try_from(total_candidates / size_count).unwrap();
    let remainder = total_candidates % size_count;
    sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| (size, base + u64::from(i < remainder)))
        .collect()
}

#[must_use]
#[expect(clippy::missing_panics_doc)]
pub fn greedy_distribute_alloc(
    min_size: usize,
    max_size: usize,
    count: usize,
    histogram: &HashMap<usize, BigUint>,
) -> Vec<(usize, u64)> {
    (min_size..=max_size)
        .scan(u64::try_from(count).unwrap(), |remaining, size| {
            let available = histogram
                .get(&size)
                .map_or(0, |c| c.to_u64().unwrap_or(u64::MAX));

            let take = (*remaining).min(available);
            *remaining = remaining.saturating_sub(available);

            Some((size, take))
        })
        .collect::<Vec<_>>()
}
