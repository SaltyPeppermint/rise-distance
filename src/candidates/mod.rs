//! Guide-candidate construction from an e-graph.

mod allocation;

pub mod count;
pub mod draw;
mod package;

pub use allocation::{SelectionPolicy, SizeAllocation, uniform_candidate_allocation};

pub use package::ExactCandidatePackage;

use std::borrow::Borrow;

use hashbrown::HashMap;

use crate::Counter;

/// Convolve all child histograms into a single result (left-to-right).
pub fn convolve<C: Counter, H: Borrow<HashMap<usize, C>>>(
    histograms: &[H],
    budget: usize,
) -> HashMap<usize, C> {
    let mut acc = HashMap::from([(0, C::one())]);
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

fn convolve_at<C: Counter>(histograms: &[&HashMap<usize, C>], budget: usize) -> Option<C> {
    if histograms.iter().any(|h| h.is_empty()) {
        return None;
    }
    convolve(histograms, budget).get(&budget).cloned()
}

/// Convolve child histograms right-to-left, returning suffix intermediates.
/// `suffix[i]` = convolution of children `i..n`, mapping budget -> count.
pub fn suffix_convolutions<C: Counter, H: Borrow<HashMap<usize, C>>>(
    histograms: &[H],
    budget: usize,
) -> Vec<HashMap<usize, C>> {
    let n = histograms.len();
    let mut suffix = vec![HashMap::new(); n + 1];
    suffix[n] = HashMap::from([(0, C::one())]);

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
                    .and_modify(|c: &mut C| *c += &product)
                    .or_insert(product);
            }
        }
    }

    suffix
}
