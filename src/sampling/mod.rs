//! Guide-sample construction from an e-graph.

mod count;
mod draw;
mod weigher;

use std::borrow::Borrow;

use num::{BigUint, ToPrimitive, Zero};
use smallvec::SmallVec;

use crate::utils::HashMap;

pub use draw::{SamplePackage, FrontierPackage, WholePackage};

/// Add the convolution of `a` and `b` (capped at `budget`) into `out`.
fn convolve_into(
    a: &HashMap<usize, BigUint>,
    b: &HashMap<usize, BigUint>,
    budget: usize,
    out: &mut HashMap<usize, BigUint>,
) {
    for (&s_a, c_a) in a {
        for (&s_b, c_b) in b {
            let total = s_a + s_b;
            if total > budget {
                continue;
            }
            let product = c_a * c_b;
            out.entry(total)
                .and_modify(|c| *c += &product)
                .or_insert(product);
        }
    }
}

/// Convolve all child histograms into a single result (left-to-right).
fn convolve<H: Borrow<HashMap<usize, BigUint>>>(
    histograms: &[H],
    budget: usize,
) -> HashMap<usize, BigUint> {
    let mut acc = [(0, BigUint::ONE)].into_iter().collect();
    let mut prev = HashMap::default();

    for h in histograms {
        std::mem::swap(&mut acc, &mut prev);
        convolve_into(&prev, h.borrow(), budget, &mut acc);
        prev.clear();
    }

    acc
}

/// The convolution of two histograms evaluated at exactly `total`:
/// `sum over a + b = total of hist(a) * rest(b)`, iterating the smaller map.
fn convolve_entry(
    hist: &HashMap<usize, BigUint>,
    rest: &HashMap<usize, BigUint>,
    total: usize,
) -> BigUint {
    let (outer, inner) = if hist.len() <= rest.len() {
        (hist, rest)
    } else {
        (rest, hist)
    };
    outer
        .iter()
        .filter_map(|(&a, count_a)| {
            let count_b = total.checked_sub(a).and_then(|b| inner.get(&b))?;
            Some(count_a * count_b)
        })
        .fold(BigUint::ZERO, |acc, c| acc + c)
}

/// The convolution of all `histograms` at exactly `budget`, or `None` when it
/// is zero.
///
/// Only children `1..` are convolved into a table.
fn convolve_at<H: Borrow<HashMap<usize, BigUint>>>(
    histograms: &[H],
    budget: usize,
) -> Option<BigUint> {
    if histograms.iter().any(|h| h.borrow().is_empty()) {
        return None;
    }
    let Some((first, rest)) = histograms.split_first() else {
        // The empty product: one filling, of size zero.
        return (budget == 0).then_some(BigUint::ONE);
    };
    // `convolve` of nothing is the empty product, so a unary node reads its
    // lone child's histogram straight off.
    let tail = convolve(rest, budget);
    let count = convolve_entry(first.borrow(), &tail, budget);
    (!count.is_zero()).then_some(count)
}

/// Convolve child histograms right-to-left, returning suffix intermediates.
/// `suffix[i]` = convolution of children `i..n`, mapping budget -> count.
fn suffix_convolutions<H: Borrow<HashMap<usize, BigUint>>>(
    histograms: &[H],
    budget: usize,
) -> Vec<HashMap<usize, BigUint>> {
    let n = histograms.len();
    let mut suffix = vec![HashMap::default(); n + 1];
    suffix[n] = [(0, BigUint::ONE)].into_iter().collect();

    for i in (0..n).rev() {
        let (left, right) = suffix.split_at_mut(i + 1);
        convolve_into(histograms[i].borrow(), &right[0], budget, &mut left[i]);
    }

    suffix
}

/// Greedily allocate up to `count` sample requests across ascending sizes in
/// `histogram`.
fn greedy_distribute_alloc(count: usize, histogram: &HashMap<usize, BigUint>) -> Vec<(usize, u64)> {
    let mut sizes = histogram.keys().copied().collect::<Vec<_>>();
    sizes.sort_unstable();

    let mut remaining = u64::try_from(count).unwrap();
    let mut requests = Vec::new();
    for size in sizes {
        if remaining == 0 {
            break;
        }
        let available = histogram[&size].to_u64().unwrap_or(u64::MAX);
        let take = remaining.min(available);
        if take > 0 {
            requests.push((size, take));
            remaining -= take;
        }
    }
    requests
}

/// Cartesian product over an iterator of per-slot option iterators.
/// For zero slots, returns `[[]]` (a single empty combination).
fn cartesian_product<T, S, I>(slots: I) -> Vec<SmallVec<[T; 2]>>
where
    T: Copy,
    S: IntoIterator<Item = T>,
    I: IntoIterator<Item = S>,
{
    let mut combos = vec![SmallVec::new()];
    for slot in slots {
        let options = slot.into_iter().collect::<SmallVec<[T; 4]>>();
        if options.is_empty() {
            return Vec::new();
        }
        let mut next = Vec::with_capacity(combos.len() * options.len());
        for prefix in &combos {
            for &opt in &options {
                let mut combo = prefix.clone();
                combo.push(opt);
                next.push(combo);
            }
        }
        combos = next;
    }
    combos
}
