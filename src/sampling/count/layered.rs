//! Size-layered term counting.
//!
//! Because an e-node costs one, counts at a size depend only on smaller sizes.
//! [`LayeredDp`] therefore handles cyclic e-graphs without a fixpoint. Root
//! budgets restrict counting to states usable below the requested size limit.
//! See `docs/counting/layered_counting.md`.

use std::hash::Hash;

use num::{BigUint, Zero};

use crate::utils::HashMap;

use crate::sampling::convolve_entry;

/// Per key, per node: suffix convolution tables in the shape of
/// [`suffix_convolutions`](super::super::suffix_convolutions), truncated to the
/// positions that carry information.
///
/// SPACE SAVING:
///
/// `suffix_convolutions` produces `n + 1` tables for an `n`-ary node, but
/// only positions `1..n - 1` are stored, so `tables[j]` is position `j + 1`:
/// - Position `n` is the empty product `{0: 1}`, and position `n - 1`
///   convolves the last child against it, so it is a verbatim copy of that
///   child's histogram. Both are reconstructed on read.
/// - Position `0` is only ever read at the current layer's total, to sum the
///   key's count. It is computed on the fly and never stored.
///
/// In an e-graph of mostly binary nodes this is the difference between three
/// tables per node and none.
type SuffixTables<K> = HashMap<K, Vec<Vec<HashMap<usize, BigUint>>>>;

/// Size-layered counting over e-class keys or current/previous class pairs.
/// Each key contains nodes represented by their child keys.
pub struct LayeredDp<K> {
    /// Per key, per node: canonical child keys, aligned with node order.
    children_of: HashMap<K, Vec<Vec<K>>>,
    /// Largest size computed per key; unbudgeted keys are skipped.
    budgets: HashMap<K, usize>,
    /// Per-key, per-node suffix tables, extended one total per layer.
    suffix: SuffixTables<K>,
    /// Per key: size -> count histogram. Zero counts are never stored.
    data: HashMap<K, HashMap<usize, BigUint>>,
    /// The last completed layer.
    size: usize,
}

impl<K: Copy + Eq + Hash> LayeredDp<K> {
    /// Every budgeted key must occur in `children_of`; unbudgeted children
    /// have no terms.
    pub fn new(children_of: HashMap<K, Vec<Vec<K>>>, budgets: HashMap<K, usize>) -> Self {
        let suffix = budgets
            .keys()
            .map(|&k| {
                let tables = children_of[&k]
                    .iter()
                    // The first and the two trailing positions stay implicit,
                    // so an `n`-ary node keeps `n - 2` tables and a node of
                    // arity below three none.
                    .map(|children| vec![HashMap::default(); children.len().saturating_sub(2)])
                    .collect();
                (k, tables)
            })
            .collect();

        Self {
            children_of,
            budgets,
            suffix,
            data: HashMap::default(),
            size: 0,
        }
    }

    /// Complete and return the next size layer. Counts through it are final.
    pub fn step(&mut self) -> usize {
        self.size += 1;
        let size = self.size;
        // Children of a size-`size` term share this budget; it is also the
        // single new total the suffix tables gain this layer.
        let total = size - 1;

        let Self {
            children_of,
            budgets,
            suffix,
            data,
            ..
        } = self;

        // Extend the stored suffix tables by `total` and sum each key's count
        // at `size`: the number of ways any of its nodes fills its children
        // with `total`, i.e. the sum over its nodes of suffix position 0.
        // Subterm sizes are >= 1, so every part of `total` is <= size - 1:
        // exactly the histogram entries that already exist, and those are
        // final. For the same reason the `total` entry inserted into
        // `tables[j + 1]` in this very loop can never feed into `tables[j]`.
        // This layer's counts are collected and only land in `data` after
        // the loop.
        let mut layer = Vec::new();
        for (&k, _) in budgets.iter().filter(|&(_, &budget)| size <= budget) {
            let per_node = suffix.get_mut(&k).unwrap();
            let mut count = BigUint::ZERO;
            for (children, tables) in children_of[&k].iter().zip(per_node.iter_mut()) {
                match children.as_slice() {
                    // A leaf is one term of size one, with nothing to fill.
                    [] => {
                        if total == 0 {
                            count += 1u32;
                        }
                    }
                    // A lone child takes the whole total itself.
                    [only] => {
                        if let Some(c) = data.get(only).and_then(|hist| hist.get(&total)) {
                            count += c;
                        }
                    }
                    [first, .., last] => {
                        // `tables[j]` is position `j + 1`, filled by child
                        // `j + 1` against position `j + 2`.
                        for j in (0..tables.len()).rev() {
                            let Some(child_hist) = data.get(&children[j + 1]) else {
                                continue;
                            };
                            let (head, tail) = tables.split_at_mut(j + 1);
                            // `tail` is empty exactly when position `j + 2`
                            // is the implicit last-child one; read that
                            // child's histogram directly. Truncating it at
                            // this key's budget would change nothing: every
                            // part of `total` is <= `total`.
                            let Some(rest) = tail.first().or_else(|| data.get(last)) else {
                                continue;
                            };
                            let c = convolve_entry(child_hist, rest, total);
                            if !c.is_zero() {
                                head[j].insert(total, c);
                            }
                        }
                        // Position 0, only needed at `total`. Position 1 is
                        // the last child itself for a binary node.
                        let rest = tables.first().or_else(|| data.get(last));
                        if let (Some(first_hist), Some(rest)) = (data.get(first), rest) {
                            count += convolve_entry(first_hist, rest, total);
                        }
                    }
                }
            }
            if !count.is_zero() {
                layer.push((k, count));
            }
        }
        for (k, count) in layer {
            data.entry(k).or_default().insert(size, count);
        }

        size
    }

    #[must_use]
    pub const fn data(&self) -> &HashMap<K, HashMap<usize, BigUint>> {
        &self.data
    }

    #[must_use]
    pub const fn budgets(&self) -> &HashMap<K, usize> {
        &self.budgets
    }

    /// The DP's working tables, in the truncated [`SuffixTables`] layout.
    #[cfg(test)]
    pub const fn suffix(&self) -> &SuffixTables<K> {
        &self.suffix
    }

    /// Consume the DP, returning the histograms and dropping the suffix
    /// tables.
    pub fn into_data(self) -> HashMap<K, HashMap<usize, BigUint>> {
        self.data
    }
}
