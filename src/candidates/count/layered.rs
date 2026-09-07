//! Size-layered term counting.
//!
//! Because an e-node costs one, counts at a size depend only on smaller sizes.
//! [`LayeredDp`] therefore handles cyclic e-graphs without a fixpoint. Root
//! budgets restrict counting to states usable below the requested size limit.
//! See `docs/counting/layered_counting.md`.

use std::hash::Hash;

use hashbrown::HashMap;
use num::{BigUint, Zero};

use crate::candidates::convolve_entry;

/// Per key, per node: suffix convolution tables in the shape of
/// [`suffix_convolutions`](super::super::suffix_convolutions), truncated to the
/// positions that carry information.
///
/// SPACE SAVING:
///
/// `suffix_convolutions` produces `n + 1` tables for an `n`-ary node, but the
/// last two are never worth materializing: position `n` is the empty product
/// `{0: 1}`, and position `n - 1` convolves the last child against that empty
/// product, so it is a verbatim copy of that child's histogram. Only positions
/// `0..n - 1` are stored. We have a lot of nodes for which this kicks in.
///
/// Both implicit positions are reconstructed on read. In an e-graph of mostly
/// binary nodes this is the difference between three tables per node and one.
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
                    // The two trailing positions stay implicit, so an `n`-ary
                    // node keeps `n - 1` tables and a leaf or unary node none.
                    .map(|children| vec![HashMap::new(); children.len().saturating_sub(1)])
                    .collect();
                (k, tables)
            })
            .collect();

        Self {
            children_of,
            budgets,
            suffix,
            data: HashMap::new(),
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

        // Extend the suffix tables by `total`. Subterm sizes are >= 1, so
        // every part of `total` is <= size - 1: exactly the histogram
        // entries that already exist, and those are final. For the same
        // reason the `total` entry inserted into `tables[i + 1]` in this
        // very loop can never feed into `tables[i]`, and `data` still holds
        // nothing at `size` — this layer's histograms land below.
        for (&k, _) in budgets.iter().filter(|&(_, &budget)| size <= budget) {
            let per_node = suffix.get_mut(&k).unwrap();
            for (children, tables) in children_of[&k].iter().zip(per_node.iter_mut()) {
                // A leaf stores no tables and has no last child to stand in
                // for the implicit position.
                let Some(last) = children.last() else {
                    continue;
                };
                for i in (0..tables.len()).rev() {
                    let Some(child_hist) = data.get(&children[i]) else {
                        continue;
                    };
                    let (head, tail) = tables.split_at_mut(i + 1);
                    // `tail` is empty exactly when position `i + 1` is the
                    // implicit last-child one; read that child's histogram
                    // directly. Truncating it at this key's budget would
                    // change nothing: every part of `total` is <= `total`.
                    let Some(rest) = tail.first().or_else(|| data.get(last)) else {
                        continue;
                    };
                    let count = convolve_entry(child_hist, rest, total);
                    if count != BigUint::ZERO {
                        head[i].insert(total, count);
                    }
                }
            }
        }

        // A key's count at `size` is the number of ways any of its nodes
        // fills its children with `total`, i.e. the sum over its nodes of
        // suffix position 0 implicit for arity below two.
        for (&k, _) in budgets.iter().filter(|&(_, &budget)| size <= budget) {
            let count = children_of[&k]
                .iter()
                .zip(&suffix[&k])
                .filter_map(|(children, tables)| match children.as_slice() {
                    // A leaf is one term of size one, with nothing to fill.
                    [] => (total == 0).then_some(&BigUint::ONE),
                    // A lone child takes the whole total itself.
                    [only] => data.get(only)?.get(&total),
                    _ => tables[0].get(&total),
                })
                .sum::<BigUint>();
            if !count.is_zero() {
                data.entry(k).or_default().insert(size, count);
            }
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
