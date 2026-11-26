//! Size-layered term counting.
//!
//! Because an e-node costs one, counts at a size depend only on smaller sizes,
//! handling cyclic e-graphs without a fixpoint. Root budgets restrict counting
//! to states usable below the requested size limit.
//! See `docs/counting/layered_counting.md`.

mod budgets;
pub(super) mod novel;
pub(super) mod whole;

use std::hash::Hash;

use num_bigint::BigUint;

use crate::sampling;
use crate::utils::HashMap;

/// Size-layered histograms plus a sparse suffix convolution cache populated
/// only for nodes with arity >= 3.
///
/// For an `n`-ary node (`n >= 3`), `suffix[node_key]` stores suffix positions
/// `1..n - 1` (`n - 2` tables), extended by one total per layer. Nodes of
/// arity <= 2 need no suffix tables, so `suffix` stays empty for binary-or-less
/// languages.
pub(super) struct LayerTable<K, N> {
    pub(super) data: HashMap<K, HashMap<usize, BigUint>>,
    pub(super) suffix: HashMap<N, Vec<HashMap<usize, BigUint>>>,
}

impl<K, N> LayerTable<K, N> {
    pub(super) fn new() -> Self {
        Self {
            data: HashMap::default(),
            suffix: HashMap::default(),
        }
    }

    /// Consume the table, returning the histograms and dropping any suffix cache.
    pub(super) fn into_data(self) -> HashMap<K, HashMap<usize, BigUint>> {
        self.data
    }
}

impl<K: Eq + Hash, N: Eq + Hash> LayerTable<K, N> {
    /// Number of ways `children` can sum to `total`, or `None` if zero,
    /// extending the sparse suffix cache when `children.len() >= 3`.
    ///
    /// Subterm sizes are >= 1, so every part of `total` is <= `total` and
    /// already finalized in `self.data` (and strictly below `total` in any
    /// later suffix table). Because `self.suffix` is extended incrementally in
    /// place for arity >= 3, callers must invoke `node_count` at every
    /// consecutive `total` (`2, 3, ...`) in ascending order without gaps for
    /// each active `node_key`.
    pub(super) fn node_count(
        &mut self,
        node_key: N,
        children: &[K],
        total: usize,
    ) -> Option<BigUint> {
        let count = match children {
            [] => return (total == 0).then_some(BigUint::ONE),
            [only] => {
                return self
                    .data
                    .get(only)
                    .and_then(|hist| hist.get(&total))
                    .cloned();
            }
            [first, last] => {
                sampling::convolve_entry(self.data.get(first)?, self.data.get(last)?, total)
            }
            [first, .., last] => {
                // Every stored suffix covers at least two children, so no entry
                // can be non-zero for `total < 2`.
                if total < 2 {
                    return None;
                }
                let tables = self
                    .suffix
                    .entry(node_key)
                    .or_insert_with(|| vec![HashMap::default(); children.len() - 2]);
                // `tables[j]` is suffix position `j + 1` (children `j + 1..n`).
                for j in (0..tables.len()).rev() {
                    let Some(child_hist) = self.data.get(&children[j + 1]) else {
                        continue;
                    };
                    let (head, tail) = tables.split_at_mut(j + 1);
                    let Some(rest) = tail.first().or_else(|| self.data.get(last)) else {
                        continue;
                    };
                    let c = sampling::convolve_entry(child_hist, rest, total);
                    if c != BigUint::ZERO {
                        head[j].insert(total, c);
                    }
                }
                sampling::convolve_entry(self.data.get(first)?, &tables[0], total)
            }
        };
        (count != BigUint::ZERO).then_some(count)
    }
}
