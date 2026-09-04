//! Size-layered term counting.
//!
//! Because an e-node costs one, counts at a size depend only on smaller sizes.
//! [`LayeredDp`] therefore handles cyclic e-graphs without a fixpoint. Root
//! budgets restrict counting to states usable below the requested size limit.
//! See `docs/counting/layered_counting.md`.

use std::hash::Hash;

use egg::{Analysis, EGraph, Id, Language};
use hashbrown::HashMap;
use num::{BigUint, Zero};

use crate::candidates::convolve_entry;
use crate::candidates::count::budgets::RootBudgets;

/// Count distinct terms within pre-established root budgets.
///
/// The suffix tables are the DP's working state and die with it: they dwarf
/// the histograms, and drawers rederive the child-size splits they need on the
/// fly from these histograms instead.
pub(crate) fn count_histograms_rooted<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    rooted: &RootBudgets,
) -> Histograms<Id, BigUint> {
    let mut dp = plain_dp_rooted(egraph, rooted);
    for _ in 0..rooted.limit() {
        dp.step();
    }
    dp.into_data()
}

/// Create an unstepped plain DP for the root budgets.
pub(crate) fn plain_dp_rooted<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    rooted: &RootBudgets,
) -> LayeredDp<Id> {
    assert!(egraph.clean);
    let children_of = plain_children_of(egraph, rooted.budgets().keys().copied());
    LayeredDp::new(children_of, rooted.budgets().clone())
}

fn plain_children_of<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    ids: impl Iterator<Item = Id>,
) -> HashMap<Id, Vec<Vec<Id>>> {
    ids.map(|id| {
        let per_node = egraph[id]
            .nodes
            .iter()
            .map(|node| node.children().iter().map(|&c| egraph.find(c)).collect())
            .collect();
        (id, per_node)
    })
    .collect()
}

// ============================================================================
// Exact root-size scan.
// ============================================================================

/// Find the first `stop_after` root sizes with terms within `rooted`.
///
/// The plain analogue of `find_novel_root_sizes`: with no previous boundary to
/// subtract, every size the root can extract at counts.
pub(crate) fn find_plain_root_sizes<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    root: Id,
    stop_after: usize,
    rooted: &RootBudgets,
) -> Vec<usize> {
    let root = egraph.find(root);
    let mut plain = plain_dp_rooted(egraph, rooted);

    let mut sizes = Vec::new();
    for _ in 0..rooted.limit() {
        let size = plain.step();

        // Final as of this layer. Zero-count entries are absent and read as 0.
        let count = plain.data().get(&root).and_then(|hist| hist.get(&size));
        if count.is_some_and(|count| *count != BigUint::ZERO) {
            sizes.push(size);
            if sizes.len() >= stop_after {
                break;
            }
        }
    }
    sizes
}

/// Per key: size -> count histogram.
pub type Histograms<K, C> = HashMap<K, HashMap<usize, C>>;

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
    data: Histograms<K, BigUint>,
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
    pub const fn data(&self) -> &Histograms<K, BigUint> {
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
    pub fn into_data(self) -> Histograms<K, BigUint> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, SymbolLang};

    use crate::candidates::count::budgets::root_budgets;

    use super::super::super::suffix_convolutions;
    use super::*;

    fn rooted_counts(
        egraph: &EGraph<SymbolLang, ()>,
        root: Id,
        limit: usize,
    ) -> Histograms<Id, BigUint> {
        let budgets = root_budgets(egraph, root, limit);
        count_histograms_rooted(egraph, &budgets)
    }

    #[test]
    fn simple_term_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        let data = rooted_counts(&egraph, apb, 10);
        let root_data = &data[&egraph.find(apb)];

        assert_eq!(root_data[&5], 1usize.into());
    }

    #[test]
    fn slightly_complicated_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();
        egraph.union(b, apb);
        egraph.rebuild();

        let data = rooted_counts(&egraph, apb, 10);

        let root_data = &data[&egraph.find(apb)];
        assert_eq!(root_data[&5], 16usize.into());
    }

    #[test]
    fn rooted_caps_deep_classes_and_skips_unreachable() {
        // x = {a, f(x)} (cyclic), root = {g(x)}, z unreachable.
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let fa = egraph.add(SymbolLang::new("f", vec![a]));
        let root = egraph.add(SymbolLang::new("g", vec![a]));
        let z = egraph.add(SymbolLang::leaf("z"));

        egraph.union(a, fa);
        egraph.rebuild();

        let limit = 6;
        let rooted = rooted_counts(&egraph, root, limit);

        // x can spend at most limit - 1 through g; one term per size.
        let x_hist = &rooted[&egraph.find(a)];
        let mut x_sizes = x_hist.keys().copied().collect::<Vec<_>>();
        x_sizes.sort_unstable();
        assert_eq!(x_sizes, (1..limit).collect::<Vec<_>>());

        let root_hist = &rooted[&egraph.find(root)];
        let mut root_sizes = root_hist.keys().copied().collect::<Vec<_>>();
        root_sizes.sort_unstable();
        assert_eq!(root_sizes, (2..=limit).collect::<Vec<_>>());

        assert!(!rooted.contains_key(&egraph.find(z)));
    }

    #[test]
    fn sibling_minimums_tighten_budgets() {
        // root = +(x, y) with x = {a, f(x)} (cyclic) and min size 3 for y:
        // x's budget is limit - 1 (the + node) - 3 (the smallest y) = 6.
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let fa = egraph.add(SymbolLang::new("f", vec![a]));
        let b = egraph.add(SymbolLang::leaf("b"));
        let fb = egraph.add(SymbolLang::new("f", vec![b]));
        let ffb = egraph.add(SymbolLang::new("f", vec![fb]));
        let root = egraph.add(SymbolLang::new("+", vec![a, ffb]));

        egraph.union(a, fa);
        egraph.rebuild();

        let rooted = rooted_counts(&egraph, root, 10);

        let mut x_sizes = rooted[&egraph.find(a)].keys().copied().collect::<Vec<_>>();
        x_sizes.sort_unstable();
        assert_eq!(x_sizes, (1..=6).collect::<Vec<_>>());

        let mut root_sizes = rooted[&egraph.find(root)]
            .keys()
            .copied()
            .collect::<Vec<_>>();
        root_sizes.sort_unstable();
        assert_eq!(root_sizes, (5..=10).collect::<Vec<_>>());
    }

    #[test]
    fn suffix_tables_match_suffix_convolutions() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));
        let gab = egraph.add(SymbolLang::new("g", vec![apb, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        let limit = 9;
        let budgets = root_budgets(&egraph, gab, limit);
        let mut dp = plain_dp_rooted(&egraph, &budgets);
        for _ in 0..budgets.limit() {
            dp.step();
        }

        for (&id, per_node) in dp.suffix() {
            for (node, tables) in egraph[id].nodes.iter().zip(per_node) {
                let children = node
                    .children()
                    .iter()
                    .map(|&c| egraph.find(c))
                    .collect::<Vec<_>>();
                let histograms = children
                    .iter()
                    .map(|c| dp.data().get(c).cloned().unwrap_or_default())
                    .collect::<Vec<_>>();
                let budget = budgets.budget(id).unwrap() - 1;
                let expected = suffix_convolutions(&histograms, budget);

                // Only positions `0..n - 1` are stored; the last two are
                // implicit, and drawers rebuild them with
                // `suffix_convolutions` over the same histograms.
                assert_eq!(tables.len(), children.len().saturating_sub(1));
                assert_eq!(tables.as_slice(), &expected[..tables.len()]);
            }
        }
    }
}
