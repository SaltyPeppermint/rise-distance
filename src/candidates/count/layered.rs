//! Size-layered term counting.
//!
//! Because an e-node costs one, counts at a size depend only on smaller sizes.
//! [`LayeredDp`] therefore handles cyclic e-graphs without a fixpoint. Root
//! budgets restrict counting to states usable below the requested size limit.
//! See `docs/counting/layered_counting.md`.

use std::hash::Hash;

use egg::{Analysis, EGraph, Id, Language};
use hashbrown::HashMap;
use num::BigUint;

use crate::Counter;
use crate::candidates::count::budgets::RootBudgets;

/// Histograms and per-node suffix convolutions from one counting run.
#[derive(Debug)]
pub struct CountData<C> {
    /// Distinct-term counts by size and canonical class.
    pub(crate) data: HashMap<Id, HashMap<usize, C>>,
    /// Per-class, per-node suffix convolutions used to split child sizes.
    pub(crate) suffix: HashMap<Id, Vec<Vec<HashMap<usize, C>>>>,
}

/// Count distinct terms within pre-established root budgets.
pub(crate) fn count_terms_rooted<C, L, N>(
    egraph: &EGraph<L, N>,
    rooted: &RootBudgets,
) -> CountData<C>
where
    C: Counter,
    L: Language,
    N: Analysis<L>,
{
    let mut dp = plain_dp_rooted(egraph, rooted);
    for _ in 0..rooted.limit() {
        dp.step();
    }
    let (data, mut suffix) = dp.into_parts();
    suffix.retain(|id, _| data.contains_key(id));
    CountData { data, suffix }
}

/// Create an unstepped plain DP for the root budgets.
pub(crate) fn plain_dp_rooted<C, L, N>(
    egraph: &EGraph<L, N>,
    rooted: &RootBudgets,
) -> LayeredDp<Id, C>
where
    C: Counter,
    L: Language,
    N: Analysis<L>,
{
    assert!(egraph.clean);
    let children_of = plain_children_of(egraph, rooted.budgets().keys().copied());
    LayeredDp::new(children_of, rooted.budgets().clone())
}

fn plain_children_of<L, N>(
    egraph: &EGraph<L, N>,
    ids: impl Iterator<Item = Id>,
) -> HashMap<Id, Vec<Vec<Id>>>
where
    L: Language,
    N: Analysis<L>,
{
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
    let mut plain: LayeredDp<Id, BigUint> = plain_dp_rooted(egraph, rooted);

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
type Histograms<K, C> = HashMap<K, HashMap<usize, C>>;

/// Per key, per node: suffix convolution tables in the shape of
/// [`suffix_convolutions`](super::suffix_convolutions).
type SuffixTables<K, C> = HashMap<K, Vec<Vec<HashMap<usize, C>>>>;

/// Size-layered counting over e-class keys or current/previous class pairs.
/// Each key contains nodes represented by their child keys.
pub struct LayeredDp<K, C> {
    /// Per key, per node: canonical child keys, aligned with node order.
    children_of: HashMap<K, Vec<Vec<K>>>,
    /// Largest size computed per key; unbudgeted keys are skipped.
    budgets: HashMap<K, usize>,
    /// Per-key, per-node suffix tables, extended one total per layer.
    suffix: SuffixTables<K, C>,
    /// Per key: size -> count histogram. Zero counts are never stored.
    data: Histograms<K, C>,
    /// The last completed layer.
    size: usize,
}

impl<K: Copy + Eq + Hash, C: Counter> LayeredDp<K, C> {
    /// Every budgeted key must occur in `children_of`; unbudgeted children
    /// have no terms.
    pub fn new(children_of: HashMap<K, Vec<Vec<K>>>, budgets: HashMap<K, usize>) -> Self {
        let suffix = budgets
            .keys()
            .map(|&k| {
                let tables = children_of[&k]
                    .iter()
                    .map(|children| {
                        let mut tables = vec![HashMap::new(); children.len() + 1];
                        tables[children.len()].insert(0, C::one());
                        tables
                    })
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
        // very loop can never feed into `tables[i]`.
        for (&k, &budget) in budgets.iter() {
            if size > budget {
                continue;
            }
            let per_node = suffix.get_mut(&k).unwrap();
            for (children, tables) in children_of[&k].iter().zip(per_node.iter_mut()) {
                for i in (0..children.len()).rev() {
                    let Some(child_hist) = data.get(&children[i]) else {
                        continue;
                    };
                    let (head, tail) = tables.split_at_mut(i + 1);
                    let count = convolve_entry(child_hist, &tail[0], total);
                    if count != C::zero() {
                        head[i].insert(total, count);
                    }
                }
            }
        }

        // A key's count at `size` is the number of ways any of its nodes
        // fills its children with `total`.
        for (&k, &budget) in budgets.iter() {
            if size > budget {
                continue;
            }
            let count = suffix[&k]
                .iter()
                .filter_map(|tables| tables[0].get(&total))
                .sum::<C>();
            if count != C::zero() {
                data.entry(k).or_default().insert(size, count);
            }
        }

        size
    }

    #[must_use]
    pub const fn data(&self) -> &Histograms<K, C> {
        &self.data
    }

    #[must_use]
    pub const fn budgets(&self) -> &HashMap<K, usize> {
        &self.budgets
    }

    /// Consume the DP, returning the histograms and suffix tables.
    pub fn into_parts(self) -> (Histograms<K, C>, SuffixTables<K, C>) {
        (self.data, self.suffix)
    }
}

/// The convolution of two histograms evaluated at exactly `total`:
/// `sum over a + b = total of hist(a) * rest(b)`, iterating the smaller map.
fn convolve_entry<C: Counter>(
    hist: &HashMap<usize, C>,
    rest: &HashMap<usize, C>,
    total: usize,
) -> C {
    let (outer, inner) = if hist.len() <= rest.len() {
        (hist, rest)
    } else {
        (rest, hist)
    };
    outer
        .iter()
        .filter_map(|(&a, count_a)| {
            let count_b = total.checked_sub(a).and_then(|b| inner.get(&b))?;
            Some(count_a.to_owned() * count_b)
        })
        .fold(C::zero(), |acc, c| acc + c)
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, SymbolLang};
    use num::BigUint;

    use crate::candidates::count::budgets::root_budgets;

    use super::super::super::suffix_convolutions;
    use super::*;

    fn rooted_counts(
        egraph: &EGraph<SymbolLang, ()>,
        root: Id,
        limit: usize,
    ) -> CountData<BigUint> {
        let budgets = root_budgets(egraph, root, limit);
        count_terms_rooted(egraph, &budgets)
    }

    #[test]
    fn simple_term_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        let data = rooted_counts(&egraph, apb, 10).data;
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

        let data = rooted_counts(&egraph, apb, 10).data;

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
        let x_hist = &rooted.data[&egraph.find(a)];
        let mut x_sizes = x_hist.keys().copied().collect::<Vec<_>>();
        x_sizes.sort_unstable();
        assert_eq!(x_sizes, (1..limit).collect::<Vec<_>>());

        let root_hist = &rooted.data[&egraph.find(root)];
        let mut root_sizes = root_hist.keys().copied().collect::<Vec<_>>();
        root_sizes.sort_unstable();
        assert_eq!(root_sizes, (2..=limit).collect::<Vec<_>>());

        assert!(!rooted.data.contains_key(&egraph.find(z)));
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

        let mut x_sizes = rooted.data[&egraph.find(a)]
            .keys()
            .copied()
            .collect::<Vec<_>>();
        x_sizes.sort_unstable();
        assert_eq!(x_sizes, (1..=6).collect::<Vec<_>>());

        let mut root_sizes = rooted.data[&egraph.find(root)]
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
        let result = count_terms_rooted::<BigUint, _, _>(&egraph, &budgets);

        for (&id, per_node) in &result.suffix {
            for (node, tables) in egraph[id].nodes.iter().zip(per_node) {
                let histograms = node
                    .children()
                    .iter()
                    .map(|&c| {
                        result
                            .data
                            .get(&egraph.find(c))
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect::<Vec<_>>();
                let expected = suffix_convolutions(&histograms, budgets.budget(id).unwrap() - 1);
                assert_eq!(tables, &expected);
            }
        }
    }
}
