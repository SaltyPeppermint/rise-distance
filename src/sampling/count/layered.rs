//! Size-layered term counting.
//!
//! The number of distinct terms of exactly `size` extractable from an
//! e-class depends only on child counts at sizes *strictly below* `size`:
//! an e-node contributes 1 to the size, so its children share a budget of
//! `size - 1`. Indexed by size, the dependency relation is therefore acyclic
//! even on cyclic e-graphs, and one pass per size layer computes every count
//! exactly once — no fixpoint iteration. (This replaced a worklist fixpoint
//! that re-convolved entire histograms every time a class on a cycle gained
//! an entry, costing up to `limit` full recomputations per class.)
//!
//! The layer loop itself is generic over the key type ([`LayeredDp`]): the
//! same argument applies verbatim to the joint counts over
//! `(curr_class, prev_class)` pairs, which reuse this kernel with matches as
//! nodes (see `novel.rs` and `docs/incremental_probe.md`). Stepping one layer
//! at a time also lets the fingerprint probe stop as soon as it has seen
//! enough novel sizes at the root.
//!
//! When `roots` are given, the computation is further restricted to what
//! extractions of size <= `limit` from a root can touch: only classes
//! reachable from the roots are counted, each capped at its *budget* — the
//! largest size a subterm rooted at that class can take in any such
//! extraction. Deep classes then carry a handful of histogram entries
//! instead of ~`limit`, which shrinks every convolution touching them.

use std::hash::Hash;

use egg::{Analysis, AstSize, EGraph, Id, Language};
use hashbrown::HashMap;

use crate::Counter;
use crate::analysis::semilattice::SemiLatticeAnalysis;
use crate::utils::UniqueQueue;

/// Histograms and per-node suffix convolution tables from one counting run.
pub struct CountData<C> {
    /// Per canonical class: term-size histogram (size -> count of distinct
    /// terms). Classes without any term within their budget are absent.
    pub data: HashMap<Id, HashMap<usize, C>>,
    /// Per canonical class in `data`, per node index:
    /// `suffix[class][node][i]` maps a total to the number of ways children
    /// `i..` of that node can be filled with subterm sizes of that total.
    /// Same shape as [`suffix_convolutions`](super::suffix_convolutions),
    /// with `suffix[class][node][arity]` the empty-product base `{0: 1}`.
    pub suffix: HashMap<Id, Vec<Vec<HashMap<usize, C>>>>,
}

/// Count the distinct terms extractable from each e-class, by size.
///
/// With `roots: None` every class is counted up to `limit`; with roots, see
/// the module docs: classes unreachable from the roots are skipped and the
/// rest are capped at their budget, which keeps histograms and suffix tables
/// exact for every query a root-driven consumer can make.
pub fn count_terms<C, L, N>(
    limit: usize,
    egraph: &EGraph<L, N>,
    roots: Option<&[Id]>,
) -> CountData<C>
where
    C: Counter,
    L: Language,
    N: Analysis<L>,
{
    let mut dp = plain_dp(limit, egraph, roots);
    for _ in 0..limit {
        dp.step();
    }
    let (data, mut suffix) = dp.into_parts();

    // Suffix tables are only read for classes one can sample from, i.e.
    // classes with a nonempty histogram.
    suffix.retain(|id, _| data.contains_key(id));

    CountData { data, suffix }
}

/// The plain-count instantiation of [`LayeredDp`]: keys are e-class ids,
/// nodes are the class's e-nodes, budgets from `roots` as in
/// [`count_terms`]. Not stepped yet — callers drive the layers themselves.
pub fn plain_dp<C, L, N>(
    limit: usize,
    egraph: &EGraph<L, N>,
    roots: Option<&[Id]>,
) -> LayeredDp<Id, C>
where
    C: Counter,
    L: Language,
    N: Analysis<L>,
{
    assert!(egraph.clean);

    let budgets = match roots {
        Some(roots) => {
            let mut min_sizes = HashMap::new();
            AstSize.one_shot_analysis(egraph, &mut min_sizes);
            class_budgets(egraph, roots, limit, &min_sizes)
        }
        None => egraph.classes().map(|class| (class.id, limit)).collect(),
    };

    // Canonicalized children per (class, node), aligned with `nodes` order.
    let children_of: HashMap<Id, Vec<Vec<Id>>> = budgets
        .keys()
        .map(|&id| {
            let per_node = egraph[id]
                .nodes
                .iter()
                .map(|node| node.children().iter().map(|&c| egraph.find(c)).collect())
                .collect();
            (id, per_node)
        })
        .collect();

    LayeredDp::new(children_of, budgets)
}

/// Per key: size -> count histogram.
type Histograms<K, C> = HashMap<K, HashMap<usize, C>>;

/// Per key, per node: suffix convolution tables in the shape of
/// [`suffix_convolutions`](super::suffix_convolutions).
type SuffixTables<K, C> = HashMap<K, Vec<Vec<HashMap<usize, C>>>>;

/// The size-layered counting kernel, generic over the key type: e-class ids
/// for plain counts, `(curr_class, prev_class)` pairs for joint counts. Per
/// key there is a list of *nodes* (e-nodes resp. matches), each a list of
/// child keys; a key's count at `size` sums, over its nodes, the ways to
/// fill the node's children with subterm sizes totalling `size - 1`.
/// [`step`](Self::step) completes one size layer; every histogram entry at
/// sizes <= the completed layer is final.
pub struct LayeredDp<K, C> {
    /// Per key, per node: canonical child keys, aligned with node order.
    children_of: HashMap<K, Vec<Vec<K>>>,
    /// Per key: the largest size worth computing. Keys of `children_of`
    /// without a budget are skipped entirely.
    budgets: HashMap<K, usize>,
    /// Per budgeted key, per node: suffix tables, grown by one total per
    /// layer.
    suffix: SuffixTables<K, C>,
    /// Per key: size -> count histogram. Zero counts are never stored.
    data: Histograms<K, C>,
    /// The last completed layer.
    size: usize,
}

impl<K: Copy + Eq + Hash, C: Counter> LayeredDp<K, C> {
    /// `children_of` must contain every key of `budgets`; child keys missing
    /// from `budgets` are treated as having no terms.
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

    /// Complete the next size layer and return it. Once this returns `s`,
    /// every `data` entry at sizes <= `s` is final.
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

/// Largest subterm size each class can take in any extraction of size <=
/// `limit` from one of `roots`. Through a parent node, a child may use the
/// parent's budget minus 1 (the node itself) minus the minimal subterm sizes
/// of its siblings; the budget is the maximum of that over all parent
/// positions. Worklist relaxation: budgets only grow, are bounded by
/// `limit`, and strictly shrink along any cycle, so this terminates. Classes
/// absent from the result cannot appear in any such extraction.
fn class_budgets<L, N>(
    egraph: &EGraph<L, N>,
    roots: &[Id],
    limit: usize,
    min_sizes: &HashMap<Id, usize>,
) -> HashMap<Id, usize>
where
    L: Language,
    N: Analysis<L>,
{
    let mut budgets: HashMap<Id, usize> = roots
        .iter()
        .map(|&root| (egraph.find(root), limit))
        .collect();
    let mut pending: UniqueQueue<Id> = budgets.keys().copied().collect();

    while let Some(id) = pending.pop() {
        let Some(children_total) = budgets[&id].checked_sub(1) else {
            continue;
        };
        for node in &egraph[id].nodes {
            let children = node.children();
            let mins_sum: usize = children
                .iter()
                .map(|&child| min_sizes[&egraph.find(child)])
                .sum();
            if mins_sum > children_total {
                // The node cannot fit within this class's budget at all.
                continue;
            }
            for &child in children {
                let child = egraph.find(child);
                // The child gets whatever remains when its siblings are as
                // small as possible.
                let child_budget = children_total - (mins_sum - min_sizes[&child]);
                if budgets.get(&child).is_none_or(|&b| b < child_budget) {
                    budgets.insert(child, child_budget);
                    pending.insert(child);
                }
            }
        }
    }
    budgets
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, SymbolLang};
    use num::BigUint;

    use super::super::suffix_convolutions;
    use super::*;

    #[test]
    fn simple_term_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        let data = count_terms::<BigUint, _, _>(10, &egraph, None).data;
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

        let data = count_terms::<BigUint, _, _>(10, &egraph, None).data;

        let root_data = &data[&egraph.find(apb)];
        assert_eq!(root_data[&5], 16usize.into());
    }

    #[test]
    fn rooted_matches_unrooted_at_the_root() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));
        let root = egraph.add(SymbolLang::new("f", vec![apb]));

        egraph.union(a, apb);
        egraph.rebuild();

        let full = count_terms::<BigUint, _, _>(11, &egraph, None);
        let rooted = count_terms::<BigUint, _, _>(11, &egraph, Some(&[root]));

        let canon = egraph.find(root);
        assert_eq!(full.data[&canon], rooted.data[&canon]);
        assert_eq!(full.suffix[&canon], rooted.suffix[&canon]);
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
        let rooted = count_terms::<BigUint, _, _>(limit, &egraph, Some(&[root]));

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

        // Unrooted counts x all the way to the limit.
        let full = count_terms::<BigUint, _, _>(limit, &egraph, None);
        assert_eq!(full.data[&egraph.find(a)].len(), limit);
        assert!(full.data.contains_key(&egraph.find(z)));
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

        let rooted = count_terms::<BigUint, _, _>(10, &egraph, Some(&[root]));

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
        let _gab = egraph.add(SymbolLang::new("g", vec![apb, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        let limit = 9;
        let result = count_terms::<BigUint, _, _>(limit, &egraph, None);

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
                let expected = suffix_convolutions(&histograms, limit - 1);
                assert_eq!(tables, &expected);
            }
        }
    }
}
