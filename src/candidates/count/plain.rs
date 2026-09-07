use egg::{Analysis, EGraph, Id, Language};
use hashbrown::HashMap;
use num::{BigUint, Zero};

use crate::candidates::count::budgets::RootBudgets;
use crate::candidates::count::layered::LayeredDp;

/// Count distinct terms within pre-established root budgets.
pub(crate) fn count_histograms_rooted<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    rooted: &RootBudgets,
) -> HashMap<Id, HashMap<usize, BigUint>> {
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
    let children_of = rooted
        .budgets()
        .keys()
        .map(|id| {
            let per_node = egraph[*id]
                .nodes
                .iter()
                .map(|node| node.children().iter().map(|&c| egraph.find(c)).collect())
                .collect();
            (*id, per_node)
        })
        .collect();
    LayeredDp::new(children_of, rooted.budgets().clone())
}

// ============================================================================
// Exact root-size scan.
// ============================================================================

/// Find the smallest root size that makes at least `min_extractable` terms
/// available within `rooted`.
///
/// The plain analogue of `find_novel_root_sizes`: with no previous boundary to
/// subtract, every term the root can extract counts toward the threshold.
///
/// # Errors
///
/// Returns the terms found when `rooted` is exhausted below `min_extractable`.
pub(crate) fn find_plain_root_size<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    root: Id,
    min_extractable: usize,
    rooted: &RootBudgets,
) -> Result<usize, BigUint> {
    let root = egraph.find(root);
    let mut plain = plain_dp_rooted(egraph, rooted);

    let mut term_count = BigUint::ZERO;
    for _ in 0..rooted.limit() {
        let size = plain.step();

        // Final as of this layer. Zero-count entries are absent and read as 0.
        let count = plain.data().get(&root).and_then(|hist| hist.get(&size));
        if let Some(count) = count.filter(|count| !count.is_zero()) {
            term_count += count;
            if term_count >= min_extractable.into() {
                return Ok(size);
            }
        }
    }
    Err(term_count)
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, SymbolLang};

    use super::super::super::suffix_convolutions;
    use super::*;

    fn rooted_counts(
        egraph: &EGraph<SymbolLang, ()>,
        root: Id,
        limit: usize,
    ) -> HashMap<Id, HashMap<usize, BigUint>> {
        let budgets = RootBudgets::of_root(egraph, root, limit);
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
        let budgets = RootBudgets::of_root(&egraph, gab, limit);
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
