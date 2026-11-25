use egg::{Analysis, EGraph, Id, Language};
use num::{BigUint, Zero};

use super::budgets::RootBudgets;
use super::layered::LayeredDp;
use crate::utils::HashMap;

/// Per-class size histograms keyed by canonical e-class id.
pub(crate) type ClassHistograms = HashMap<Id, HashMap<usize, BigUint>>;

/// Create an unstepped whole-graph DP for the root budgets.
pub(super) fn dp<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    budgets: &RootBudgets,
) -> LayeredDp<Id> {
    assert!(egraph.clean);
    let children_of = budgets
        .budgets()
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
    LayeredDp::new(children_of, budgets.budgets().clone())
}

/// Find the smallest root size `<= cap` that makes at least `min_extractable`
/// terms available, returning `(max_size, counts)` pruned to `max_size`'s root
/// budgets in a single DP pass.
///
/// # Errors
///
/// Returns the total terms found when `cap` is exhausted below `min_extractable`.
pub(crate) fn count<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    root: Id,
    cap: usize,
    min_extractable: usize,
) -> Result<(usize, ClassHistograms), BigUint> {
    let root = egraph.find(root);
    let cap_budgets = RootBudgets::new(egraph, root, cap);
    let mut dp = dp(egraph, &cap_budgets);

    let threshold = BigUint::from(min_extractable);
    let mut term_count = BigUint::ZERO;
    for _ in 0..cap_budgets.limit() {
        let size = dp.step();
        if let Some(count) = dp.data().get(&root).and_then(|hist| hist.get(&size))
            && !count.is_zero()
        {
            term_count += count;
            if term_count >= threshold {
                let mut counts = dp.into_data();
                RootBudgets::new(egraph, root, size).prune_histograms(&mut counts, |c| c);
                return Ok((size, counts));
            }
        }
    }
    Err(term_count)
}

#[cfg(test)]
mod tests {
    use egg::SymbolLang;

    use super::*;
    use crate::sampling;

    #[test]
    fn simple_term_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        // Sizes 1, 3, 5 each have 1 term, so 3 terms reaches size 5.
        let (used_size, data) = count(&egraph, apb, 10, 3).unwrap();
        assert_eq!(used_size, 5);
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

        // Sizes 1 (2), 3 (4), 5 (16) sum to 22 terms through size 5.
        let (used_size, data) = count(&egraph, apb, 10, 22).unwrap();
        assert_eq!(used_size, 5);

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
        // Root has 1 term at each size 2..=6 (5 total).
        let (used_size, rooted) = count(&egraph, root, limit, 5).unwrap();
        assert_eq!(used_size, limit);

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

        // Root has 1 term at each size 5..=10 (6 total).
        let (used_size, rooted) = count(&egraph, root, 10, 6).unwrap();
        assert_eq!(used_size, 10);

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
        // A ternary node is the only kind that stores a table.
        let root = egraph.add(SymbolLang::new("h", vec![gab, apb, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        let limit = 9;
        let budgets = RootBudgets::new(&egraph, root, limit);
        let mut dp = dp(&egraph, &budgets);
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
                let expected = sampling::suffix_convolutions(&histograms, budget);

                // Only positions `1..n - 1` are stored; the first and the
                // last two are implicit, and drawers rebuild them with
                // `suffix_convolutions` over the same histograms.
                assert_eq!(tables.len(), children.len().saturating_sub(2));
                assert_eq!(tables.as_slice(), &expected[1..=tables.len()]);
            }
        }
    }
}
