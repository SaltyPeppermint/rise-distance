use egg::{Analysis, EGraph, Id, Language};
use num::{BigUint, Zero};

use super::LayerTable;
use super::budgets::{self, MinSizes, RootBudgets};
use crate::utils::HashMap;

/// Per-class size histograms keyed by canonical e-class id.
pub(crate) type ClassHistograms = HashMap<Id, HashMap<usize, BigUint>>;

/// Advance `whole` by one size layer directly over `egraph`.
pub(super) fn step<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    budgets: &RootBudgets<'_>,
    whole: &mut LayerTable<Id, (Id, usize)>,
    size: usize,
) {
    assert!(egraph.clean);
    let total = size - 1;
    let layer = budgets
        .budgets()
        .iter()
        .filter(|&(_, &budget)| size <= budget)
        .filter_map(|(&id, _)| {
            let count = egraph[id]
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| budgets.node_fits(egraph, id, node))
                .filter_map(|(node_idx, node)| {
                    debug_assert!(node.children().iter().all(|&c| egraph.find(c) == c));
                    whole.node_count(|| (id, node_idx), node.children(), total)
                })
                .sum::<BigUint>();
            (!count.is_zero()).then_some((id, count))
        })
        .collect::<Vec<_>>();
    for (id, count) in layer {
        whole.data.entry(id).or_default().insert(size, count);
    }
}

/// Find the smallest root size `<= cap` that makes at least `min_extractable`
/// terms available, returning `(max_size, counts)` pruned to `max_size`'s root
/// budgets.
///
/// Budgets are grown geometrically from `min_size(root) + initial_step` up to
/// `cap`, reusing a single `AstSize` pass so early successes avoid allocating
/// DP tables for the full `cap`.
///
/// # Errors
///
/// Returns the total terms found when `cap` is exhausted below `min_extractable`.
pub(crate) fn count<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    root: Id,
    cap: usize,
    initial_step: usize,
    min_extractable: usize,
) -> Result<(usize, ClassHistograms), BigUint> {
    let root = egraph.find(root);
    let min_sizes = MinSizes::new(egraph);
    let root_min = min_sizes.get(root);
    let threshold = BigUint::from(min_extractable);
    let mut last_term_count = BigUint::ZERO;

    for try_cap in budgets::doubling_caps(root_min, cap, initial_step) {
        let try_budgets = min_sizes.budgets_for(egraph, root, try_cap);
        let mut whole = LayerTable::new();

        let mut term_count = BigUint::ZERO;
        for size in 1..=try_budgets.limit() {
            step(egraph, &try_budgets, &mut whole, size);
            if let Some(count) = whole.data.get(&root).and_then(|hist| hist.get(&size)) {
                term_count += count;
                if term_count >= threshold {
                    let mut counts = whole.into_data();
                    drop(try_budgets);
                    min_sizes
                        .budgets_for(egraph, root, size)
                        .prune_histograms(&mut counts, |c| c);
                    return Ok((size, counts));
                }
            }
        }
        last_term_count = term_count;
    }

    Err(last_term_count)
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
        let (used_size, data) = count(&egraph, apb, 10, 2, 3).unwrap();
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
        let (used_size, data) = count(&egraph, apb, 10, 2, 22).unwrap();
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
        let (used_size, rooted) = count(&egraph, root, limit, 2, 5).unwrap();
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
        let (used_size, rooted) = count(&egraph, root, 10, 2, 6).unwrap();
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
        let big = egraph.add(SymbolLang::new("g", vec![gab, gab]));
        let oversized = egraph.add(SymbolLang::new("h", vec![big, gab, gab]));
        // Only fitting nodes of arity >= 3 store suffix tables.
        let root = egraph.add(SymbolLang::new("h", vec![gab, apb, b]));
        egraph.union(root, oversized);

        egraph.union(a, apb);
        egraph.rebuild();

        let limit = 9;
        let min_sizes = MinSizes::new(&egraph);
        let budgets = min_sizes.budgets_for(&egraph, root, limit);
        let mut whole = LayerTable::new();
        for size in 1..=budgets.limit() {
            step(&egraph, &budgets, &mut whole, size);
        }

        for &id in budgets.budgets().keys() {
            for (node_idx, node) in egraph[id].nodes.iter().enumerate() {
                let children = node.children();
                let fits = budgets.node_fits(&egraph, id, node);
                let Some(tables) = whole.suffix.get(&(id, node_idx)) else {
                    assert!(
                        children.len() <= 2 || !fits,
                        "fitting nodes of arity >= 3 populate suffix tables"
                    );
                    continue;
                };
                assert!(
                    children.len() >= 3 && fits,
                    "arity <= 2 or oversized nodes allocate no suffix tables"
                );
                let histograms = children
                    .iter()
                    .map(|c| whole.data.get(c).cloned().unwrap_or_default())
                    .collect::<Vec<_>>();
                let budget = budgets.budget(id).unwrap() - 1;
                let expected = sampling::suffix_convolutions(&histograms, budget);

                // Only positions `1..n - 1` are stored; the first and the
                // last two are implicit.
                assert_eq!(tables.len(), children.len() - 2);
                assert_eq!(tables.as_slice(), &expected[1..=tables.len()]);
            }
        }
    }
}
