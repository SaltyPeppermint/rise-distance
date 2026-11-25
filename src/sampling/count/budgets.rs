use egg::{Analysis, AstSize, EGraph, Id, Language};
use num::BigUint;

use crate::analysis::semilattice::SemiLatticeAnalysis;
use crate::utils::{HashMap, UniqueQueue};

/// Per-class size bounds and minima for extractions from one root.
#[derive(Debug, Clone)]
pub(super) struct RootBudgets {
    budgets: HashMap<Id, usize>,
    min_sizes: HashMap<Id, usize>,
    limit: usize,
}

impl RootBudgets {
    #[must_use]
    pub(super) fn budget(&self, id: Id) -> Option<usize> {
        self.budgets.get(&id).copied()
    }

    #[must_use]
    pub(super) fn min_size(&self, id: Id) -> usize {
        self.min_sizes[&id]
    }

    #[must_use]
    pub(super) const fn budgets(&self) -> &HashMap<Id, usize> {
        &self.budgets
    }

    #[must_use]
    pub(super) const fn limit(&self) -> usize {
        self.limit
    }

    /// Whether `node` fits its class budget at minimum child sizes.
    #[must_use]
    pub(super) fn node_fits<L: Language, N: Analysis<L>>(
        &self,
        egraph: &EGraph<L, N>,
        class: Id,
        node: &L,
    ) -> bool {
        let Some(budget) = self.budget(class) else {
            return false;
        };
        node.children()
            .iter()
            .try_fold(1usize, |size, &child| {
                size.checked_add(self.min_size(egraph.find(child)))
            })
            .is_some_and(|minimum| minimum <= budget)
    }

    /// Drop entries outside these budgets from a histogram table keyed by `K`.
    pub(super) fn prune_histograms<K, F>(
        &self,
        data: &mut HashMap<K, HashMap<usize, BigUint>>,
        curr_of: F,
    ) where
        K: Copy,
        F: Fn(K) -> Id,
    {
        data.retain(|&key, hist| {
            let Some(limit) = self.budget(curr_of(key)) else {
                return false;
            };
            hist.retain(|&size, _| size <= limit);
            !hist.is_empty()
        });
    }

    /// Compute canonical class budgets and minima for a root and size limit.
    pub(super) fn of_root<L: Language, N: Analysis<L>>(
        egraph: &EGraph<L, N>,
        root: Id,
        limit: usize,
    ) -> Self {
        assert!(egraph.clean);
        let mut min_sizes = HashMap::default();
        AstSize.one_shot_analysis(egraph, &mut min_sizes);
        let budgets = class_budgets(egraph, egraph.find(root), limit, &min_sizes);
        Self {
            budgets,
            min_sizes,
            limit,
        }
    }
}

/// Largest subterm size usable by each class below `limit` from `root`.
/// Unreachable classes are ignored and not included in the result.
fn class_budgets<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    root: Id,
    limit: usize,
    min_sizes: &HashMap<Id, usize>,
) -> HashMap<Id, usize> {
    let mut budgets = [(egraph.find(root), limit)]
        .into_iter()
        .collect::<HashMap<_, _>>();
    let mut pending = budgets.keys().copied().collect::<UniqueQueue<_>>();

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
