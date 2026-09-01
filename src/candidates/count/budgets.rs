use egg::{Analysis, AstSize, EGraph, Id, Language};
use hashbrown::HashMap;

use crate::analysis::semilattice::SemiLatticeAnalysis;
use crate::utils::UniqueQueue;

/// Per-class size bounds and minima for extractions from one root.
#[derive(Debug, Clone)]
pub(crate) struct RootBudgets {
    budgets: HashMap<Id, usize>,
    min_sizes: HashMap<Id, usize>,
    limit: usize,
}

impl RootBudgets {
    #[must_use]
    pub(crate) fn budget(&self, id: Id) -> Option<usize> {
        self.budgets.get(&id).copied()
    }

    #[must_use]
    pub(crate) fn min_size(&self, id: Id) -> usize {
        self.min_sizes[&id]
    }

    #[must_use]
    pub(crate) const fn budgets(&self) -> &HashMap<Id, usize> {
        &self.budgets
    }

    #[must_use]
    pub(crate) const fn limit(&self) -> usize {
        self.limit
    }

    /// Whether `node` fits its class budget at minimum child sizes.
    #[must_use]
    pub(crate) fn node_fits<L: Language, N: Analysis<L>>(
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
}

/// Compute canonical class budgets and minima for a root and size limit.
pub(crate) fn root_budgets<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    root: Id,
    limit: usize,
) -> RootBudgets {
    assert!(egraph.clean);
    let mut raw_min_sizes = HashMap::new();
    AstSize.one_shot_analysis(egraph, &mut raw_min_sizes);
    let min_sizes = egraph
        .classes()
        .map(|class| {
            let id = egraph.find(class.id);
            (id, raw_min_sizes[&id])
        })
        .collect();
    let budgets = class_budgets(egraph, egraph.find(root), limit, &min_sizes);
    RootBudgets {
        budgets,
        min_sizes,
        limit,
    }
}

/// Largest subterm size usable by each class below `limit` from `root`.
/// Unreachable classes are omitted.
fn class_budgets<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    root: Id,
    limit: usize,
    min_sizes: &HashMap<Id, usize>,
) -> HashMap<Id, usize> {
    let mut budgets = HashMap::from([(egraph.find(root), limit)]);
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
