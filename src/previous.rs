//! Compact lookup index for an earlier e-graph boundary.

use egg::{Id, Language, UnionEvent};
use hashbrown::HashMap;

#[cfg(test)]
use egg::{Analysis, EGraph, RecExpr};

use crate::utils::DenseUnionFind;
#[cfg(test)]
use crate::{MyLanguage, OriginLang};

/// The lookup behavior novelty matching needs from the previous state.
pub(crate) trait PreviousLookup<L: Language> {
    fn lookup(&self, node: L) -> Option<Id>;
}

/// Canonical-node-to-class lookup reconstructed for one earlier boundary.
///
/// Class ids are replay representatives. They are intentionally used only as
/// opaque keys by novelty counting.
#[derive(Debug)]
pub(crate) struct PrevIndex<L: Language> {
    memo: HashMap<L, Id>,
}

impl<L: Language> PrevIndex<L> {
    /// Reconstruct an earlier hashcons relation from the final raw-node
    /// history and a borrowed effective-union log.
    pub(crate) fn from_union_history(
        raw_nodes: &[L],
        raw_node_count: usize,
        union_event_count: usize,
        events: &[UnionEvent],
    ) -> Self {
        assert!(
            raw_node_count <= raw_nodes.len(),
            "previous boundary contains more raw nodes than the final e-graph"
        );
        assert!(
            union_event_count <= events.len(),
            "previous boundary contains more unions than the event log"
        );

        let mut replay = DenseUnionFind::<Id>::new(raw_node_count);
        for event in &events[..union_event_count] {
            let left = usize::from(event.left);
            let right = usize::from(event.right);
            assert!(
                left < raw_node_count && right < raw_node_count,
                "union event references a node created after the boundary"
            );
            replay.union(event.left, event.right);
        }
        let mut memo = HashMap::with_capacity(raw_node_count);
        for (raw_index, raw_node) in raw_nodes[..raw_node_count].iter().enumerate() {
            let mut node = raw_node.clone();
            node.for_each_mut(|child| {
                assert!(
                    usize::from(*child) < raw_node_count,
                    "raw node references a child created after the boundary"
                );
                *child = replay.find(*child);
            });
            let owner = replay.find(Id::from(raw_index));
            if let Some(previous_owner) = memo.insert(node, owner) {
                assert_eq!(
                    replay.find(previous_owner),
                    replay.find(owner),
                    "congruent old nodes were not joined in the replay"
                );
            }
        }

        Self { memo }
    }

    #[cfg(test)]
    pub(crate) fn lookup_expr(&self, expr: &RecExpr<L>) -> Option<Id> {
        let mut ids = Vec::with_capacity(expr.len());
        for node in expr {
            let node = node.clone().map_children(|child| ids[usize::from(child)]);
            ids.push(self.lookup(node)?);
        }
        ids.last().copied()
    }

    /// Whether an origin-annotated expression existed at the indexed
    /// boundary. Origins are deliberately ignored: previous membership is a
    /// property of the lowered language expression.
    #[cfg(test)]
    pub(crate) fn contains_origin_expr(&self, expr: &RecExpr<OriginLang<L>>) -> bool
    where
        L: MyLanguage,
    {
        let mut ids: Vec<Option<Id>> = Vec::with_capacity(expr.as_ref().len());
        for node in expr.as_ref() {
            if node
                .children()
                .iter()
                .any(|&child| ids[usize::from(child)].is_none())
            {
                ids.push(None);
                continue;
            }
            let lowered = node
                .inner()
                .clone()
                .map_children(|child| ids[usize::from(child)].unwrap());
            ids.push(self.lookup(lowered));
        }
        ids.last().is_some_and(Option::is_some)
    }
}

impl<L: Language> PreviousLookup<L> for PrevIndex<L> {
    fn lookup(&self, node: L) -> Option<Id> {
        self.memo.get(&node).copied()
    }
}

#[cfg(test)]
impl<L: Language, N: Analysis<L>> PreviousLookup<L> for EGraph<L, N> {
    fn lookup(&self, node: L) -> Option<Id> {
        EGraph::lookup(self, node).map(|id| self.find(id))
    }
}

#[cfg(test)]
mod tests {
    use egg::EGraph;

    use super::*;
    use crate::langs::math::Math;
    use crate::utils::sym;

    #[test]
    fn replay_reconstructs_merge_and_congruence() {
        let mut graph = EGraph::<Math, ()>::new(());
        graph.enable_union_event_recording();
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        let fa = graph.add(Math::Ln(a));
        let fb = graph.add(Math::Ln(b));
        graph.rebuild();
        graph.union(a, b);
        graph.rebuild();

        let event_count = graph.union_event_count();
        let raw_count = graph.nodes().len();
        let index = PrevIndex::from_union_history(
            graph.nodes(),
            raw_count,
            event_count,
            graph.union_events(),
        );

        assert_eq!(
            index.lookup_expr(&"a".parse().unwrap()),
            Some(index.lookup(sym("a")).unwrap())
        );
        assert_eq!(
            index.lookup(Math::Ln(index.lookup(sym("a")).unwrap())),
            index.lookup(Math::Ln(index.lookup(sym("b")).unwrap()))
        );
        assert_eq!(graph.find(fa), graph.find(fb));
    }

    #[test]
    fn replay_boundary_excludes_later_nodes_and_unions() {
        let mut graph = EGraph::<Math, ()>::new(());
        graph.enable_union_event_recording();
        let a = graph.add(sym("a"));
        graph.rebuild();
        let raw_count = graph.nodes().len();
        let event_count = graph.union_event_count();

        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let index = PrevIndex::from_union_history(
            graph.nodes(),
            raw_count,
            event_count,
            graph.union_events(),
        );
        assert!(index.lookup(sym("a")).is_some());
        assert!(index.lookup(sym("b")).is_none());
    }

    #[test]
    fn recorder_ignores_repeated_noop_unions() {
        let mut graph = EGraph::<Math, ()>::new(());
        graph.enable_union_event_recording();
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));

        assert!(graph.union(a, b));
        assert_eq!(graph.union_event_count(), 1);
        assert!(!graph.union(a, b));
        assert_eq!(graph.union_event_count(), 1);
    }

    #[test]
    fn recorder_captures_explanation_hashcons_unions() {
        let mut graph = EGraph::<Math, ()>::new(()).with_explanations_enabled();
        graph.enable_union_event_recording();
        let a = graph.add_uncanonical(sym("a"));
        let b = graph.add_uncanonical(sym("b"));
        graph.union_trusted(a, b, "merge");
        graph.rebuild();

        graph.add_uncanonical(Math::Ln(a));
        let before = graph.union_event_count();
        graph.add_uncanonical(Math::Ln(b));

        assert_eq!(
            graph.union_event_count(),
            before + 1,
            "the explanation-only raw representative must be logged"
        );
    }

    #[test]
    fn origin_membership_ignores_origins_and_rejects_new_children() {
        let mut graph = EGraph::<Math, ()>::new(());
        graph.enable_union_event_recording();
        let a = graph.add(sym("a"));
        let fa = graph.add(Math::Ln(a));
        graph.rebuild();
        let index = PrevIndex::from_union_history(
            graph.nodes(),
            graph.nodes().len(),
            graph.union_event_count(),
            graph.union_events(),
        );

        let old = RecExpr::from(vec![
            OriginLang::new(sym("a"), Id::from(999)),
            OriginLang::new(Math::Ln(Id::from(0)), Id::from(998)),
        ]);
        let new = RecExpr::from(vec![
            OriginLang::new(sym("b"), a),
            OriginLang::new(Math::Ln(Id::from(0)), fa),
        ]);
        assert!(index.contains_origin_expr(&old));
        assert!(!index.contains_origin_expr(&new));
    }
}
