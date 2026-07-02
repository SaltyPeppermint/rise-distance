mod expr_count;

use std::fmt::Debug;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use crate::utils::UniqueQueue;

pub use expr_count::ExprCount;

pub trait CommutativeSemigroupAnalysis<L: Language, N: Analysis<L>, C = ()>: Sized + Debug {
    type Data: PartialEq;

    fn make(
        &self,
        egraph: &EGraph<L, N>,
        eclass_id: Id,
        enode: &L,
        analysis_of: &HashMap<Id, Self::Data>,
    ) -> Self::Data;

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge;

    fn one_shot_analysis(&self, egraph: &EGraph<L, N>) -> HashMap<Id, Self::Data> {
        assert!(egraph.clean);

        // We start at the leaves, since they have no children and can be directly evaluated.
        let mut analysis_pending = egraph
            .classes()
            .filter(|eclass| eclass.nodes.iter().any(|enode| enode.is_leaf()))
            // No egraph.find since we are taking the id directly from the eclass
            .map(|eclass| eclass.id)
            .collect();

        let mut data = HashMap::new();
        resolve_pending_analysis(egraph, self, &mut data, &mut analysis_pending);

        debug_assert!(egraph.classes().all(|eclass| data.contains_key(&eclass.id)));
        data
    }
}

/// Single-threaded worklist fixpoint. The counting monoid is monotone,
/// so a class's value only grows until `make` truncates it at the size
/// limit; with one worker every publish is computed from the current
/// `data`, so an empty worklist is a genuine fixpoint. (A prior
/// multi-threaded version raced here: a stale computation could clobber
/// a fresher value under a non-atomic read-modify-write, under-counting
/// classes in a cycle. Measurements showed the parallelism bought ~1.0x
/// while doing so, so it was removed rather than made race-safe.)
fn resolve_pending_analysis<L, N, B, CC>(
    egraph: &EGraph<L, N>,
    analysis: &B,
    data: &mut HashMap<Id, B::Data>,
    analysis_pending: &mut UniqueQueue<Id>,
) where
    L: Language,
    N: Analysis<L>,
    B: CommutativeSemigroupAnalysis<L, N, CC>,
{
    while let Some(id) = analysis_pending.pop() {
        let canonical_id = egraph.find(id);
        debug_assert_eq!(canonical_id, id);
        let eclass = &egraph[canonical_id];

        // Check if we can calculate the analysis for any enode
        let available_data = eclass.nodes.iter().filter_map(|n| {
            let u_node = n.clone().map_children(|child_id| egraph.find(child_id));
            // If all the childs eclass_children have data, we can calculate it!
            u_node
                .all(|child_id| data.contains_key(&child_id))
                .then(|| analysis.make(egraph, canonical_id, &u_node, data))
        });

        // If we have some info, we add that info to our storage.
        // Otherwise we have absolutely no info about the nodes so we can only put them back onto the queue.
        // and hope for a better time later.
        if let Some(computed_data) = available_data.reduce(|mut a, b| {
            analysis.merge(&mut a, b);
            a
        }) {
            // If we have gained new information, put the parents onto the queue.
            // They need to be re-evaluated.
            // Only once we have reached a fixpoint we can stop updating the parents.
            if data.get(&eclass.id) != Some(&computed_data) {
                analysis_pending.extend(eclass.parents().map(|p| egraph.find(p)));
                data.insert(eclass.id, computed_data);
            }
        } else {
            assert!(!eclass.nodes.is_empty());
            analysis_pending.insert(canonical_id);
        }
    }
}
