use hashbrown::{HashMap, HashSet};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use super::Counter;
use super::TermCount;
use crate::TreeNode;
use crate::ids::{EClassId, ExprChildId};
use crate::nodes::Label;
use crate::tree::{PartialChild, PartialTree, tree_node_to_partial};

impl<C: Counter, L: Label> TermCount<'_, C, L> {
    /// Sample unique terms across a range of sizes from root, maximizing
    /// structural overlap with a reference tree.
    ///
    /// See `sample_unique_overlap` for more info.
    #[must_use]
    pub fn sample_unique_root_overlap(
        &self,
        ref_tree: &TreeNode<L>,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<TreeNode<L>> {
        self.sample_unique_overlap(
            self.graph.root(),
            ref_tree,
            min_size,
            max_size,
            samples_per_size,
        )
    }

    /// Sample unique terms across a range of sizes, maximizing structural
    /// overlap with a reference tree.
    ///
    /// Like `sample_unique`, but uses `sample_overlap` instead of `sample` so
    /// that the locked-in structure from the reference tree is preserved in
    /// every sample. The partial tree is built once and reused across all
    /// sizes. Sizes where the fixed overlap exceeds the target are skipped.
    #[must_use]
    pub fn sample_unique_overlap(
        &self,
        id: EClassId,
        ref_tree: &TreeNode<L>,
        min_size: usize,
        max_size: usize,
        samples_per_size: &HashMap<usize, u64>,
    ) -> HashSet<TreeNode<L>> {
        let canon_id = self.graph.canonicalize(id);
        let Some(partial) = self.match_ref_tree(canon_id, ref_tree) else {
            return HashSet::new();
        };
        let fixed = partial.fixed_size(self.with_types);

        self.data
            .get(&canon_id)
            .into_iter()
            .flat_map(|h| {
                h.keys()
                    .filter(|&&s| s >= min_size && s <= max_size && s >= fixed)
                    .copied()
            })
            .par_bridge()
            .flat_map(|size| {
                let remaining = size - fixed;
                let samples = samples_per_size[&size];
                let thread_partial = &partial;
                (0..samples).into_par_iter().filter_map(move |sample| {
                    let mut rng = ChaCha12Rng::seed_from_u64(size as u64);
                    rng.set_stream(sample);
                    self.fill_holes_sampling(remaining, &mut rng, thread_partial.clone())
                })
            })
            .collect()
    }

    pub fn sample_root_overlap(
        &self,
        ref_tree: &TreeNode<L>,
        target_size: usize,
        samples: u64,
        seed: u64,
    ) -> Option<impl ParallelIterator<Item = TreeNode<L>>> {
        self.sample_overlap(self.graph.root(), ref_tree, target_size, samples, seed)
    }

    /// Sample a term of exactly `target_size` from `eclass_id`, maximizing
    /// structural overlap with `ref_tree`.
    ///
    /// Matches `ref_tree` against the e-class to lock in shared structure,
    /// then jointly samples sizes for unmatched subtrees (holes) and fills them.
    /// Returns `None` if no valid term of the target size can be produced
    /// or the tree does not match at the root
    #[must_use]
    pub fn sample_overlap(
        &self,
        eclass_id: EClassId,
        ref_tree: &TreeNode<L>,
        target_size: usize,
        samples: u64,
        seed: u64,
    ) -> Option<impl ParallelIterator<Item = TreeNode<L>>> {
        let canonical_id = self.graph.canonicalize(eclass_id);

        // Build partial tree
        let partial = self.match_ref_tree(canonical_id, ref_tree)?;

        // Budget accounting
        let fixed = partial.fixed_size(self.with_types);

        if fixed > target_size {
            return None;
        }
        let remaining = target_size - fixed;

        Some((0..samples).into_par_iter().filter_map(move |sample| {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            rng.set_stream(sample);
            self.fill_holes_sampling(remaining, &mut rng, partial.clone())
        }))
    }

    /// Match a reference tree against an e-class, producing a partial tree
    /// that maximizes structural overlap with the reference.
    ///
    /// At each e-class, finds e-nodes whose label matches the `ref_tree`'s label.
    /// If multiple match, tries all and picks the one with the largest
    /// `resolved_count`. Returns `None` if no e-node matches (caller creates a Hole).
    pub(crate) fn match_ref_tree(
        &self,
        eclass_id: EClassId,
        ref_tree: &TreeNode<L>,
    ) -> Option<PartialTree<L>> {
        let canonical_id = self.graph.canonicalize(eclass_id);
        let eclass = self.graph.class(canonical_id);
        let ty = Some(TreeNode::<L>::from_eclass(self.graph, canonical_id));

        let mut best = None;

        for node in eclass
            .nodes()
            .iter()
            .filter(|node| node.label() == ref_tree.label())
        {
            let children = node.children();
            let ref_children = ref_tree.children();

            let mut partial_children = Vec::with_capacity(children.len());
            let mut ref_idx = 0;

            for &child_id in children {
                match child_id {
                    ExprChildId::Nat(nat_id) => {
                        let nat_tree = TreeNode::<L>::from_nat(self.graph, nat_id);
                        partial_children
                            .push(PartialChild::Resolved(tree_node_to_partial(&nat_tree)));
                        ref_idx += 1;
                    }
                    ExprChildId::Data(data_id) => {
                        let data_tree = TreeNode::<L>::from_data(self.graph, data_id);
                        partial_children
                            .push(PartialChild::Resolved(tree_node_to_partial(&data_tree)));
                        ref_idx += 1;
                    }
                    ExprChildId::EClass(child_eclass_id) => {
                        if ref_idx < ref_children.len() {
                            match self.match_ref_tree(child_eclass_id, &ref_children[ref_idx]) {
                                Some(pt) => {
                                    partial_children.push(PartialChild::Resolved(pt));
                                }
                                None => {
                                    partial_children.push(PartialChild::Hole(
                                        self.graph.canonicalize(child_eclass_id),
                                    ));
                                }
                            }
                        } else {
                            partial_children
                                .push(PartialChild::Hole(self.graph.canonicalize(child_eclass_id)));
                        }
                        ref_idx += 1;
                    }
                }
            }

            let pt = PartialTree::new(node.label().clone(), ty.clone(), partial_children);
            let overlap = pt.resolved_count();

            if best
                .as_ref()
                .is_none_or(|(_, best_overlap)| overlap > *best_overlap)
            {
                best = Some((pt, overlap));
            }
        }

        best.map(|(pt, _)| pt)
    }

    /// Jointly samples sizes for unmatched subtrees (holes) and fills them.
    /// Returns `None` if no valid term of the target size can be produced.
    pub fn fill_holes_sampling<R: Rng + SeedableRng>(
        &self,
        remaining: usize,
        rng: &mut R,
        partial: PartialTree<L>,
    ) -> Option<TreeNode<L>> {
        let holes = partial.holes();

        if holes.is_empty() {
            if remaining == 0 {
                let mut empty_iter = std::iter::empty();
                return Some(partial.fill(&mut empty_iter));
            }
            return None;
        }

        // Build histograms for each hole and jointly sample sizes
        let hole_histograms = holes
            .iter()
            .map(|&hole_id| {
                self.child_histogram(ExprChildId::EClass(hole_id))
                    .into_owned()
            })
            .collect::<Vec<_>>();

        let suffix = Self::suffix_convolutions(&hole_histograms, remaining);

        // Check feasibility
        if suffix[0].get(&remaining).is_none_or(|c| *c == C::zero()) {
            return None;
        }

        Some(self.fill_inner(remaining, rng, partial, &holes, &hole_histograms, &suffix))
    }

    fn fill_inner<R: Rng + SeedableRng>(
        &self,
        remaining: usize,
        rng: &mut R,
        partial: PartialTree<L>,
        holes: &[EClassId],
        hole_histograms: &[HashMap<usize, C>],
        suffix: &[HashMap<usize, C>],
    ) -> TreeNode<L> {
        // Sequentially pick a size for each hole
        let mut budget_left = remaining;
        let mut hole_sizes = Vec::with_capacity(holes.len());

        // TODO CHECK THAT THIS IS REALLY UNIFORM
        for (i, hist) in hole_histograms.iter().enumerate() {
            let candidates: Vec<_> = hist
                .iter()
                .filter_map(|(&s, count)| {
                    budget_left
                        .checked_sub(s)
                        .and_then(|r| suffix[i + 1].get(&r))
                        .map(|rest_count| (s, count.to_owned() * rest_count))
                })
                .collect();

            let dist = WeightedIndex::new(candidates.iter().map(|(_, w)| w)).unwrap();
            let chosen = candidates[dist.sample(rng)].0;
            hole_sizes.push(chosen);
            budget_left -= chosen;
        }

        // Fill each hole
        let filled = holes
            .iter()
            .zip(hole_sizes.iter())
            .map(|(&hole_id, &hole_size)| self.sample(hole_id, hole_size, rng))
            .collect::<Vec<_>>();

        let mut fill_iter = filled.into_iter();
        partial.fill(&mut fill_iter)
    }
}

#[cfg(test)]
mod tests {
    use super::super::TermCount;
    use super::super::test_utils::*;
    use super::*;
    use crate::EGraph;
    use crate::TreeNode;
    use crate::graph::EClass;
    use crate::nodes::ENode;
    use num::BigUint;

    #[test]
    fn match_ref_tree_exact() {
        // Class 0: f(class1)
        // Class 1: leaf "a"
        // ref_tree: (f a) — should match exactly with no holes
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        let partial = tc.match_ref_tree(EClassId::new(0), &ref_tree).unwrap();
        assert_eq!(partial.resolved_count(), 2); // f + a
        assert!(partial.holes().is_empty());
        assert_eq!(partial.fixed_size(false), 2);
    }

    #[test]
    fn match_ref_tree_partial_hole() {
        // Class 0: f(class1)
        // Class 1: leaf "a", leaf "b"
        // ref_tree: (f c) — "f" matches at root, "c" does NOT match class1
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "(f c)".parse().unwrap();

        let partial = tc.match_ref_tree(EClassId::new(0), &ref_tree).unwrap();
        assert_eq!(partial.resolved_count(), 1); // only f
        assert_eq!(partial.holes().len(), 1);
        assert_eq!(partial.holes()[0], EClassId::new(1));
    }

    #[test]
    fn match_ref_tree_no_match_at_root() {
        // Class 0: leaf "a"
        // ref_tree: "b" — no match
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "b".parse().unwrap();

        assert!(tc.match_ref_tree(EClassId::new(0), &ref_tree).is_none());
    }

    #[test]
    fn match_ref_tree_best_overlap() {
        // Class 0: two nodes both labeled "f", pointing to class1 and class2 resp.
        // Class 1: leaf "a"
        // Class 2: leaf "b"
        // ref_tree: (f a) — both f-nodes match at root, but only the one
        //   pointing to class1 can match child "a"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![
                        ENode::new("f".to_owned(), vec![eid(1)]),
                        ENode::new("f".to_owned(), vec![eid(2)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        let partial = tc.match_ref_tree(EClassId::new(0), &ref_tree).unwrap();
        // Should pick the f->class1 node which fully matches (f a)
        assert_eq!(partial.resolved_count(), 2);
        assert!(partial.holes().is_empty());
    }

    #[test]
    fn sample_with_overlap_exact_match() {
        // Class 0: f(class1)
        // Class 1: leaf "a"
        // ref_tree: (f a) — exact match, target_size=2
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        let results: Vec<_> = tc
            .sample_overlap(EClassId::new(0), &ref_tree, 2, 1, 42)
            .unwrap()
            .collect();
        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.size(false), 2);
        assert_eq!(result.label(), "f");
        assert_eq!(result.children()[0].label(), "a");
    }

    #[test]
    fn sample_with_overlap_with_holes() {
        // Class 0: f(class1, class2)
        // Class 1: leaf "a", leaf "x"
        // Class 2: leaf "b", leaf "y"
        // ref_tree: (f a z) — "a" matches class1, "z" does NOT match class2
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("x".to_owned())],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("b".to_owned()), ENode::leaf("y".to_owned())],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "(f a z)".parse().unwrap();

        // target_size = 3: f(1) + a(1) + hole(1) = 3
        let results: Vec<_> = tc
            .sample_overlap(EClassId::new(0), &ref_tree, 3, 1, 42)
            .unwrap()
            .collect();
        assert_eq!(results.len(), 1);
        let result = &results[0];
        assert_eq!(result.size(false), 3);
        assert_eq!(result.label(), "f");
        // First child should be "a" (matched)
        assert_eq!(result.children()[0].label(), "a");
        // Second child should be either "b" or "y" (sampled from class 2)
        let second = result.children()[1].label();
        assert!(second == "b" || second == "y");
    }

    #[test]
    fn sample_with_overlap_budget_too_small() {
        // Class 0: f(class1)
        // Class 1: leaf "a"
        // ref_tree: (f a) — fixed size = 2, but target = 1
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "(f a)".parse().unwrap();

        assert!(
            tc.sample_overlap(EClassId::new(0), &ref_tree, 1, 1, 42)
                .is_none()
        );
    }

    #[test]
    fn sample_with_overlap_no_match_returns_none() {
        // Class 0: leaf "a", leaf "b"
        // ref_tree: "z" — no match at root, returns None
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let tc = TermCount::<BigUint, _>::new(10, false, &graph);
        let ref_tree: TreeNode<String> = "z".parse().unwrap();

        assert!(
            tc.sample_overlap(EClassId::new(0), &ref_tree, 1, 1, 42)
                .is_none()
        );
    }
}
