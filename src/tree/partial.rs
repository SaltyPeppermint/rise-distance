use crate::TypedTree;
use crate::ids::EClassId;
use crate::nodes::Label;

use super::TreeShaped;

/// A child in a partial tree: either a resolved subtree or an unresolved hole.
#[derive(Debug, Clone)]
pub enum PartialChild<L: Label> {
    /// A resolved subtree that matched the reference tree.
    Resolved(PartialTree<L>),
    /// An unresolved hole: an e-class that must be sampled later.
    Hole(EClassId),
}

/// A partially-matched tree against an e-graph, guided by a reference tree.
/// Resolved nodes come from matching the reference; holes are e-classes
/// where the reference tree did not match and need to be filled by sampling.
#[derive(Debug, Clone)]
pub struct PartialTree<L: Label> {
    label: L,
    /// Type annotation, always fully resolved from the e-graph.
    ty: Option<Box<TypedTree<L>>>,
    children: Vec<PartialChild<L>>,
}

impl<L: Label> PartialTree<L> {
    pub fn new(label: L, ty: Option<TypedTree<L>>, children: Vec<PartialChild<L>>) -> Self {
        PartialTree {
            label,
            ty: ty.map(Box::new),
            children,
        }
    }

    /// Count the number of resolved (non-hole) nodes in this partial tree.
    /// Used as the overlap metric when choosing among candidate e-node matches.
    /// Does NOT count type nodes.
    pub fn resolved_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(|c| match c {
                PartialChild::Resolved(pt) => pt.resolved_count(),
                PartialChild::Hole(_) => 0,
            })
            .sum::<usize>()
    }

    /// Compute the fixed size of the resolved portion of this tree.
    /// Matches the size accounting used by `TermCount` and `TreeNode::size_with_types`.
    /// Each resolved node contributes 1 + type overhead (if `with_types` and ty is present).
    /// Holes contribute 0.
    pub fn fixed_size(&self, with_types: bool) -> usize {
        let type_overhead = if with_types {
            self.ty.as_deref().map_or(0, |t| 1 + t.size_without_types())
        } else {
            0
        };
        1 + type_overhead
            + self
                .children
                .iter()
                .map(|c| match c {
                    PartialChild::Resolved(pt) => pt.fixed_size(with_types),
                    PartialChild::Hole(_) => 0,
                })
                .sum::<usize>()
    }

    /// Collect all holes (unresolved e-class IDs) in depth-first order.
    pub fn holes(&self) -> Vec<EClassId> {
        let mut result = Vec::new();
        self.collect_holes(&mut result);
        result
    }

    fn collect_holes(&self, out: &mut Vec<EClassId>) {
        for child in &self.children {
            match child {
                PartialChild::Resolved(pt) => pt.collect_holes(out),
                PartialChild::Hole(id) => out.push(*id),
            }
        }
    }

    /// Fill all holes with the provided `TreeNode`s (in the same depth-first
    /// order as `holes()` returns them), producing a complete `TreeNode`.
    #[expect(clippy::missing_panics_doc, clippy::impl_trait_in_params)]
    pub fn fill(self, fill: &mut impl Iterator<Item = TypedTree<L>>) -> TypedTree<L> {
        let children = self
            .children
            .into_iter()
            .map(|c| match c {
                PartialChild::Resolved(pt) => pt.fill(fill),
                PartialChild::Hole(_) => fill.next().expect("not enough fill trees"),
            })
            .collect();
        TypedTree::new_typed(self.label, children, self.ty.map(|b| *b))
    }
}

/// Convert a fully-resolved `TreeNode` into a `PartialTree` with no holes.
pub fn tree_node_to_partial<L: Label>(tree: &TypedTree<L>) -> PartialTree<L> {
    PartialTree::new(
        tree.label().clone(),
        tree.ty().cloned(),
        tree.children()
            .iter()
            .map(|c| PartialChild::Resolved(tree_node_to_partial(c)))
            .collect(),
    )
}
