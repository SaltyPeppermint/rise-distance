//! Zhang-Shasha Tree Edit Distance Algorithm
//!
//! Zhang-Shasha computes the edit distance between two ordered labeled trees.
//! The algorithm runs in O(n1 * n2 * min(depth1, leaves1) * min(depth2, leaves2))
//! time and O(n1 * n2) space.

use std::fmt::{self, Display};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp, SexpError};

use super::graph::Graph;
use super::ids::{AnyId, DataChildId, DataId, EClassId, FunId, NatId, TypeChildId};
use super::nodes::Label;

pub trait TreeShaped<L: Label>: Sized {
    /// Returns true if this node has no children.
    fn is_leaf(&self) -> bool;

    fn children(&self) -> &[Self];

    fn children_mut(&mut self) -> &mut Vec<Self>;

    fn label(&self) -> &L;

    fn ty(&self) -> Option<&Self>;

    fn flatten(&self, with_types: bool) -> FlattenedTreeNode<L> {
        if !with_types {
            return FlattenedTreeNode {
                label: self.label().clone(),
                children: self
                    .children()
                    .iter()
                    .map(|c| c.flatten(with_types))
                    .collect(),
            };
        }
        if let Some(ty) = &self.ty() {
            FlattenedTreeNode {
                label: L::type_of(),
                children: vec![
                    FlattenedTreeNode {
                        label: self.label().clone(),
                        children: self
                            .children()
                            .iter()
                            .map(|c| c.flatten(with_types))
                            .collect(),
                    },
                    ty.flatten(with_types),
                ],
            }
        } else {
            FlattenedTreeNode {
                label: self.label().clone(),
                children: self
                    .children()
                    .iter()
                    .map(|c| c.flatten(with_types))
                    .collect(),
            }
        }
    }

    fn size(&self, with_types: bool) -> usize {
        if with_types {
            self.size_with_types()
        } else {
            self.size_without_types()
        }
    }

    /// Count total number of nodes in this tree.
    fn size_without_types(&self) -> usize {
        1 + self
            .children()
            .iter()
            .map(Self::size_without_types)
            .sum::<usize>()
    }

    fn size_with_types(&self) -> usize {
        1 + self
            .children()
            .iter()
            .map(Self::size_with_types)
            .sum::<usize>()
            + self.ty().map_or(0, |t| t.size_with_types())
    }
}

/// A node in a labeled, ordered tree.
#[derive(Debug, Clone, Serialize, Deserialize, std::hash::Hash, PartialEq, Eq)]
#[serde(bound(deserialize = "L: Label"))]
pub struct TreeNode<L: Label> {
    label: L,
    ty: Option<Box<TreeNode<L>>>,
    children: Vec<TreeNode<L>>,
}

impl<L: Label> TreeNode<L> {
    /// Create a leaf node with no children.
    pub fn leaf_untyped(label: L) -> Self {
        TreeNode {
            label,
            ty: None,
            children: Vec::new(),
        }
    }

    /// Create a leaf node with no children.
    pub fn leaf_typed(label: L, ty: Option<TreeNode<L>>) -> Self {
        TreeNode {
            label,
            ty: ty.map(Box::new),
            children: Vec::new(),
        }
    }

    /// Create a node with the given children.
    pub fn new_untyped(label: L, children: Vec<TreeNode<L>>) -> Self {
        TreeNode {
            label,
            ty: None,
            children,
        }
    }

    /// Create a node with the given children.
    pub fn new_typed(label: L, children: Vec<TreeNode<L>>, ty: Option<TreeNode<L>>) -> Self {
        TreeNode {
            label,
            ty: ty.map(Box::new),
            children,
        }
    }

    /// Build a type tree from an e-class's type annotation.
    #[must_use]
    pub fn from_eclass(graph: &Graph<L>, id: EClassId) -> Option<Self> {
        let ty_id = graph.class(id).ty()?;
        Some(Self::from_type(graph, *ty_id))
    }

    fn from_type(graph: &Graph<L>, id: TypeChildId) -> Self {
        match id {
            TypeChildId::Nat(nat_id) => Self::from_nat(graph, nat_id),
            TypeChildId::Type(fun_ty_id) => Self::from_fun(graph, fun_ty_id),
            TypeChildId::Data(data_ty_id) => Self::from_data(graph, data_ty_id),
        }
    }

    fn from_fun(graph: &Graph<L>, id: FunId) -> Self {
        let node = graph.fun_ty(id).label().to_owned();
        let children = graph
            .fun_ty(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_type(graph, c_id))
            .collect();
        TreeNode::new_untyped(node, children)
    }

    #[must_use]
    pub fn from_data(graph: &Graph<L>, id: DataId) -> Self {
        let node = graph.data_ty(id).label().to_owned();
        let children = graph
            .data_ty(id)
            .children()
            .iter()
            .map(|&c_id| match c_id {
                DataChildId::Nat(nat_id) => Self::from_nat(graph, nat_id),
                DataChildId::DataType(data_ty_id) => Self::from_data(graph, data_ty_id),
            })
            .collect();
        TreeNode::new_untyped(node, children)
    }

    #[must_use]
    pub fn from_nat(graph: &Graph<L>, id: NatId) -> Self {
        let node = graph.nat(id).label().to_owned();
        let children = graph
            .nat(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_nat(graph, c_id))
            .collect();
        TreeNode::new_untyped(node, children)
    }
}

impl<L: Label> TreeShaped<L> for TreeNode<L> {
    /// Returns true if this node has no children.
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn children(&self) -> &[Self] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<Self> {
        &mut self.children
    }

    fn label(&self) -> &L {
        &self.label
    }

    fn ty(&self) -> Option<&TreeNode<L>> {
        self.ty.as_deref()
    }
}

impl<L: Label + Display> Display for TreeNode<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ty.is_some() {
            write!(f, "({} ", L::type_of())?;
        }
        if self.is_leaf() {
            write!(f, "{}", self.label)?;
        } else {
            write!(f, "({}", self.label)?;
            for child in &self.children {
                write!(f, " {child}")?;
            }
            write!(f, ")")?;
        }
        if let Some(ty) = &self.ty {
            write!(f, " {ty})")?;
        }
        Ok(())
    }
}

impl<L> FromStr for TreeNode<L>
where
    L: Label + FromStr,
    <L as FromStr>::Err: Display,
{
    type Err = SexpError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        /// Parse a type tree (no typeOf wrappers).
        fn parse_expr<L>(sexp: Sexp) -> Result<TreeNode<L>, SexpError>
        where
            L: Label + FromStr,
            <L as FromStr>::Err: Display,
        {
            match sexp {
                Sexp::String(s) => Ok(TreeNode::leaf_untyped(
                    s.parse::<L>()
                        .map_err(|e| SexpError::Other(e.to_string()))?,
                )),
                Sexp::List(mut sexps) => {
                    if sexps.len() == 3
                        && let Some(Sexp::String(s)) = sexps.first()
                        && s == &L::type_of().to_string()
                    {
                        sexps.remove(0);
                        let expr_sexp = sexps.remove(0);
                        let type_sexp = sexps.remove(0);

                        let mut expr_node = parse_expr(expr_sexp)?;
                        let type_node = parse_expr(type_sexp)?;

                        expr_node.ty = Some(Box::new(type_node));
                        return Ok(expr_node);
                    }

                    let mut iter = sexps.into_iter();
                    let Some(Sexp::String(label)) = iter.next() else {
                        return Err(SexpError::Other("expected (label ...)".to_owned()));
                    };
                    Ok(TreeNode::new_untyped(
                        label
                            .parse::<L>()
                            .map_err(|e| SexpError::Other(e.to_string()))?,
                        iter.map(parse_expr).collect::<Result<_, _>>()?,
                    ))
                }
                Sexp::Empty => Err(SexpError::Other("empty sexp".to_owned())),
            }
        }

        symbolic_expressions::parser::parse_str(s).and_then(parse_expr)
    }
}

impl IntoSexp for TreeNode<String> {
    fn into_sexp(&self) -> Sexp {
        if self.is_leaf() {
            // Leaf with no type - just the label
            Sexp::String(self.label.clone())
        } else {
            Sexp::List(
                vec![Sexp::String(self.label.clone())]
                    .into_iter()
                    .chain(self.children().iter().map(Sexp::from))
                    .collect::<Vec<_>>(),
            )
        }
    }
}

/// A node in a labeled, ordered tree.
#[derive(Serialize, Debug, Clone, std::hash::Hash, PartialEq, Eq)]
pub struct TreeNodeWithOrigin<L: Label> {
    label: L,
    ty: Option<Box<TreeNodeWithOrigin<L>>>,
    children: Vec<TreeNodeWithOrigin<L>>,
    origin: AnyId,
}

impl<L: Label> TreeNodeWithOrigin<L> {
    /// Create a leaf node with no children.
    pub fn leaf_untyped(label: L, origin: AnyId) -> Self {
        TreeNodeWithOrigin {
            label,
            ty: None,
            children: Vec::new(),
            origin,
        }
    }

    /// Create a leaf node with no children.
    pub fn leaf_typed(label: L, ty: Option<TreeNodeWithOrigin<L>>, origin: AnyId) -> Self {
        TreeNodeWithOrigin {
            label,
            ty: ty.map(Box::new),
            children: Vec::new(),
            origin,
        }
    }

    /// Create a node with the given children.
    pub fn new_untyped(label: L, children: Vec<TreeNodeWithOrigin<L>>, origin: AnyId) -> Self {
        TreeNodeWithOrigin {
            label,
            ty: None,
            children,
            origin,
        }
    }

    /// Create a node with the given children.
    pub fn new_typed(
        label: L,
        children: Vec<TreeNodeWithOrigin<L>>,
        ty: Option<TreeNodeWithOrigin<L>>,
        origin: AnyId,
    ) -> Self {
        TreeNodeWithOrigin {
            label,
            ty: ty.map(Box::new),
            children,
            origin,
        }
    }

    /// Build a type tree from an e-class's type annotation.
    #[must_use]
    pub fn from_eclass(graph: &Graph<L>, id: EClassId) -> Option<Self> {
        let ty_id = graph.class(id).ty()?;
        Some(Self::from_type(graph, *ty_id))
    }

    fn from_type(graph: &Graph<L>, id: TypeChildId) -> Self {
        match id {
            TypeChildId::Nat(nat_id) => Self::from_nat(graph, nat_id),
            TypeChildId::Type(fun_ty_id) => Self::from_fun(graph, fun_ty_id),
            TypeChildId::Data(data_ty_id) => Self::from_data(graph, data_ty_id),
        }
    }

    fn from_fun(graph: &Graph<L>, id: FunId) -> Self {
        let node = graph.fun_ty(id).label().to_owned();
        let children = graph
            .fun_ty(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_type(graph, c_id))
            .collect();
        TreeNodeWithOrigin::new_untyped(node, children, id.into())
    }

    #[must_use]
    pub fn from_data(graph: &Graph<L>, id: DataId) -> Self {
        let node = graph.data_ty(id).label().to_owned();
        let children = graph
            .data_ty(id)
            .children()
            .iter()
            .map(|&c_id| match c_id {
                DataChildId::Nat(nat_id) => Self::from_nat(graph, nat_id),
                DataChildId::DataType(data_ty_id) => Self::from_data(graph, data_ty_id),
            })
            .collect();
        TreeNodeWithOrigin::new_untyped(node, children, id.into())
    }

    #[must_use]
    pub fn from_nat(graph: &Graph<L>, id: NatId) -> Self {
        let node = graph.nat(id).label().to_owned();
        let children = graph
            .nat(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_nat(graph, c_id))
            .collect();
        TreeNodeWithOrigin::new_untyped(node, children, id.into())
    }

    pub fn origin(&self) -> AnyId {
        self.origin
    }
}

impl<L: Label> TreeShaped<L> for TreeNodeWithOrigin<L> {
    /// Returns true if this node has no children.
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn children(&self) -> &[Self] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut Vec<Self> {
        &mut self.children
    }

    fn label(&self) -> &L {
        &self.label
    }

    fn ty(&self) -> Option<&Self> {
        self.ty.as_deref()
    }
}

impl<L: Label + Display> Display for TreeNodeWithOrigin<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ty.is_some() {
            write!(f, "({} ", L::type_of())?;
        }
        if self.is_leaf() {
            write!(f, "{}", self.label)?;
        } else {
            write!(f, "({}", self.label)?;
            for child in &self.children {
                write!(f, " {child}")?;
            }
            write!(f, ")")?;
        }
        if let Some(ty) = &self.ty {
            write!(f, " {ty})")?;
        }
        Ok(())
    }
}

impl<L: Label> From<TreeNodeWithOrigin<L>> for TreeNode<L> {
    fn from(value: TreeNodeWithOrigin<L>) -> Self {
        TreeNode {
            label: value.label,
            ty: value.ty.map(|t| Box::new((*t).into())),
            children: value.children.into_iter().map(|x| x.into()).collect(),
        }
    }
}

/// A node in a labeled, ordered tree.
#[derive(Debug, Clone, std::hash::Hash, PartialEq, Eq)]
pub struct FlattenedTreeNode<L: Label> {
    label: L,
    children: Vec<FlattenedTreeNode<L>>,
}

impl<L: Label> FlattenedTreeNode<L> {
    pub fn children(&self) -> &[FlattenedTreeNode<L>] {
        &self.children
    }

    pub fn label(&self) -> &L {
        &self.label
    }

    /// Returns true if this node has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn size(&self) -> usize {
        1 + self.children.iter().map(Self::size).sum::<usize>()
    }
}

impl<L: Label + Display> Display for FlattenedTreeNode<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_leaf() {
            write!(f, "{}", self.label)
        } else {
            write!(f, "({}", self.label)?;
            for child in &self.children {
                write!(f, " {child}")?;
            }
            write!(f, ")")
        }
    }
}

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
    ty: Option<Box<TreeNode<L>>>,
    children: Vec<PartialChild<L>>,
}

impl<L: Label> PartialTree<L> {
    pub fn new(label: L, ty: Option<TreeNode<L>>, children: Vec<PartialChild<L>>) -> Self {
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
    pub fn fill(self, fill: &mut impl Iterator<Item = TreeNode<L>>) -> TreeNode<L> {
        let children = self
            .children
            .into_iter()
            .map(|c| match c {
                PartialChild::Resolved(pt) => pt.fill(fill),
                PartialChild::Hole(_) => fill.next().expect("not enough fill trees"),
            })
            .collect();
        TreeNode::new_typed(self.label, children, self.ty.map(|b| *b))
    }
}

/// Convert a fully-resolved `TreeNode` into a `PartialTree` with no holes.
pub fn tree_node_to_partial<L: Label>(tree: &TreeNode<L>) -> PartialTree<L> {
    PartialTree::new(
        tree.label().clone(),
        tree.ty().cloned(),
        tree.children()
            .iter()
            .map(|c| PartialChild::Resolved(tree_node_to_partial(c)))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sexp_roundtrip_simple() {
        use symbolic_expressions::IntoSexp;

        // Parse a simple s-expression and serialize it back
        let input = "(f a b)";
        let tree = input.parse::<TreeNode<String>>().unwrap();

        assert_eq!(tree.label(), "f");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "a");
        assert_eq!(tree.children()[1].label(), "b");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_nested() {
        use symbolic_expressions::IntoSexp;

        // Nested s-expressions
        let input = "(a (b c) (d e))";
        let tree = input.parse::<TreeNode<String>>().unwrap();

        assert_eq!(tree.label(), "a");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "b");
        assert_eq!(tree.children()[0].children()[0].label(), "c");
        assert_eq!(tree.children()[1].label(), "d");
        assert_eq!(tree.children()[1].children()[0].label(), "e");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_complex_type() {
        use symbolic_expressions::IntoSexp;

        // Expression with a type-like structure
        let input = "(-> int (-> int int))";
        let tree = input.parse::<TreeNode<String>>().unwrap();

        assert_eq!(tree.label(), "->");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "int");
        assert_eq!(tree.children()[1].label(), "->");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_large() {
        use symbolic_expressions::IntoSexp;

        let input = "(natLam (natLam (natLam (lam (lam (app (app map (lam (app (app map (lam (app (app (app reduce add) 0.0) (app (app map (lam (app (app mul (app fst $e0)) (app snd $e0)))) (app (app zip $e1) $e0))))) (app transpose $e1)))) $e1))))))";
        let tree = input.parse::<TreeNode<String>>().unwrap();

        assert_eq!(tree.label(), "natLam");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_leaf() {
        use symbolic_expressions::IntoSexp;

        let input = "x";
        let tree = input.parse::<TreeNode<String>>().unwrap();

        assert_eq!(tree.label(), "x");
        assert!(tree.is_leaf());

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }
}
