use std::fmt::{self, Display};
use std::hash::{BuildHasher, Hash, Hasher};

use hashbrown::DefaultHashBuilder;
use serde::Serialize;

use crate::Tree;
use crate::graph::Graph;
use crate::ids::{AnyId, DataChildId, DataId, EClassId, FunId, NatId, TypeChildId};
use crate::nodes::Label;

use super::TreeShaped;

/// A node in a labeled, ordered tree.
#[derive(Serialize, Debug, Clone, Eq)]
pub struct OriginTree<L: Label> {
    label: L,
    ty: Option<Box<OriginTree<L>>>,
    children: Vec<OriginTree<L>>,
    origin: AnyId,
    /// Precomputed structural hash for O(1) hashing.
    #[serde(skip)]
    cached_hash: u64,
}

impl<L: Label> PartialEq for OriginTree<L> {
    fn eq(&self, other: &Self) -> bool {
        if self.cached_hash != other.cached_hash {
            return false;
        }
        self.origin == other.origin
            && self.label == other.label
            && self.ty == other.ty
            && self.children == other.children
    }
}

impl<L: Label> Hash for OriginTree<L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cached_hash.hash(state);
    }
}

fn compute_hash<L: Label>(
    label: &L,
    ty: Option<&OriginTree<L>>,
    children: &[OriginTree<L>],
    origin: AnyId,
) -> u64 {
    let mut hasher = DefaultHashBuilder::default().build_hasher();
    label.hash(&mut hasher);
    origin.hash(&mut hasher);
    if let Some(t) = ty {
        t.cached_hash.hash(&mut hasher);
    }
    for child in children {
        child.cached_hash.hash(&mut hasher);
    }
    hasher.finish()
}

impl<L: Label> OriginTree<L> {
    /// Create a leaf node with no children.
    pub fn leaf_untyped(label: L, origin: AnyId) -> Self {
        let cached_hash = compute_hash(&label, None, &[], origin);
        OriginTree {
            label,
            ty: None,
            children: Vec::new(),
            origin,
            cached_hash,
        }
    }

    /// Create a leaf node with no children.
    pub fn leaf_typed(label: L, ty: Option<OriginTree<L>>, origin: AnyId) -> Self {
        let cached_hash = compute_hash(&label, ty.as_ref(), &[], origin);
        OriginTree {
            label,
            ty: ty.map(Box::new),
            children: Vec::new(),
            origin,
            cached_hash,
        }
    }

    /// Create a node with the given children.
    pub fn new_untyped(label: L, children: Vec<OriginTree<L>>, origin: AnyId) -> Self {
        let cached_hash = compute_hash(&label, None, &children, origin);
        OriginTree {
            label,
            ty: None,
            children,
            origin,
            cached_hash,
        }
    }

    /// Create a node with the given children.
    pub fn new_typed(
        label: L,
        children: Vec<OriginTree<L>>,
        ty: Option<OriginTree<L>>,
        origin: AnyId,
    ) -> Self {
        let cached_hash = compute_hash(&label, ty.as_ref(), &children, origin);
        OriginTree {
            label,
            ty: ty.map(Box::new),
            children,
            origin,
            cached_hash,
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
        OriginTree::new_untyped(node, children, id.into())
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
        OriginTree::new_untyped(node, children, id.into())
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
        OriginTree::new_untyped(node, children, id.into())
    }

    pub fn origin(&self) -> AnyId {
        self.origin
    }
}

impl<L: Label> TreeShaped<L> for OriginTree<L> {
    /// Returns true if this node has no children.
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn children(&self) -> &[Self] {
        &self.children
    }

    fn label(&self) -> &L {
        &self.label
    }

    fn ty(&self) -> Option<&Self> {
        self.ty.as_deref()
    }
}

impl<L: Label + Display> Display for OriginTree<L> {
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

impl<L: Label> From<OriginTree<L>> for Tree<L> {
    fn from(value: OriginTree<L>) -> Self {
        Tree {
            label: value.label,
            ty: value.ty.map(|t| Box::new((*t).into())),
            children: value.children.into_iter().map(|x| x.into()).collect(),
        }
    }
}
