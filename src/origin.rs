use core::fmt;
use std::{fmt::Display, hash::Hash};

use egg::{Id, Language, RecExpr};
use serde::Serialize;

pub type OriginExpr<L> = RecExpr<OriginNode<L>>;

/// A node in a labeled, ordered tree.
#[derive(Serialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct OriginNode<L: Language> {
    pub node: L,
    pub ty: Option<Id>,
    pub origin: Id,
}

impl<L: Language> OriginNode<L> {
    /// Create a leaf node with no children.
    pub fn new_untyped(node: L, origin: Id) -> Self {
        OriginNode {
            node,
            ty: None,
            origin,
        }
    }

    /// Create a leaf node with no children.
    pub fn new_typed(node: L, ty: Id, origin: Id) -> Self {
        OriginNode {
            node,
            ty: Some(ty),
            origin,
        }
    }

    // /// Build a type tree from an e-class's type annotation.
    // #[must_use]
    // pub fn from_eclass<N: TypeAnalysis<L>>(graph: &EClass<L>) -> Option<Self> {
    //     let ty_id = graph.class(id).data.ty()?;
    //     Some(Self::from_type(graph, *ty_id))
    // }

    pub fn ty(&self) -> Option<Id> {
        self.ty
    }

    pub fn origin(&self) -> Id {
        self.origin
    }
}

impl<L: Language> Language for OriginNode<L> {
    type Discriminant = L::Discriminant;

    fn discriminant(&self) -> Self::Discriminant {
        self.node.discriminant()
    }

    fn matches(&self, other: &Self) -> bool {
        self.node.matches(&other.node)
    }

    fn children(&self) -> &[Id] {
        self.node.children()
    }

    fn children_mut(&mut self) -> &mut [Id] {
        self.node.children_mut()
    }
}

pub fn strip<L: Language>(oe: OriginExpr<L>) -> RecExpr<L> {
    todo!()
}

pub fn display_oe<L: Language>(oe: &OriginExpr<L>) -> String {
    todo!()
}

// impl<L: Language> TreeShaped<L> for OriginNode<L> {
//     /// Returns true if this node has no children.
//     fn is_leaf(&self) -> bool {
//         self.node.is_leaf()
//     }

//     fn children(&self) -> &[Id] {
//         &self.node.children()
//     }

//     fn node(&self) -> &L {
//         &self.label
//     }

//     fn ty(&self) -> Option<Id> {
//         self.ty.as_deref()
//     }
// }

// impl<L: Label> From<OriginTree<L>> for Tree<L> {
//     fn from(value: OriginTree<L>) -> Self {
//         Tree {
//             label: value.label,
//             ty: value.ty.map(|t| Box::new((*t).into())),
//             children: value.children.into_iter().map(|x| x.into()).collect(),
//         }
//     }
// }
