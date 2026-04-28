use std::fmt::{self, Display};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp, SexpError};

use crate::graph::Graph;
use crate::ids::{DataChildId, DataId, EClassId, FunId, NatId, TypeChildId};
use crate::nodes::Label;

use super::TreeShaped;

/// A node in a labeled, ordered tree.
#[derive(Debug, Clone, Serialize, Deserialize, std::hash::Hash, PartialEq, Eq)]
#[serde(bound(deserialize = "L: Label"))]
pub struct TypedTree<L: Label> {
    pub(super) label: L,
    pub(super) ty: Option<Box<TypedTree<L>>>,
    pub(super) children: Vec<TypedTree<L>>,
}

impl<L: Label> TypedTree<L> {
    /// Create a leaf node with no children.
    pub fn leaf_untyped(label: L) -> Self {
        TypedTree {
            label,
            ty: None,
            children: Vec::new(),
        }
    }

    /// Create a leaf node with no children.
    pub fn leaf_typed(label: L, ty: Option<TypedTree<L>>) -> Self {
        TypedTree {
            label,
            ty: ty.map(Box::new),
            children: Vec::new(),
        }
    }

    /// Create a node with the given children.
    pub fn new_untyped(label: L, children: Vec<TypedTree<L>>) -> Self {
        TypedTree {
            label,
            ty: None,
            children,
        }
    }

    /// Create a node with the given children.
    pub fn new_typed(label: L, children: Vec<TypedTree<L>>, ty: Option<TypedTree<L>>) -> Self {
        TypedTree {
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
        TypedTree::new_untyped(node, children)
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
        TypedTree::new_untyped(node, children)
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
        TypedTree::new_untyped(node, children)
    }
}

impl<L: Label> TreeShaped<L> for TypedTree<L> {
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

    fn ty(&self) -> Option<&TypedTree<L>> {
        self.ty.as_deref()
    }
}

impl<L: Label + Display> Display for TypedTree<L> {
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

impl<L> FromStr for TypedTree<L>
where
    L: Label + FromStr,
    <L as FromStr>::Err: Display,
{
    type Err = SexpError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        /// Parse a type tree (no typeOf wrappers).
        fn parse_expr<L>(sexp: Sexp) -> Result<TypedTree<L>, SexpError>
        where
            L: Label + FromStr,
            <L as FromStr>::Err: Display,
        {
            match sexp {
                Sexp::String(s) => Ok(TypedTree::leaf_untyped(
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
                    Ok(TypedTree::new_untyped(
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

impl IntoSexp for TypedTree<String> {
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
