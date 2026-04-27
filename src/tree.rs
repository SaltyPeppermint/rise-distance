use std::fmt::{self, Display};
use std::str::FromStr;

use egg::{Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp, SexpError};

use crate::origin::OriginNode;

/// A node in a labeled, ordered tree.
#[derive(Debug, Clone, Serialize, std::hash::Hash, PartialEq, Eq)]
#[serde(bound(deserialize = "L: Language"))]
pub struct Tree<L: Language> {
    pub(super) node: L,
    pub(super) ty: Option<Box<Tree<L>>>,
    pub(super) children: Vec<Tree<L>>,
}

impl<L: Language> Tree<L> {
    /// Create a leaf node with no children.
    pub fn leaf_untyped(node: L) -> Self {
        Tree {
            node,
            ty: None,
            children: Vec::new(),
        }
    }

    /// Create a leaf node with no children.
    pub fn leaf_typed(node: L, ty: Option<Tree<L>>) -> Self {
        Tree {
            node,
            ty: ty.map(Box::new),
            children: Vec::new(),
        }
    }

    /// Create a node with the given children.
    pub fn new_untyped(node: L, children: Vec<Tree<L>>) -> Self {
        Tree {
            node,
            ty: None,
            children,
        }
    }

    /// Create a node with the given children.
    pub fn new_typed(node: L, children: Vec<Tree<L>>, ty: Option<Tree<L>>) -> Self {
        Tree {
            node,
            ty: ty.map(Box::new),
            children,
        }
    }

    fn from_rec_expr(expr: &RecExpr<OriginNode<L>>) -> Self {
        fn rec<LL: Language>(expr: &RecExpr<OriginNode<LL>>, id: Id) -> Tree<LL> {
            let OriginNode {
                node,
                ty,
                origin: _,
            } = expr[id].clone();
            let children = node
                .children()
                .iter()
                .map(|c_id| rec(expr, *c_id))
                .collect();

            Tree {
                node: node.map_children(|_| Id::from(0)),
                ty: ty.map(|ty_id| Box::new(rec(expr, ty_id))),
                children,
            }
        }
        rec(expr, Id::from(expr.len() - 1))
    }

    pub fn size(&self, with_types: bool) -> usize {
        if with_types {
            self.size_with_types()
        } else {
            self.size_without_types()
        }
    }

    /// Count total number of nodes in this tree.
    pub fn size_without_types(&self) -> usize {
        1 + self
            .children()
            .iter()
            .map(Self::size_without_types)
            .sum::<usize>()
    }

    pub fn size_with_types(&self) -> usize {
        1 + self
            .children()
            .iter()
            .map(Self::size_with_types)
            .sum::<usize>()
            + self.ty().map_or(0, |t| t.size_with_types())
    }

    // /// Build a type tree from an e-class's type annotation.
    // #[must_use]
    // pub fn from_eclass(graph: &EGraph<L>, id: EClassId) -> Option<Self> {
    //     let ty_id = graph.class(id).ty()?;
    //     Some(Self::from_type(graph, *ty_id))
    // }

    // fn from_type(graph: &EGraph<L>, id: TypeChildId) -> Self {
    //     match id {
    //         TypeChildId::Nat(nat_id) => Self::from_nat(graph, nat_id),
    //         TypeChildId::Type(fun_ty_id) => Self::from_fun(graph, fun_ty_id),
    //         TypeChildId::Data(data_ty_id) => Self::from_data(graph, data_ty_id),
    //     }
    // }

    // fn from_fun(graph: &EGraph<L>, id: FunId) -> Self {
    //     let node = graph.fun_ty(id).label().to_owned();
    //     let children = graph
    //         .fun_ty(id)
    //         .children()
    //         .iter()
    //         .map(|&c_id| Self::from_type(graph, c_id))
    //         .collect();
    //     Tree::new_untyped(node, children)
    // }

    // #[must_use]
    // pub fn from_data(graph: &EGraph<L>, id: DataId) -> Self {
    //     let node = graph.data_ty(id).label().to_owned();
    //     let children = graph
    //         .data_ty(id)
    //         .children()
    //         .iter()
    //         .map(|&c_id| match c_id {
    //             DataChildId::Nat(nat_id) => Self::from_nat(graph, nat_id),
    //             DataChildId::DataType(data_ty_id) => Self::from_data(graph, data_ty_id),
    //         })
    //         .collect();
    //     Tree::new_untyped(node, children)
    // }

    // #[must_use]
    // pub fn from_nat(graph: &EGraph<L>, id: NatId) -> Self {
    //     let node = graph.nat(id).label().to_owned();
    //     let children = graph
    //         .nat(id)
    //         .children()
    //         .iter()
    //         .map(|&c_id| Self::from_nat(graph, c_id))
    //         .collect();
    //     Tree::new_untyped(node, children)
    // }

    /// Returns true if this node has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn children(&self) -> &[Self] {
        &self.children
    }

    pub fn node(&self) -> &L {
        &self.node
    }

    pub fn ty(&self) -> Option<&Tree<L>> {
        self.ty.as_deref()
    }
}

impl<L: Language + Display> Display for Tree<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ty.is_some() {
            write!(f, "(type_of ",)?;
        }
        if self.is_leaf() {
            write!(f, "{}", self.node)?;
        } else {
            write!(f, "({}", self.node)?;
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

impl<L> FromStr for Tree<L>
where
    L: Language + FromStr,
    <L as FromStr>::Err: Display,
{
    type Err = SexpError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        /// Parse a type tree (no typeOf wrappers).
        fn parse_expr<L>(sexp: Sexp) -> Result<Tree<L>, SexpError>
        where
            L: Language + FromStr,
            <L as FromStr>::Err: Display,
        {
            match sexp {
                Sexp::String(s) => Ok(Tree::leaf_untyped(
                    s.parse::<L>()
                        .map_err(|e| SexpError::Other(e.to_string()))?,
                )),
                Sexp::List(mut sexps) => {
                    if sexps.len() == 3
                        && let Some(Sexp::String(s)) = sexps.first()
                        && s == "type_of"
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
                    let Some(Sexp::String(node)) = iter.next() else {
                        return Err(SexpError::Other("expected (label ...)".to_owned()));
                    };
                    Ok(Tree::new_untyped(
                        node.parse::<L>()
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

// impl IntoSexp for Tree<String> {
//     fn into_sexp(&self) -> Sexp {
//         if self.is_leaf() {
//             // Leaf with no type - just the label
//             Sexp::String(self.label.clone())
//         } else {
//             Sexp::List(
//                 vec![Sexp::String(self.label.clone())]
//                     .into_iter()
//                     .chain(self.children().iter().map(Sexp::from))
//                     .collect::<Vec<_>>(),
//             )
//         }
//     }
// }
