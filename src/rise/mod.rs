//! Typed representation of the Rise language for tree edit distance computation.
//!
//! This module provides a proper typed AST representation of Rise expressions,
//! with S-expression parsing and serialization support.

mod address;
mod expr;
mod label;
mod nat;
mod primitive;
mod types;

use std::num::ParseIntError;

pub use address::Address;
pub use expr::{Expr, ExprNode, LiteralData};
pub use label::RiseLabel;
pub use nat::Nat;
pub use primitive::Primitive;
pub use types::{DataType, ScalarType, Type};

use symbolic_expressions::SexpError;
use thiserror::Error;

/// Error type for parsing Rise expressions.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseError {
    #[error("S-expression parse error: {0}")]
    Sexp(String),
    #[error("invalid expression: {0}")]
    Expr(String),
    #[error("invalid primitive: {0}")]
    Prim(String),
    #[error("invalid type: {0}")]
    Type(String),
    #[error("invalid nat: {0}")]
    Nat(String),
    #[error("invalid address: {0}")]
    Address(String),
    #[error("invalid label: {0}")]
    Label(String),
    #[error("invalid variable index '{input}': {reason}")]
    VarIndex {
        input: String,
        reason: ParseIntError,
    },
    #[error("invalid literal '{input}': {reason}")]
    Literal {
        input: String,
        reason: ParseIntError,
    },
}

impl From<SexpError> for ParseError {
    fn from(e: SexpError) -> Self {
        ParseError::Sexp(format!("{e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tree::{Tree, TreeShaped};
    use crate::zs::tree_distance_unit;

    #[test]
    fn rise_label_works_with_zs() {
        // Create two simple trees with RiseLabel
        let tree1 = Tree::new_untyped(
            RiseLabel::App,
            vec![
                Tree::leaf_untyped(RiseLabel::Primitive(Primitive::Map)),
                Tree::new_untyped(
                    RiseLabel::Lambda,
                    vec![Tree::leaf_untyped(RiseLabel::Var(0))],
                ),
            ],
        )
        .flatten(false);

        let tree2 = Tree::new_untyped(
            RiseLabel::App,
            vec![
                Tree::leaf_untyped(RiseLabel::Primitive(Primitive::Map)),
                Tree::new_untyped(
                    RiseLabel::Lambda,
                    vec![Tree::leaf_untyped(RiseLabel::Var(1))],
                ),
            ],
        )
        .flatten(false);

        // Same structure, different variable index - should be distance 1
        let distance = tree_distance_unit(&tree1, &tree2);
        assert_eq!(distance, 1);

        // Identical trees should have distance 0
        let distance_same = tree_distance_unit(&tree1, &tree1);
        assert_eq!(distance_same, 0);
    }

    #[test]
    fn rise_label_with_floats_in_zs() {
        use ordered_float::OrderedFloat;

        let tree1 = Tree::leaf_untyped(RiseLabel::FloatLit(OrderedFloat(3.11))).flatten(false);
        let tree2 = Tree::leaf_untyped(RiseLabel::FloatLit(OrderedFloat(3.11))).flatten(false);
        let tree3 = Tree::leaf_untyped(RiseLabel::FloatLit(OrderedFloat(2.71))).flatten(false);

        // Same float value - distance 0
        assert_eq!(tree_distance_unit(&tree1, &tree2), 0);

        // Different float value - distance 1 (relabel)
        assert_eq!(tree_distance_unit(&tree1, &tree3), 1);
    }

    #[test]
    fn rise_expr_to_tree_with_zs() {
        let expr1 = "(app map (lam $e0))".parse::<Expr>().unwrap();
        let expr2 = "(app map (lam $e1))".parse::<Expr>().unwrap();

        let tree1 = expr1.to_tree().flatten(false);
        let tree2 = expr2.to_tree().flatten(false);

        // Different variable index - distance 1
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1);
    }
}
