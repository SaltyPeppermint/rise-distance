use std::fmt::{self, Display};
use std::str::FromStr;

use egg::{Id, Symbol};
use serde::{Deserialize, Serialize};

use crate::Label;
use crate::egg::ToEgg;
use crate::tree::TreeShaped;

use super::{Constant, Math};

impl From<&Math> for MathLabel {
    fn from(value: &Math) -> Self {
        match value {
            Math::Diff(_) => MathLabel::Diff,
            Math::Integral(_) => MathLabel::Integral,
            Math::Add(_) => MathLabel::Add,
            Math::Sub(_) => MathLabel::Sub,
            Math::Mul(_) => MathLabel::Mul,
            Math::Div(_) => MathLabel::Div,
            Math::Pow(_) => MathLabel::Pow,
            Math::Ln(_) => MathLabel::Ln,
            Math::Sqrt(_) => MathLabel::Sqrt,
            Math::Sin(_) => MathLabel::Sin,
            Math::Cos(_) => MathLabel::Cos,
            Math::Constant(not_nan) => MathLabel::Constant(*not_nan),
            Math::Symbol(global_symbol) => MathLabel::Symbol(*global_symbol),
        }
    }
}

// impl From<Math> for MathLabel {
//     fn from(value: Math) -> Self {
//         Self::from(&value)
//     }
// }

#[derive(Clone, Copy, Serialize, Deserialize, Debug, Hash, PartialEq, Eq)]
pub enum MathLabel {
    Diff,
    Integral,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Ln,
    Sqrt,
    Sin,
    Cos,
    Constant(Constant),
    Symbol(Symbol),
}

impl Label for MathLabel {
    fn type_of() -> Self {
        panic!("No types to see here");
    }
}

impl Display for MathLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathLabel::Diff => write!(f, "d"),
            MathLabel::Integral => write!(f, "i"),
            MathLabel::Add => write!(f, "+"),
            MathLabel::Sub => write!(f, "-"),
            MathLabel::Mul => write!(f, "*"),
            MathLabel::Div => write!(f, "/"),
            MathLabel::Pow => write!(f, "pow"),
            MathLabel::Ln => write!(f, "ln"),
            MathLabel::Sqrt => write!(f, "sqrt"),
            MathLabel::Sin => write!(f, "sin"),
            MathLabel::Cos => write!(f, "cos"),
            MathLabel::Constant(c) => write!(f, "{c}"),
            MathLabel::Symbol(s) => write!(f, "{s}"),
        }
    }
}

impl FromStr for MathLabel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "d" => MathLabel::Diff,
            "i" => MathLabel::Integral,
            "+" => MathLabel::Add,
            "-" => MathLabel::Sub,
            "*" => MathLabel::Mul,
            "/" => MathLabel::Div,
            "pow" => MathLabel::Pow,
            "ln" => MathLabel::Ln,
            "sqrt" => MathLabel::Sqrt,
            "sin" => MathLabel::Sin,
            "cos" => MathLabel::Cos,
            _ => {
                if let Ok(c) = s.parse::<Constant>() {
                    MathLabel::Constant(c)
                } else {
                    MathLabel::Symbol(s.into())
                }
            }
        })
    }
}

impl<T: TreeShaped<MathLabel>> ToEgg<MathLabel> for T {
    type Lang = Math;

    fn add_node<F: FnMut(&Self, Self::Lang) -> Id>(&self, adder: &mut F) -> Id {
        let child_ids = self
            .children()
            .iter()
            .map(|c| c.add_node(adder))
            .collect::<Vec<_>>();
        let math_node = match self.label() {
            MathLabel::Diff => Math::Diff([child_ids[0], child_ids[1]]),
            MathLabel::Integral => Math::Integral([child_ids[0], child_ids[1]]),
            MathLabel::Add => Math::Add([child_ids[0], child_ids[1]]),
            MathLabel::Sub => Math::Sub([child_ids[0], child_ids[1]]),
            MathLabel::Mul => Math::Mul([child_ids[0], child_ids[1]]),
            MathLabel::Div => Math::Div([child_ids[0], child_ids[1]]),
            MathLabel::Pow => Math::Pow([child_ids[0], child_ids[1]]),
            MathLabel::Ln => Math::Ln(child_ids[0]),
            MathLabel::Sqrt => Math::Sqrt(child_ids[0]),
            MathLabel::Sin => Math::Sin(child_ids[0]),
            MathLabel::Cos => Math::Cos(child_ids[0]),
            MathLabel::Constant(c) => Math::Constant(*c),
            MathLabel::Symbol(s) => Math::Symbol(*s),
        };
        adder(self, math_node)
    }
}

#[cfg(test)]
mod tests {
    use egg::RecExpr;

    use crate::TreeNode;

    use super::*;

    fn leaf(label: MathLabel) -> TreeNode<MathLabel> {
        TreeNode::leaf_untyped(label)
    }

    fn node(label: MathLabel, children: Vec<TreeNode<MathLabel>>) -> TreeNode<MathLabel> {
        TreeNode::new_untyped(label, children)
    }

    fn sym(s: &str) -> TreeNode<MathLabel> {
        leaf(MathLabel::Symbol(s.into()))
    }

    /// Build tree, convert to `RecExpr`, check it matches the directly parsed `RecExpr`.
    fn assert_eq_recexpr(tree: &TreeNode<MathLabel>, expected_str: &str) {
        let from_tree: RecExpr<Math> = (tree).to_rec_expr();
        let direct: RecExpr<Math> = expected_str.parse().unwrap();
        assert_eq!(from_tree, direct, "mismatch for {expected_str}");
    }

    #[test]
    fn leaf_symbol() {
        assert_eq_recexpr(&sym("x"), "x");
    }

    #[test]
    fn leaf_constant() {
        assert_eq_recexpr(&leaf(MathLabel::Constant("42".parse().unwrap())), "42");
    }

    #[test]
    fn binary_add() {
        assert_eq_recexpr(&node(MathLabel::Add, vec![sym("x"), sym("y")]), "(+ x y)");
    }

    #[test]
    fn unary_ln() {
        assert_eq_recexpr(&node(MathLabel::Ln, vec![sym("x")]), "(ln x)");
    }

    #[test]
    fn unary_sqrt() {
        assert_eq_recexpr(&node(MathLabel::Sqrt, vec![sym("x")]), "(sqrt x)");
    }

    #[test]
    fn nested() {
        // (+ (* x 1) y)
        let one = leaf(MathLabel::Constant("1".parse().unwrap()));
        let mul = node(MathLabel::Mul, vec![sym("x"), one]);
        let add = node(MathLabel::Add, vec![mul, sym("y")]);
        assert_eq_recexpr(&add, "(+ (* x 1) y)");
    }

    #[test]
    fn deeply_nested() {
        // (d (sin (+ x y)) x)
        let sum = node(MathLabel::Add, vec![sym("x"), sym("y")]);
        let sin = node(MathLabel::Sin, vec![sum]);
        let diff = node(MathLabel::Diff, vec![sin, sym("x")]);
        assert_eq_recexpr(&diff, "(d (sin (+ x y)) x)");
    }

    #[test]
    fn all_binary_ops() {
        let ops = [
            (MathLabel::Add, "+"),
            (MathLabel::Sub, "-"),
            (MathLabel::Mul, "*"),
            (MathLabel::Div, "/"),
            (MathLabel::Pow, "pow"),
            (MathLabel::Diff, "d"),
            (MathLabel::Integral, "i"),
        ];
        for (label, op_str) in ops {
            let tree = node(label, vec![sym("x"), sym("y")]);
            assert_eq_recexpr(&tree, &format!("({op_str} x y)"));
        }
    }

    #[test]
    fn all_unary_ops() {
        let ops = [
            (MathLabel::Ln, "ln"),
            (MathLabel::Sqrt, "sqrt"),
            (MathLabel::Sin, "sin"),
            (MathLabel::Cos, "cos"),
        ];
        for (label, op_str) in ops {
            let tree = node(label, vec![sym("x")]);
            assert_eq_recexpr(&tree, &format!("({op_str} x)"));
        }
    }
}
