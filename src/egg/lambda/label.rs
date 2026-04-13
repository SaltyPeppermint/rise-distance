use std::fmt::{self, Display};
use std::str::FromStr;

use egg::{Id, Symbol};
use serde::{Deserialize, Serialize};

use crate::Label;
use crate::egg::ToEgg;
use crate::tree::TreeShaped;

use super::Lambda;

impl From<&Lambda> for LambdaLabel {
    fn from(value: &Lambda) -> Self {
        match value {
            Lambda::Bool(b) => LambdaLabel::Bool(*b),
            Lambda::Num(n) => LambdaLabel::Num(*n),
            Lambda::Var(_) => LambdaLabel::Var,
            Lambda::Add(_) => LambdaLabel::Add,
            Lambda::Eq(_) => LambdaLabel::Eq,
            Lambda::App(_) => LambdaLabel::App,
            Lambda::Lambda(_) => LambdaLabel::Lam,
            Lambda::Let(_) => LambdaLabel::Let,
            Lambda::Fix(_) => LambdaLabel::Fix,
            Lambda::If(_) => LambdaLabel::If,
            Lambda::Symbol(s) => LambdaLabel::Symbol(*s),
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, Hash, PartialEq, Eq)]
pub enum LambdaLabel {
    Bool(bool),
    Num(i32),
    Var,
    Add,
    Eq,
    App,
    Lam,
    Let,
    Fix,
    If,
    Symbol(Symbol),
}

impl Label for LambdaLabel {
    fn type_of() -> Self {
        panic!("No types to see here");
    }
}

impl Display for LambdaLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LambdaLabel::Bool(b) => write!(f, "{b}"),
            LambdaLabel::Num(n) => write!(f, "{n}"),
            LambdaLabel::Var => write!(f, "var"),
            LambdaLabel::Add => write!(f, "+"),
            LambdaLabel::Eq => write!(f, "="),
            LambdaLabel::App => write!(f, "app"),
            LambdaLabel::Lam => write!(f, "lam"),
            LambdaLabel::Let => write!(f, "let"),
            LambdaLabel::Fix => write!(f, "fix"),
            LambdaLabel::If => write!(f, "if"),
            LambdaLabel::Symbol(s) => write!(f, "{s}"),
        }
    }
}

impl FromStr for LambdaLabel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "true" => LambdaLabel::Bool(true),
            "false" => LambdaLabel::Bool(false),
            "var" => LambdaLabel::Var,
            "+" => LambdaLabel::Add,
            "=" => LambdaLabel::Eq,
            "app" => LambdaLabel::App,
            "lam" => LambdaLabel::Lam,
            "let" => LambdaLabel::Let,
            "fix" => LambdaLabel::Fix,
            "if" => LambdaLabel::If,
            _ => {
                if let Ok(n) = s.parse::<i32>() {
                    LambdaLabel::Num(n)
                } else {
                    LambdaLabel::Symbol(s.into())
                }
            }
        })
    }
}

impl<T: TreeShaped<LambdaLabel>> ToEgg<LambdaLabel> for T {
    type Lang = Lambda;

    fn add_node<F: FnMut(&Self, Self::Lang) -> Id>(&self, adder: &mut F) -> Id {
        let child_ids = self
            .children()
            .iter()
            .map(|c| c.add_node(adder))
            .collect::<Vec<_>>();
        let lambda_node = match self.label() {
            LambdaLabel::Bool(b) => Lambda::Bool(*b),
            LambdaLabel::Num(n) => Lambda::Num(*n),
            LambdaLabel::Var => Lambda::Var(child_ids[0]),
            LambdaLabel::Add => Lambda::Add([child_ids[0], child_ids[1]]),
            LambdaLabel::Eq => Lambda::Eq([child_ids[0], child_ids[1]]),
            LambdaLabel::App => Lambda::App([child_ids[0], child_ids[1]]),
            LambdaLabel::Lam => Lambda::Lambda([child_ids[0], child_ids[1]]),
            LambdaLabel::Let => Lambda::Let([child_ids[0], child_ids[1], child_ids[2]]),
            LambdaLabel::Fix => Lambda::Fix([child_ids[0], child_ids[1]]),
            LambdaLabel::If => Lambda::If([child_ids[0], child_ids[1], child_ids[2]]),
            LambdaLabel::Symbol(s) => Lambda::Symbol(*s),
        };
        adder(self, lambda_node)
    }
}

#[cfg(test)]
mod tests {
    use egg::RecExpr;

    use crate::Tree;

    use super::*;

    fn leaf(label: LambdaLabel) -> Tree<LambdaLabel> {
        Tree::leaf_untyped(label)
    }

    fn node(label: LambdaLabel, children: Vec<Tree<LambdaLabel>>) -> Tree<LambdaLabel> {
        Tree::new_untyped(label, children)
    }

    fn sym(s: &str) -> Tree<LambdaLabel> {
        leaf(LambdaLabel::Symbol(s.into()))
    }

    fn assert_eq_recexpr(tree: &Tree<LambdaLabel>, expected_str: &str) {
        let from_tree: RecExpr<Lambda> = tree.to_rec_expr();
        let direct: RecExpr<Lambda> = expected_str.parse().unwrap();
        assert_eq!(from_tree, direct, "mismatch for {expected_str}");
    }

    #[test]
    fn leaf_symbol() {
        assert_eq_recexpr(&sym("x"), "x");
    }

    #[test]
    fn leaf_bool_true() {
        assert_eq_recexpr(&leaf(LambdaLabel::Bool(true)), "true");
    }

    #[test]
    fn leaf_bool_false() {
        assert_eq_recexpr(&leaf(LambdaLabel::Bool(false)), "false");
    }

    #[test]
    fn leaf_num() {
        assert_eq_recexpr(&leaf(LambdaLabel::Num(42)), "42");
    }

    #[test]
    fn unary_var() {
        assert_eq_recexpr(&node(LambdaLabel::Var, vec![sym("x")]), "(var x)");
    }

    #[test]
    fn binary_add() {
        assert_eq_recexpr(&node(LambdaLabel::Add, vec![sym("x"), sym("y")]), "(+ x y)");
    }

    #[test]
    fn binary_eq() {
        assert_eq_recexpr(&node(LambdaLabel::Eq, vec![sym("x"), sym("y")]), "(= x y)");
    }

    #[test]
    fn binary_app() {
        assert_eq_recexpr(
            &node(LambdaLabel::App, vec![sym("f"), sym("x")]),
            "(app f x)",
        );
    }

    #[test]
    fn binary_lam() {
        assert_eq_recexpr(
            &node(LambdaLabel::Lam, vec![sym("x"), sym("body")]),
            "(lam x body)",
        );
    }

    #[test]
    fn binary_fix() {
        assert_eq_recexpr(
            &node(LambdaLabel::Fix, vec![sym("f"), sym("body")]),
            "(fix f body)",
        );
    }

    #[test]
    fn ternary_let() {
        assert_eq_recexpr(
            &node(LambdaLabel::Let, vec![sym("x"), sym("e"), sym("body")]),
            "(let x e body)",
        );
    }

    #[test]
    fn ternary_if() {
        assert_eq_recexpr(
            &node(
                LambdaLabel::If,
                vec![leaf(LambdaLabel::Bool(true)), sym("then"), sym("else")],
            ),
            "(if true then else)",
        );
    }

    #[test]
    fn nested() {
        // (app (lam x (var x)) 0)
        let var_x = node(LambdaLabel::Var, vec![sym("x")]);
        let lam = node(LambdaLabel::Lam, vec![sym("x"), var_x]);
        let zero = leaf(LambdaLabel::Num(0));
        let app = node(LambdaLabel::App, vec![lam, zero]);
        assert_eq_recexpr(&app, "(app (lam x (var x)) 0)");
    }

    #[test]
    fn roundtrip_from_lambda() {
        // Build Lambda nodes and convert back via LambdaLabel::from
        assert_eq!(
            LambdaLabel::from(&Lambda::Bool(true)),
            LambdaLabel::Bool(true)
        );
        assert_eq!(LambdaLabel::from(&Lambda::Num(-1)), LambdaLabel::Num(-1));
        assert_eq!(
            LambdaLabel::from(&Lambda::Symbol("z".into())),
            LambdaLabel::Symbol("z".into())
        );
    }

    #[test]
    fn fromstr_roundtrip() {
        let cases: &[(&str, LambdaLabel)] = &[
            ("true", LambdaLabel::Bool(true)),
            ("false", LambdaLabel::Bool(false)),
            ("7", LambdaLabel::Num(7)),
            ("var", LambdaLabel::Var),
            ("+", LambdaLabel::Add),
            ("=", LambdaLabel::Eq),
            ("app", LambdaLabel::App),
            ("lam", LambdaLabel::Lam),
            ("let", LambdaLabel::Let),
            ("fix", LambdaLabel::Fix),
            ("if", LambdaLabel::If),
            ("foo", LambdaLabel::Symbol("foo".into())),
        ];
        for (s, expected) in cases {
            let parsed: LambdaLabel = s.parse().unwrap();
            assert_eq!(parsed, *expected, "fromstr failed for {s:?}");
            assert_eq!(expected.to_string(), *s, "display failed for {expected:?}");
        }
    }
}
