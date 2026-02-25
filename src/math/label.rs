use std::fmt::{self, Display};
use std::str::FromStr;

use egg::Symbol;
use serde::{Deserialize, Serialize};

use crate::Label;

use super::{Constant, Math};

impl Math {
    pub(super) fn to_label(s: &Self) -> MathLabel {
        match s {
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
