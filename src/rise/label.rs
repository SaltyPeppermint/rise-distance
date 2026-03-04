//! Label type for Rise that implements the Label trait.

use std::fmt::{self, Display};
use std::str::FromStr;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::address::Address;
use super::expr::{Expr, ExprNode, LiteralData};
use super::nat::Nat;
use super::primitive::Primitive;
use super::types::{DataType, ScalarType, Type};
use crate::nodes::Label;
use crate::tree::TreeNode;

/// A compact label type for Rise that implements the Label trait.
///
/// This allows using Rise expressions directly with the e-graph infrastructure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RiseLabel {
    // Expression labels
    Var(usize),
    App,
    Lambda,
    NatApp,
    NatLambda,
    DataApp,
    DataLambda,
    AddrApp,
    AddrLambda,
    NatNatLambda,
    IndexLiteral,
    TypeOf,
    // Literals
    BoolLit(bool),
    IntLit(i32),
    FloatLit(OrderedFloat<f32>),
    DoubleLit(OrderedFloat<f64>),
    NatLit(i64),
    // Primitives
    Primitive(Primitive),
    // Nat labels
    NatVar(usize),
    NatCst(i64),
    NatAdd,
    NatMul,
    NatPow,
    NatMod,
    NatFloorDiv,
    // Address labels
    AddrVar(usize),
    Global,
    Local,
    Private,
    Constant,
    // Type labels
    Fun,
    NatFun,
    DataFun,
    AddrFun,
    NatNatFun,
    // Data type labels
    DataVar(usize),
    Scalar(ScalarType),
    NatT,
    IdxT,
    PairT,
    ArrT,
    VecT,
}

impl Label for RiseLabel {
    fn type_of() -> Self {
        RiseLabel::TypeOf
    }
}

#[expect(clippy::match_same_arms)]
impl Display for RiseLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiseLabel::Var(i) => write!(f, "$e{i}"),
            RiseLabel::App => write!(f, "app"),
            RiseLabel::Lambda => write!(f, "lam"),
            RiseLabel::NatApp => write!(f, "natApp"),
            RiseLabel::NatLambda => write!(f, "natLam"),
            RiseLabel::DataApp => write!(f, "dataApp"),
            RiseLabel::DataLambda => write!(f, "dataLam"),
            RiseLabel::AddrApp => write!(f, "addrApp"),
            RiseLabel::AddrLambda => write!(f, "addrLam"),
            RiseLabel::NatNatLambda => write!(f, "natNatLam"),
            RiseLabel::IndexLiteral => write!(f, "idxL"),
            RiseLabel::TypeOf => write!(f, "typeOf"),
            RiseLabel::BoolLit(b) => write!(f, "{b}"),
            RiseLabel::IntLit(i) => write!(f, "{i}i"),
            RiseLabel::FloatLit(n) => write!(f, "{}f", n.0),
            RiseLabel::DoubleLit(n) => write!(f, "{}d", n.0),
            RiseLabel::NatLit(n) => write!(f, "{n}n"),
            RiseLabel::Primitive(p) => write!(f, "{p}"),
            RiseLabel::NatVar(i) => write!(f, "$n{i}"),
            RiseLabel::NatCst(n) => write!(f, "{n}n"),
            RiseLabel::NatAdd => write!(f, "natAdd"),
            RiseLabel::NatMul => write!(f, "natMul"),
            RiseLabel::NatPow => write!(f, "natPow"),
            RiseLabel::NatMod => write!(f, "natMod"),
            RiseLabel::NatFloorDiv => write!(f, "natFloorDiv"),
            RiseLabel::AddrVar(i) => write!(f, "$a{i}"),
            RiseLabel::Global => write!(f, "global"),
            RiseLabel::Local => write!(f, "local"),
            RiseLabel::Private => write!(f, "private"),
            RiseLabel::Constant => write!(f, "constant"),
            RiseLabel::Fun => write!(f, "fun"),
            RiseLabel::NatFun => write!(f, "natFun"),
            RiseLabel::DataFun => write!(f, "dataFun"),
            RiseLabel::AddrFun => write!(f, "addrFun"),
            RiseLabel::NatNatFun => write!(f, "natNatFun"),
            RiseLabel::DataVar(i) => write!(f, "$d{i}"),
            RiseLabel::Scalar(s) => write!(f, "{s}"),
            RiseLabel::NatT => write!(f, "natT"),
            RiseLabel::IdxT => write!(f, "idxT"),
            RiseLabel::PairT => write!(f, "pairT"),
            RiseLabel::ArrT => write!(f, "arrT"),
            RiseLabel::VecT => write!(f, "vecT"),
        }
    }
}

impl FromStr for RiseLabel {
    type Err = super::ParseError;

    #[expect(clippy::too_many_lines)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use super::ParseError;

        // Expression variable: $e<index>
        if let Some(rest) = s.strip_prefix("$e") {
            return rest.parse::<usize>().map(RiseLabel::Var).map_err(|reason| {
                ParseError::VarIndex {
                    input: s.to_owned(),
                    reason,
                }
            });
        }

        // Nat variable: $n<index>
        if let Some(rest) = s.strip_prefix("$n") {
            return rest
                .parse::<usize>()
                .map(RiseLabel::NatVar)
                .map_err(|reason| ParseError::VarIndex {
                    input: s.to_owned(),
                    reason,
                });
        }

        // Data variable: $d<index>
        if let Some(rest) = s.strip_prefix("$d") {
            return rest
                .parse::<usize>()
                .map(RiseLabel::DataVar)
                .map_err(|reason| ParseError::VarIndex {
                    input: s.to_owned(),
                    reason,
                });
        }

        // Address variable: $a<index>
        if let Some(rest) = s.strip_prefix("$a") {
            return rest
                .parse::<usize>()
                .map(RiseLabel::AddrVar)
                .map_err(|reason| ParseError::VarIndex {
                    input: s.to_owned(),
                    reason,
                });
        }

        // Integer literal: <n>i
        if let Some(num) = s.strip_suffix('i')
            && let Ok(value) = num.parse::<i32>()
        {
            return Ok(RiseLabel::IntLit(value));
        }

        // Float literal: <n>f
        if let Some(num) = s.strip_suffix('f')
            && let Ok(value) = num.parse::<f32>()
        {
            return Ok(RiseLabel::FloatLit(OrderedFloat(value)));
        }

        // Double literal: <n>d
        if let Some(num) = s.strip_suffix('d')
            && let Ok(value) = num.parse::<f64>()
        {
            return Ok(RiseLabel::DoubleLit(OrderedFloat(value)));
        }

        // Nat literal: <n>n
        if let Some(num) = s.strip_suffix('n')
            && let Ok(value) = num.parse::<i64>()
        {
            return Ok(RiseLabel::NatCst(value));
        }

        // Keywords and primitives
        Ok(match s {
            // Expression labels
            "app" => RiseLabel::App,
            "lam" => RiseLabel::Lambda,
            "natApp" => RiseLabel::NatApp,
            "natLam" => RiseLabel::NatLambda,
            "dataApp" => RiseLabel::DataApp,
            "dataLam" => RiseLabel::DataLambda,
            "addrApp" => RiseLabel::AddrApp,
            "addrLam" => RiseLabel::AddrLambda,
            "natNatLam" => RiseLabel::NatNatLambda,
            "idxL" => RiseLabel::IndexLiteral,
            "typeOf" => RiseLabel::TypeOf,

            // Boolean literals
            "true" => RiseLabel::BoolLit(true),
            "false" => RiseLabel::BoolLit(false),

            // Nat operations
            "natAdd" => RiseLabel::NatAdd,
            "natMul" => RiseLabel::NatMul,
            "natPow" => RiseLabel::NatPow,
            "natMod" => RiseLabel::NatMod,
            "natFloorDiv" => RiseLabel::NatFloorDiv,

            // Address labels
            "global" => RiseLabel::Global,
            "local" => RiseLabel::Local,
            "private" => RiseLabel::Private,
            "constant" => RiseLabel::Constant,

            // Type labels
            "fun" => RiseLabel::Fun,
            "natFun" => RiseLabel::NatFun,
            "dataFun" => RiseLabel::DataFun,
            "addrFun" => RiseLabel::AddrFun,
            "natNatFun" => RiseLabel::NatNatFun,

            // Scalar types
            "bool" => RiseLabel::Scalar(ScalarType::Bool),
            "i8" => RiseLabel::Scalar(ScalarType::I8),
            "i16" => RiseLabel::Scalar(ScalarType::I16),
            "i32" => RiseLabel::Scalar(ScalarType::I32),
            "i64" => RiseLabel::Scalar(ScalarType::I64),
            "u8" => RiseLabel::Scalar(ScalarType::U8),
            "u16" => RiseLabel::Scalar(ScalarType::U16),
            "u32" => RiseLabel::Scalar(ScalarType::U32),
            "u64" => RiseLabel::Scalar(ScalarType::U64),
            "f16" => RiseLabel::Scalar(ScalarType::F16),
            "f32" => RiseLabel::Scalar(ScalarType::F32),
            "f64" => RiseLabel::Scalar(ScalarType::F64),

            // Data type labels
            "natT" => RiseLabel::NatT,
            "idxT" => RiseLabel::IdxT,
            "pairT" => RiseLabel::PairT,
            "arrT" => RiseLabel::ArrT,
            "vecT" => RiseLabel::VecT,

            // // Float/Double without suffix (decimal number like "0.0")
            // _ if s.contains('.') || s.contains('E') || s.contains('e') => {
            //     if let Ok(value) = s.parse::<f32>() {
            //         RiseLabel::FloatLit(OrderedFloat(value))
            //     } else {
            //         // Try as primitive
            //         RiseLabel::Primitive(Primitive::from_name(s)?)
            //     }
            // }

            // Everything else is a primitive
            _ => RiseLabel::Primitive(Primitive::from_name(s)?),
        })
    }
}

impl Serialize for RiseLabel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for RiseLabel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        RiseLabel::from_str(&s).map_err(serde::de::Error::custom)
    }
}

impl From<&TreeNode<RiseLabel>> for Expr {
    fn from(tree: &TreeNode<RiseLabel>) -> Self {
        let ty = tree.ty().map(tree_to_type);
        let node = match tree.label() {
            // Leaf expression labels
            RiseLabel::Var(i) => ExprNode::Var(*i),
            RiseLabel::BoolLit(b) => ExprNode::Literal(LiteralData::Bool(*b)),
            RiseLabel::IntLit(n) => ExprNode::Literal(LiteralData::Int(*n)),
            RiseLabel::FloatLit(f) => ExprNode::Literal(LiteralData::Float(*f)),
            RiseLabel::DoubleLit(d) => ExprNode::Literal(LiteralData::Double(*d)),
            RiseLabel::NatLit(n) => ExprNode::Literal(LiteralData::Nat(*n)),
            RiseLabel::Primitive(p) => ExprNode::Primitive(p.clone()),

            // Nat labels in expression position -> NatLiteral
            RiseLabel::NatVar(_)
            | RiseLabel::NatCst(_)
            | RiseLabel::NatAdd
            | RiseLabel::NatMul
            | RiseLabel::NatPow
            | RiseLabel::NatMod
            | RiseLabel::NatFloorDiv => ExprNode::NatLiteral(tree_to_nat(tree)),

            // Binary expression: (app f e)
            RiseLabel::App => {
                let f = Expr::from(&tree.children()[0]);
                let e = Expr::from(&tree.children()[1]);
                ExprNode::App(Box::new(f), Box::new(e))
            }

            // Unary expression lambdas
            RiseLabel::Lambda => ExprNode::Lambda(Box::new(Expr::from(&tree.children()[0]))),
            RiseLabel::NatLambda => ExprNode::NatLambda(Box::new(Expr::from(&tree.children()[0]))),
            RiseLabel::DataLambda => {
                ExprNode::DataLambda(Box::new(Expr::from(&tree.children()[0])))
            }
            RiseLabel::AddrLambda => {
                ExprNode::AddrLambda(Box::new(Expr::from(&tree.children()[0])))
            }
            RiseLabel::NatNatLambda => {
                ExprNode::NatNatLambda(Box::new(Expr::from(&tree.children()[0])))
            }

            // Application with heterogeneous second child
            RiseLabel::NatApp => {
                let f = Expr::from(&tree.children()[0]);
                let n = tree_to_nat(&tree.children()[1]);
                ExprNode::NatApp(Box::new(f), n)
            }
            RiseLabel::DataApp => {
                let f = Expr::from(&tree.children()[0]);
                let dt = tree_to_data_type(&tree.children()[1]);
                ExprNode::DataApp(Box::new(f), dt)
            }
            RiseLabel::AddrApp => {
                let f = Expr::from(&tree.children()[0]);
                let addr = tree_to_address(&tree.children()[1]);
                ExprNode::AddrApp(Box::new(f), addr)
            }

            // IndexLiteral: (idxL nat nat)
            RiseLabel::IndexLiteral => {
                let i = tree_to_nat(&tree.children()[0]);
                let n = tree_to_nat(&tree.children()[1]);
                ExprNode::IndexLiteral(i, n)
            }

            _ => panic!(
                "unexpected label in expression position: {:?}",
                tree.label()
            ),
        };
        Expr { node, ty }
    }
}

fn tree_to_nat(tree: &TreeNode<RiseLabel>) -> Nat {
    match tree.label() {
        RiseLabel::NatVar(i) => Nat::Var(*i),
        RiseLabel::NatCst(n) => Nat::Cst(*n),
        RiseLabel::NatAdd => Nat::Add(
            Box::new(tree_to_nat(&tree.children()[0])),
            Box::new(tree_to_nat(&tree.children()[1])),
        ),
        RiseLabel::NatMul => Nat::Mul(
            Box::new(tree_to_nat(&tree.children()[0])),
            Box::new(tree_to_nat(&tree.children()[1])),
        ),
        RiseLabel::NatPow => Nat::Pow(
            Box::new(tree_to_nat(&tree.children()[0])),
            Box::new(tree_to_nat(&tree.children()[1])),
        ),
        RiseLabel::NatMod => Nat::Mod(
            Box::new(tree_to_nat(&tree.children()[0])),
            Box::new(tree_to_nat(&tree.children()[1])),
        ),
        RiseLabel::NatFloorDiv => Nat::FloorDiv(
            Box::new(tree_to_nat(&tree.children()[0])),
            Box::new(tree_to_nat(&tree.children()[1])),
        ),
        _ => panic!("unexpected label in nat position: {:?}", tree.label()),
    }
}

fn tree_to_address(tree: &TreeNode<RiseLabel>) -> Address {
    match tree.label() {
        RiseLabel::AddrVar(i) => Address::Var(*i),
        RiseLabel::Global => Address::Global,
        RiseLabel::Local => Address::Local,
        RiseLabel::Private => Address::Private,
        RiseLabel::Constant => Address::Constant,
        _ => panic!("unexpected label in address position: {:?}", tree.label()),
    }
}

fn tree_to_data_type(tree: &TreeNode<RiseLabel>) -> DataType {
    match tree.label() {
        RiseLabel::DataVar(i) => DataType::Var(*i),
        RiseLabel::Scalar(s) => DataType::Scalar(s.clone()),
        RiseLabel::NatT => DataType::NatT,
        RiseLabel::IdxT => DataType::Index(Box::new(tree_to_nat(&tree.children()[0]))),
        RiseLabel::PairT => DataType::Pair(
            Box::new(tree_to_data_type(&tree.children()[0])),
            Box::new(tree_to_data_type(&tree.children()[1])),
        ),
        RiseLabel::ArrT => DataType::Array(
            Box::new(tree_to_nat(&tree.children()[0])),
            Box::new(tree_to_data_type(&tree.children()[1])),
        ),
        RiseLabel::VecT => DataType::Vector(
            Box::new(tree_to_nat(&tree.children()[0])),
            Box::new(tree_to_data_type(&tree.children()[1])),
        ),
        _ => panic!("unexpected label in data type position: {:?}", tree.label()),
    }
}

fn tree_to_type(tree: &TreeNode<RiseLabel>) -> Type {
    match tree.label() {
        RiseLabel::Fun => Type::Fun(
            Box::new(tree_to_type(&tree.children()[0])),
            Box::new(tree_to_type(&tree.children()[1])),
        ),
        RiseLabel::NatFun => Type::NatFun(Box::new(tree_to_type(&tree.children()[0]))),
        RiseLabel::DataFun => Type::DataFun(Box::new(tree_to_type(&tree.children()[0]))),
        RiseLabel::AddrFun => Type::AddrFun(Box::new(tree_to_type(&tree.children()[0]))),
        RiseLabel::NatNatFun => Type::NatNatFun(Box::new(tree_to_type(&tree.children()[0]))),
        // Data type labels in type position -> Type::Data
        _ => Type::Data(tree_to_data_type(tree)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rise_label_implements_label() {
        assert_eq!(RiseLabel::type_of(), RiseLabel::TypeOf);
        assert!(RiseLabel::TypeOf.is_type_of());
        assert!(!RiseLabel::App.is_type_of());
    }

    #[test]
    fn rise_label_from_str() {
        assert_eq!(RiseLabel::from_str("app").unwrap(), RiseLabel::App);
        assert_eq!(RiseLabel::from_str("lam").unwrap(), RiseLabel::Lambda);
        assert_eq!(RiseLabel::from_str("$e0").unwrap(), RiseLabel::Var(0));
        assert_eq!(RiseLabel::from_str("$e123").unwrap(), RiseLabel::Var(123));
        assert_eq!(RiseLabel::from_str("$n5").unwrap(), RiseLabel::NatVar(5));
        assert_eq!(RiseLabel::from_str("32n").unwrap(), RiseLabel::NatCst(32));
        assert_eq!(RiseLabel::from_str("-1n").unwrap(), RiseLabel::NatCst(-1));
        assert_eq!(
            RiseLabel::from_str("0.0f").unwrap(),
            RiseLabel::FloatLit(OrderedFloat(0.0))
        );
        assert_eq!(
            RiseLabel::from_str("map").unwrap(),
            RiseLabel::Primitive(Primitive::Map)
        );
        assert_eq!(
            RiseLabel::from_str("f32").unwrap(),
            RiseLabel::Scalar(ScalarType::F32)
        );
    }

    #[test]
    fn tree_to_expr_roundtrip() {
        let exprs = [
            "(app map (lam $e0))",
            "$e0",
            "(natLam (natLam $e0))",
            "(natApp map 5n)",
            "(idxL $n0 10n)",
            "(typeOf (lam (typeOf $e0 f32)) (fun f32 f32))",
            "(dataApp map (arrT 10n f32))",
            "(addrApp $e0 global)",
        ];
        for input in exprs {
            let expr = input.parse::<Expr>().unwrap();
            let tree = expr.to_tree();
            let recovered = Expr::from(&tree);
            assert_eq!(expr, recovered, "roundtrip failed for {input}");
        }
    }

    #[test]
    fn rise_label_roundtrip() {
        let labels = vec![
            RiseLabel::App,
            RiseLabel::Lambda,
            RiseLabel::Var(0),
            RiseLabel::Var(123),
            RiseLabel::NatVar(5),
            RiseLabel::NatCst(32),
            RiseLabel::NatCst(-1),
            RiseLabel::FloatLit(OrderedFloat(3.11)),
            RiseLabel::Primitive(Primitive::Map),
            RiseLabel::Scalar(ScalarType::F32),
        ];

        for label in labels {
            let s = label.to_string();
            let parsed = RiseLabel::from_str(&s).unwrap();
            assert_eq!(label, parsed, "roundtrip failed for {label:?}");
        }
    }
}
