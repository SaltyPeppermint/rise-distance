use std::fmt::Display;

use egg::{FromOp, Id, Language, RecExpr};
use serde::{Deserialize, Serialize};

use crate::utils;

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OriginLang<L> {
    inner: L,
    origin: Id,
}

impl<L> OriginLang<L> {
    pub const fn new(inner: L, origin: Id) -> Self {
        Self { inner, origin }
    }

    pub const fn inner(&self) -> &L {
        &self.inner
    }

    pub const fn origin(&self) -> Id {
        self.origin
    }
}

impl<L: Language> Language for OriginLang<L> {
    type Discriminant = (L::Discriminant, Id);

    fn discriminant(&self) -> Self::Discriminant {
        (self.inner.discriminant(), self.origin)
    }

    fn matches(&self, other: &Self) -> bool {
        self.inner.matches(&other.inner)
    }

    fn children(&self) -> &[Id] {
        self.inner.children()
    }

    fn children_mut(&mut self) -> &mut [Id] {
        self.inner.children_mut()
    }
}

impl<L: FromOp> FromOp for OriginLang<L> {
    type Error = L::Error;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        let (inner_op, origin) = op
            .rsplit_once('@')
            .and_then(|(head, tail)| tail.parse::<usize>().ok().map(|id| (head, Id::from(id))))
            .unwrap_or((op, utils::id0()));
        Ok(OriginLang {
            inner: L::from_op(inner_op, children)?,
            origin,
        })
    }
}

impl<L: Display> Display for OriginLang<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.inner, self.origin)
    }
}

#[must_use]
pub fn lower<L: Language>(higher: RecExpr<OriginLang<L>>) -> RecExpr<L> {
    higher.into_iter().map(|n| n.inner).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::langs::math::{self, Math, TinyConstVal};
    use crate::langs::prop::Prop;
    use num::BigInt;
    use num::rational::Ratio;

    #[test]
    fn math_origin_sexpr_roundtrips_with_origins() {
        let mut expr = RecExpr::default();
        let x = expr.add(OriginLang::new(math::sym("x"), Id::from(5)));
        let c = expr.add(OriginLang::new(
            Math::Constant(Ratio::new(BigInt::from(-1), BigInt::from(2))),
            Id::from(11),
        ));
        let mul = expr.add(OriginLang::new(Math::Mul([x, c]), Id::from(147)));
        let tc = expr.add(OriginLang::new(
            Math::TinyConst(TinyConstVal(Ratio::new(BigInt::from(3), BigInt::from(4)))),
            Id::from(42),
        ));
        let _root = expr.add(OriginLang::new(Math::Add([mul, tc]), Id::from(263)));

        let json = serde_json::to_string(&expr).unwrap();
        let s: String = serde_json::from_str(&json).unwrap();
        assert_eq!(s, "(+@263 (*@147 x@5 -1/2@11) tinyconst_3/4@42)");

        let parsed: RecExpr<OriginLang<Math>> = s.parse().unwrap();
        assert_eq!(parsed, expr);
    }

    #[test]
    fn prop_origin_sexpr_roundtrips_with_origins() {
        let mut expr = RecExpr::default();
        let a = expr.add(OriginLang::new(Prop::Symbol("a".into()), Id::from(7)));
        let not_a = expr.add(OriginLang::new(Prop::Not(a), Id::from(13)));
        let t = expr.add(OriginLang::new(Prop::Bool(true), Id::from(9)));
        let _root = expr.add(OriginLang::new(Prop::Implies([not_a, t]), Id::from(21)));

        let s = expr.to_string();
        assert_eq!(s, "(->@21 (~@13 a@7) true@9)");

        let parsed: RecExpr<OriginLang<Prop>> = s.parse().unwrap();
        assert_eq!(parsed, expr);
    }

    #[test]
    fn unannotated_sexpr_defaults_origin_to_zero() {
        let parsed: RecExpr<OriginLang<Math>> = "(+ x 1)".parse().unwrap();
        assert!(parsed.iter().all(|n| n.origin() == utils::id0()));
        assert_eq!(lower(parsed).to_string(), "(+ x 1)");
    }
}
