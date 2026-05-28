use std::fmt::Display;

use egg::{FromOp, Id, Language, RecExpr};
use serde::{Deserialize, Serialize};

use crate::MyLanguage;
use crate::egg::id0;

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound = "L: MyLanguage")]
pub struct OriginLang<L: MyLanguage> {
    inner: L,
    origin: Id,
}

impl<L: MyLanguage> OriginLang<L> {
    pub fn new(inner: L, origin: Id) -> Self {
        Self { inner, origin }
    }

    pub fn inner(&self) -> &L {
        &self.inner
    }

    pub fn origin(&self) -> Id {
        self.origin
    }
}

impl<L: MyLanguage> Language for OriginLang<L> {
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

impl<L: MyLanguage> FromOp for OriginLang<L> {
    type Error = L::Error;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        Ok(OriginLang {
            inner: L::from_op(op, children)?,
            origin: id0(),
        })
    }
}

impl<L: MyLanguage> Display for OriginLang<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

#[must_use]
pub fn lower<L: MyLanguage>(higher: RecExpr<OriginLang<L>>) -> RecExpr<L> {
    higher.into_iter().map(|n| n.inner).collect()
}
