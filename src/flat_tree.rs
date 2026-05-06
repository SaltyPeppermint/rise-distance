use std::fmt::{self, Display};

use egg::{Id, RecExpr};

use crate::egg::id0;
use crate::{MyLanguage, OriginLang};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FlatTree<L: MyLanguage> {
    pub(super) label: L,
    pub(super) children: Vec<FlatTree<L>>,
}

impl<L: MyLanguage> FlatTree<L> {
    pub fn children(&self) -> &[FlatTree<L>] {
        &self.children
    }

    pub fn label(&self) -> &L {
        &self.label
    }

    /// Returns true if this node has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn size(&self) -> usize {
        1 + self.children.iter().map(Self::size).sum::<usize>()
    }
}

impl<L: MyLanguage> From<&RecExpr<L>> for FlatTree<L> {
    fn from(value: &RecExpr<L>) -> Self {
        fn rec<LL: MyLanguage>(expr: &RecExpr<LL>, id: Id) -> FlatTree<LL> {
            let children = expr[id]
                .children()
                .iter()
                .map(|c_id| rec(expr, *c_id))
                .collect();
            let label = expr[id].clone().map_children(|_| id0());
            FlatTree { label, children }
        }
        rec(value, value.root())
    }
}

impl<L: MyLanguage> From<&RecExpr<OriginLang<L>>> for FlatTree<L> {
    fn from(value: &RecExpr<OriginLang<L>>) -> Self {
        fn rec<LL: MyLanguage>(expr: &RecExpr<OriginLang<LL>>, id: Id) -> FlatTree<LL> {
            let children = expr[id]
                .inner()
                .children()
                .iter()
                .map(|c_id| rec(expr, *c_id))
                .collect();
            let label = expr[id].inner().clone().map_children(|_| id0());
            FlatTree { label, children }
        }
        rec(value, value.root())
    }
}

impl<L: MyLanguage + Display> Display for FlatTree<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_leaf() {
            write!(f, "{}", self.label)
        } else {
            write!(f, "({}", self.label)?;
            for child in &self.children {
                write!(f, " {child}")?;
            }
            write!(f, ")")
        }
    }
}
