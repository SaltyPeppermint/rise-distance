use std::fmt::{self, Display};

use crate::nodes::Label;

#[derive(Debug, Clone, std::hash::Hash, PartialEq, Eq)]
pub struct UnfoldedTree<L: Label> {
    pub(super) label: L,
    pub(super) children: Vec<UnfoldedTree<L>>,
}

impl<L: Label> UnfoldedTree<L> {
    pub fn children(&self) -> &[UnfoldedTree<L>] {
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

impl<L: Label + Display> Display for UnfoldedTree<L> {
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
