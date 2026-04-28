mod flat;
mod origin;
mod partial;
mod typed;

pub use flat::FlatTree;
pub use origin::OriginTree;
pub use partial::{PartialChild, PartialTree, tree_node_to_partial};
pub use typed::TypedTree;

use crate::Label;

pub trait TreeShaped<L: Label>: Sized {
    /// Returns true if this node has no children.
    fn is_leaf(&self) -> bool;

    fn children(&self) -> &[Self];

    fn label(&self) -> &L;

    fn ty(&self) -> Option<&Self>;

    fn flatten(&self, with_types: bool) -> FlatTree<L> {
        if !with_types {
            return FlatTree {
                label: self.label().clone(),
                children: self
                    .children()
                    .iter()
                    .map(|c| c.flatten(with_types))
                    .collect(),
            };
        }
        if let Some(ty) = &self.ty() {
            FlatTree {
                label: L::type_of(),
                children: vec![
                    FlatTree {
                        label: self.label().clone(),
                        children: self
                            .children()
                            .iter()
                            .map(|c| c.flatten(with_types))
                            .collect(),
                    },
                    ty.flatten(with_types),
                ],
            }
        } else {
            FlatTree {
                label: self.label().clone(),
                children: self
                    .children()
                    .iter()
                    .map(|c| c.flatten(with_types))
                    .collect(),
            }
        }
    }

    fn size(&self, with_types: bool) -> usize {
        if with_types {
            self.size_with_types()
        } else {
            self.size_without_types()
        }
    }

    /// Count total number of nodes in this tree.
    fn size_without_types(&self) -> usize {
        1 + self
            .children()
            .iter()
            .map(Self::size_without_types)
            .sum::<usize>()
    }

    fn size_with_types(&self) -> usize {
        1 + self
            .children()
            .iter()
            .map(Self::size_with_types)
            .sum::<usize>()
            + self.ty().map_or(0, |t| t.size_with_types())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sexp_roundtrip_simple() {
        use symbolic_expressions::IntoSexp;

        // Parse a simple s-expression and serialize it back
        let input = "(f a b)";
        let tree = input.parse::<TypedTree<String>>().unwrap();

        assert_eq!(tree.label(), "f");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "a");
        assert_eq!(tree.children()[1].label(), "b");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_nested() {
        use symbolic_expressions::IntoSexp;

        // Nested s-expressions
        let input = "(a (b c) (d e))";
        let tree = input.parse::<TypedTree<String>>().unwrap();

        assert_eq!(tree.label(), "a");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "b");
        assert_eq!(tree.children()[0].children()[0].label(), "c");
        assert_eq!(tree.children()[1].label(), "d");
        assert_eq!(tree.children()[1].children()[0].label(), "e");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_complex_type() {
        use symbolic_expressions::IntoSexp;

        // Expression with a type-like structure
        let input = "(-> int (-> int int))";
        let tree = input.parse::<TypedTree<String>>().unwrap();

        assert_eq!(tree.label(), "->");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "int");
        assert_eq!(tree.children()[1].label(), "->");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_large() {
        use symbolic_expressions::IntoSexp;

        let input = "(natLam (natLam (natLam (lam (lam (app (app map (lam (app (app map (lam (app (app (app reduce add) 0.0) (app (app map (lam (app (app mul (app fst $e0)) (app snd $e0)))) (app (app zip $e1) $e0))))) (app transpose $e1)))) $e1))))))";
        let tree = input.parse::<TypedTree<String>>().unwrap();

        assert_eq!(tree.label(), "natLam");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_leaf() {
        use symbolic_expressions::IntoSexp;

        let input = "x";
        let tree = input.parse::<TypedTree<String>>().unwrap();

        assert_eq!(tree.label(), "x");
        assert!(tree.is_leaf());

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }
}
