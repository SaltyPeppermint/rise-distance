use std::cmp::Reverse;
use std::fmt::Display;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{tree::FlattenedTreeNode, tree_distance};

use super::{EditCosts, Label};

/// Structural Distance: More `overlap` is better, otherwise fall back on `zs_sum` as a tiebreaker
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StructuralDistance {
    // More overlap is good, so to make the derives meaningful, we have to reverse them!
    overlap: Reverse<usize>,
    zs_sum: usize,
}

impl Serialize for StructuralDistance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Flat {
            structural_overlap: usize,
            structural_zs_sum: usize,
        }
        Flat {
            structural_overlap: self.overlap.0,
            structural_zs_sum: self.zs_sum,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StructuralDistance {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Flat {
            structural_overlap: usize,
            structural_zs_sum: usize,
        }
        let flat = Flat::deserialize(deserializer)?;
        Ok(StructuralDistance {
            overlap: Reverse(flat.structural_overlap),
            zs_sum: flat.structural_zs_sum,
        })
    }
}

impl Display for StructuralDistance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "o:{}/zs:{}", self.overlap.0, self.zs_sum)
    }
}

impl StructuralDistance {
    fn new(overlap: usize, zs_sum: usize) -> Self {
        Self {
            overlap: Reverse(overlap),
            zs_sum,
        }
    }

    #[must_use]
    pub fn worst() -> Self {
        Self {
            overlap: Reverse(0),
            zs_sum: usize::MAX,
        }
    }

    #[must_use]
    pub fn overlap(&self) -> usize {
        self.overlap.0
    }

    #[must_use]
    pub fn zs_sum(&self) -> usize {
        self.zs_sum
    }
}

impl std::ops::Add for StructuralDistance {
    type Output = StructuralDistance;

    fn add(self, rhs: Self) -> Self::Output {
        StructuralDistance {
            overlap: Reverse(self.overlap.0 + rhs.overlap.0),
            zs_sum: self.zs_sum + rhs.zs_sum,
        }
    }
}

impl std::iter::Sum for StructuralDistance {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), std::ops::Add::add)
    }
}

/// Very simple structural diff.
/// starting from the root, is already present.
pub fn structural_diff<L: Label, C: EditCosts<L>>(
    reference: &FlattenedTreeNode<L>,
    candidate: &FlattenedTreeNode<L>,
    costs: &C,
) -> StructuralDistance {
    fn rec<L: Label, C: EditCosts<L>>(
        reference: &FlattenedTreeNode<L>,
        candidate: &FlattenedTreeNode<L>,
        costs: &C,
    ) -> StructuralDistance {
        if reference.label() != candidate.label() {
            return StructuralDistance::new(0, tree_distance(reference, candidate, costs));
        }
        // This node matched -> count 1 for overlap
        let children_diff: StructuralDistance = reference
            .children()
            .iter()
            .zip(candidate.children())
            .map(|(r, c)| rec(r, c, costs))
            .sum();

        StructuralDistance::new(1, 0) + children_diff
    }
    rec(reference, candidate, costs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnitCost;

    use crate::test_utils::*;
    use crate::tree::TreeShaped;

    fn sd(overlap: usize, zs_sum: usize) -> StructuralDistance {
        StructuralDistance::new(overlap, zs_sum)
    }

    #[test]
    fn identical_trees() {
        let tree1 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        )
        .flatten(true);
        let tree2 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        )
        .flatten(true);
        // All 3 nodes match: overlap = 3, zs_sum = 0
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), sd(3, 0));
    }

    #[test]
    fn different_leaves() {
        let tree1 = leaf("a".to_owned()).flatten(true);
        let tree2 = leaf("b".to_owned()).flatten(true);
        // Labels differ at root -> tree_distance = 1, overlap = 0
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), sd(0, 1));
    }

    #[test]
    fn different_child_count_at_root() {
        let tree1 = node("a".to_owned(), vec![leaf("b".to_owned())]).flatten(true);
        let tree2 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        )
        .flatten(true);
        // Root matches (1), b matches (1), extra c ignored -> overlap = 2
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), sd(2, 0));
    }

    #[test]
    fn different_child_count_nested() {
        // Tree 1:    a
        //            |
        //            b
        //           / \
        //          c   d
        let tree1 = node(
            "a".to_owned(),
            vec![node(
                "b".to_owned(),
                vec![leaf("c".to_owned()), leaf("d".to_owned())],
            )],
        )
        .flatten(true);
        // Tree 2:    a
        //            |
        //            b
        //            |
        //            c
        let tree2 = node(
            "a".to_owned(),
            vec![node("b".to_owned(), vec![leaf("c".to_owned())])],
        )
        .flatten(true);
        // a(1), b(1), c(1) match, d ignored -> overlap = 3
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), sd(3, 0));
    }

    #[test]
    fn same_structure_different_labels() {
        // Same structure but different labels at root -> falls back to tree_distance
        let tree1 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        )
        .flatten(true);
        let tree2 = node(
            "x".to_owned(),
            vec![leaf("y".to_owned()), leaf("z".to_owned())],
        )
        .flatten(true);
        // Root labels differ -> tree_distance = 3 (three relabels), overlap = 0
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), sd(0, 3));
    }

    #[test]
    fn deep_matching_structure() {
        // Both have the same deep structure: a - b - c - d vs w - x - y - z
        // Root labels differ -> falls back to tree_distance = 4
        let tree1 = node(
            "a".to_owned(),
            vec![node(
                "b".to_owned(),
                vec![node("c".to_owned(), vec![leaf("d".to_owned())])],
            )],
        )
        .flatten(true);
        let tree2 = node(
            "w".to_owned(),
            vec![node(
                "x".to_owned(),
                vec![node("y".to_owned(), vec![leaf("z".to_owned())])],
            )],
        )
        .flatten(true);
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), sd(0, 4));
    }

    #[test]
    fn mismatch_at_different_depths() {
        // Tree 1:       a
        //             / | \
        //            b  c  d
        //           /|
        //          e f
        let tree1 = node(
            "a".to_owned(),
            vec![
                node(
                    "b".to_owned(),
                    vec![leaf("e".to_owned()), leaf("f".to_owned())],
                ),
                leaf("c".to_owned()),
                leaf("d".to_owned()),
            ],
        )
        .flatten(true);

        // Tree 2:       a
        //             / | \
        //            b  c  d
        //            |
        //            e
        let tree2 = node(
            "a".to_owned(),
            vec![
                node("b".to_owned(), vec![leaf("e".to_owned())]),
                leaf("c".to_owned()),
                leaf("d".to_owned()),
            ],
        )
        .flatten(true);

        // a(1), b(1), e(1), c(1), d(1) match, f ignored -> overlap = 5
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), sd(5, 0));
    }

    #[test]
    fn leaf_vs_node_with_children() {
        let tree1 = leaf("a".to_owned()).flatten(true);
        let tree2 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        )
        .flatten(true);
        // Root label matches (1), zip of 0 and 2 children = empty -> overlap = 1
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), sd(1, 0));
    }

    #[test]
    fn node_vs_leaf() {
        let tree1 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        )
        .flatten(true);
        let tree2 = leaf("a".to_owned()).flatten(true);
        // Root label matches (1), zip of 2 and 0 children = empty -> overlap = 1
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), sd(1, 0));
    }
}
