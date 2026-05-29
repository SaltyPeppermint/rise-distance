use std::fmt::{self, Display};
use std::sync::atomic::{AtomicUsize, Ordering};

use egg::{Id, RecExpr};
use rayon::prelude::*;

use crate::{MyLanguage, OriginLang, id0};

/// Core Zhang-Shasha minimum distance search over a parallel iterator of candidate trees.
///
/// Applies size-difference and Euler-string lower-bound pruning before computing
/// the full edit distance.
pub fn find_min_zs<L, CF, I>(
    candidates: I,
    reference: &RecExpr<L>,
    costs: &CF,
) -> (Option<(RecExpr<L>, usize)>, ZSStats)
where
    L: MyLanguage,
    CF: EditCosts<L>,
    I: ParallelIterator<Item = RecExpr<L>>,
{
    let ref_flat: FlatTree<L> = reference.into();

    let ref_size = ref_flat.size();
    let ref_pp = PreprocessedTree::new(&ref_flat);
    let running_best = AtomicUsize::new(usize::MAX);

    candidates
        .map(|candidate| {
            let candidate_flat: FlatTree<L> = (&candidate).into();
            let best = running_best.load(Ordering::Relaxed);

            if candidate_flat.size().abs_diff(ref_size) > best {
                return (None, ZSStats::size_pruned());
            }

            let distance = tree_distance_with_ref(&candidate_flat, &ref_pp, costs);
            running_best.fetch_min(distance, Ordering::Relaxed);

            (Some((candidate, distance)), ZSStats::compared())
        })
        .reduce(
            || (None, ZSStats::default()),
            |a, b| {
                let best = [a.0, b.0].into_iter().flatten().min_by_key(|v| v.1);
                (best, a.1 + b.1)
            },
        )
}

/// Statistics from filtered extraction
#[derive(Debug, Clone, Default)]
pub struct ZSStats {
    /// Total number of trees enumerated
    pub trees_enumerated: usize,
    /// Trees pruned by simple metric
    pub size_pruned: usize,

    /// Number of trees for which full distance was computed
    pub full_comparisons: usize,
}

impl ZSStats {
    pub(crate) fn size_pruned() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 1,
            full_comparisons: 0,
        }
    }

    pub(crate) fn compared() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 0,
            full_comparisons: 1,
        }
    }
}

impl std::ops::Add for ZSStats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            trees_enumerated: self.trees_enumerated + rhs.trees_enumerated,
            size_pruned: self.size_pruned + rhs.size_pruned,
            full_comparisons: self.full_comparisons + rhs.full_comparisons,
        }
    }
}

/// Postorder traversal information for a tree node.
#[derive(Debug, Clone)]
struct PostorderNode<'a, L: MyLanguage> {
    label: &'a L,
    leftmost_leaf: usize,
}

/// Preprocessed tree for Zhang-Shasha algorithm.
///
/// Reuse this when computing distances against multiple candidate trees.
pub struct PreprocessedTree<'a, L: MyLanguage> {
    nodes: Vec<PostorderNode<'a, L>>,
    keyroots: Vec<usize>,
}

impl<'a, L: MyLanguage> PreprocessedTree<'a, L> {
    /// Create a preprocessed tree from a tree node.
    /// This performs a single postorder traversal to compute leftmost leaf descendants
    /// and keyroots.
    pub fn new(root: &'a FlatTree<L>) -> Self {
        let mut nodes = Vec::new();

        // Perform postorder traversal and compute leftmost leaf descendants
        Self::postorder_traverse(root, &mut nodes);

        // Compute keyroots: a node is a keyroot if it's the last node (in postorder)
        // with its particular leftmost leaf value. This is equivalent to: a node is
        // a keyroot if it has no parent, or it is not the leftmost child of its parent.
        let mut keyroots = Vec::new();
        let mut leftmost_to_keyroot = vec![0; nodes.len()];

        for (i, n) in nodes.iter().enumerate() {
            // Each time we see a leftmost leaf value, update to the latest node
            leftmost_to_keyroot[n.leftmost_leaf] = i;
        }

        // Collect unique keyroots
        let mut seen = vec![false; nodes.len()];
        for &kr in &leftmost_to_keyroot {
            if !seen[kr] {
                seen[kr] = true;
                keyroots.push(kr);
            }
        }

        keyroots.sort_unstable();

        PreprocessedTree { nodes, keyroots }
    }

    fn postorder_traverse(node: &'a FlatTree<L>, nodes: &mut Vec<PostorderNode<'a, L>>) -> usize {
        // First, traverse all children
        let child_indices = node
            .children()
            .iter()
            .map(|child| Self::postorder_traverse(child, nodes))
            .collect::<Vec<_>>();

        // Current node's postorder index
        let current_idx = nodes.len();

        // Compute leftmost leaf
        let leftmost_leaf = if node.is_leaf() {
            current_idx
        } else {
            // Leftmost leaf is the leftmost leaf of the leftmost child
            nodes[child_indices[0]].leftmost_leaf
        };

        nodes.push(PostorderNode {
            label: node.label(),
            leftmost_leaf,
        });

        current_idx
    }

    /// Returns the number of nodes in the tree
    fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the postorder index of the leftmost leaf descendant of node i
    fn leftmost_leaf(&self, i: usize) -> usize {
        self.nodes[i].leftmost_leaf
    }

    /// Returns the label of node i
    fn label(&self, i: usize) -> &L {
        self.nodes[i].label
    }

    /// Returns the keyroots of the tree (nodes that start new subproblems)
    fn keyroots(&self) -> &[usize] {
        &self.keyroots
    }
}

/// Cost functions for tree edit operations.
pub trait EditCosts<L>: Send + Sync {
    /// Cost of deleting a node.
    fn delete(&self, label: &L) -> usize;

    /// Cost of inserting a node.
    fn insert(&self, label: &L) -> usize;

    /// Cost of relabeling a node.
    fn relabel(&self, from: &L, to: &L) -> usize;

    /// Euler in-out-same-node
    fn euler_in_out(&self, label: &L) -> usize;
}

/// Unit cost model: all operations cost 1, relabeling identical labels costs 0.
pub struct UnitCost;

impl<L: Eq> EditCosts<L> for UnitCost {
    fn delete(&self, _label: &L) -> usize {
        1
    }

    fn insert(&self, _label: &L) -> usize {
        1
    }

    fn relabel(&self, from: &L, to: &L) -> usize {
        usize::from(from != to)
    }

    fn euler_in_out(&self, _label: &L) -> usize {
        1
    }
}

/// Compute the Zhang-Shasha tree edit distance between two trees.
pub fn tree_distance<L: MyLanguage, C: EditCosts<L>>(
    tree1: &FlatTree<L>,
    tree2: &FlatTree<L>,
    costs: &C,
) -> usize {
    let t1 = PreprocessedTree::new(tree1);
    let t2 = PreprocessedTree::new(tree2);
    tree_distance_preprocessed(&t1, &t2, costs)
}

/// Compute distance with a pre-preprocessed reference tree.
pub fn tree_distance_with_ref<L: MyLanguage, C: EditCosts<L>>(
    candidate: &FlatTree<L>,
    reference: &PreprocessedTree<L>,
    costs: &C,
) -> usize {
    let t1 = PreprocessedTree::new(candidate);
    tree_distance_preprocessed(&t1, reference, costs)
}

/// Compute distance between two preprocessed trees.
pub fn tree_distance_preprocessed<L: MyLanguage, C: EditCosts<L>>(
    t1: &PreprocessedTree<L>,
    t2: &PreprocessedTree<L>,
    costs: &C,
) -> usize {
    let n1 = t1.size();
    let n2 = t2.size();

    if n1 == 0 && n2 == 0 {
        return 0;
    }
    if n1 == 0 {
        return (0..n2).map(|j| costs.insert(t2.label(j))).sum();
    }
    if n2 == 0 {
        return (0..n1).map(|i| costs.delete(t1.label(i))).sum();
    }

    // Tree distance matrix (permanent)
    let mut td = vec![vec![0; n2]; n1];

    // Forest distance matrix (temporary, reused for each keyroot pair)
    // We need indices from -1, so we use size+1 and offset by 1
    let mut fd = vec![vec![0; n2 + 1]; n1 + 1];

    // Compute tree distance for each pair of keyroots
    for &i in t1.keyroots() {
        for &j in t2.keyroots() {
            compute_forest_distance(t1, t2, i, j, &mut td, &mut fd, costs);
        }
    }

    // The final answer is the distance between the full trees
    td[n1 - 1][n2 - 1]
}

fn compute_forest_distance<L: MyLanguage, C: EditCosts<L>>(
    t1: &PreprocessedTree<L>,
    t2: &PreprocessedTree<L>,
    i: usize,
    j: usize,
    td: &mut [Vec<usize>],
    fd: &mut [Vec<usize>],
    costs: &C,
) {
    let l1 = t1.leftmost_leaf(i);
    let l2 = t2.leftmost_leaf(j);

    // fd[x][y] represents the forest distance between:
    // - forest of t1 from l1 to x-1 (using 1-based indexing offset)
    // - forest of t2 from l2 to y-1
    // fd[0][0] = 0 (empty forests)

    // Initialize: deleting all nodes from t1's forest
    fd[0][0] = 0;
    for x in l1..=i {
        let x_idx = x - l1 + 1;
        fd[x_idx][0] = fd[x_idx - 1][0] + costs.delete(t1.label(x));
    }

    // Initialize: inserting all nodes into empty forest from t2
    for y in l2..=j {
        let y_idx = y - l2 + 1;
        fd[0][y_idx] = fd[0][y_idx - 1] + costs.insert(t2.label(y));
    }

    // Fill in the forest distance matrix
    // Note: we intentionally use x and y as indices into td, as td stores
    // tree distances for all node pairs using their postorder indices
    #[expect(clippy::needless_range_loop)]
    for x in l1..=i {
        let x_idx = x - l1 + 1;
        let lx = t1.leftmost_leaf(x);

        for y in l2..=j {
            let y_idx = y - l2 + 1;
            let ly = t2.leftmost_leaf(y);

            let delete_cost = fd[x_idx - 1][y_idx] + costs.delete(t1.label(x));
            let insert_cost = fd[x_idx][y_idx - 1] + costs.insert(t2.label(y));

            if lx == l1 && ly == l2 {
                // Both x and y have their leftmost leaves at the start of this subproblem,
                // meaning we're computing the full subtree distance for (x, y)
                let relabel_cost =
                    fd[x_idx - 1][y_idx - 1] + costs.relabel(t1.label(x), t2.label(y));
                fd[x_idx][y_idx] = delete_cost.min(insert_cost).min(relabel_cost);
                // Store in permanent tree distance matrix
                td[x][y] = fd[x_idx][y_idx];
            } else {
                // At least one of x or y has its leftmost leaf before the start of this
                // subproblem, so we need to use the previously computed tree distance
                let match_cost = fd[lx - l1][ly - l2] + td[x][y];
                fd[x_idx][y_idx] = delete_cost.min(insert_cost).min(match_cost);
            }
        }
    }
}

/// Compute tree edit distance with unit costs.
pub fn tree_distance_unit<L: MyLanguage>(tree1: &FlatTree<L>, tree2: &FlatTree<L>) -> usize {
    tree_distance(tree1, tree2, &UnitCost)
}

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
