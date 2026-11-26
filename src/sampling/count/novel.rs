//! Novel-reachable term counts.
//!
//! A current term is novel when no previous e-class can extract it.

use egg::{Analysis, EGraph, Id, Language};
use num::{BigUint, Zero};
use smallvec::SmallVec;

use super::LayerTable;
use super::budgets::{self, MinSizes, RootBudgets};
use super::whole;
use crate::previous::{PrevId, PreviousLookup};
use crate::sampling::{self, INLINE_ARITY};
use crate::utils::{HashMap, HashSet};

/// A current e-node's match in the previous e-graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct NodeMatch {
    pub(crate) prev_class: PrevId,
    pub(crate) prev_children: SmallVec<[PrevId; INLINE_ARITY]>,
}

/// Per `(curr_class, node_idx)`: every match of that e-node in prev.
type NodeMatches = HashMap<(Id, usize), Vec<NodeMatch>>;

/// Shared extraction counts by class pair and size.
type JointTable = HashMap<(Id, PrevId), HashMap<usize, BigUint>>;

/// Joint and novel counts plus previous-node matches.
#[derive(Debug)]
pub(crate) struct NovelTermCount {
    /// Shared extraction counts by class pair and size.
    joint: JointTable,

    /// Previous classes sharing an extraction with each current class.
    cover: HashMap<Id, Vec<PrevId>>,

    /// Per `(curr_class, node_idx)`: every match of that e-node in prev.
    matches: NodeMatches,

    /// Novel extraction counts by current class and size.
    /// `data[c][s] = whole[c][s] - sum_pc joint[(c, pc)][s]`.
    data: HashMap<Id, HashMap<usize, BigUint>>,
}

impl NovelTermCount {
    /// Find the smallest root size `<= cap` that makes at least
    /// `min_extractable` novel terms available, returning `(max_size, counts)`
    /// pruned to `max_size`'s root budgets.
    ///
    /// Budgets are grown geometrically from `min_size(root) + initial_step` up
    /// to `cap`, reusing a single `AstSize` pass so early successes avoid
    /// allocating match tables and DPs for the full `cap`. `make_prev` is
    /// invoked per attempt and dropped before the DPs allocate, so the previous
    /// lookup index never overlaps in memory with the whole/joint DP tables.
    ///
    /// # Errors
    ///
    /// Returns the total novel terms found when `cap` is exhausted below
    /// `min_extractable`.
    pub(crate) fn build<L, N, P, F>(
        curr: &EGraph<L, N>,
        make_prev: F,
        root: Id,
        cap: usize,
        initial_step: usize,
        min_extractable: usize,
    ) -> Result<(usize, Self), BigUint>
    where
        L: Language,
        N: Analysis<L>,
        P: PreviousLookup<L>,
        F: Fn() -> P,
    {
        let root = curr.find(root);
        let min_sizes = MinSizes::new(curr);
        let root_min = min_sizes.get(root);
        let threshold = BigUint::from(min_extractable);
        let mut last_term_count = BigUint::ZERO;

        for try_cap in budgets::doubling_caps(root_min, cap, initial_step) {
            let try_budgets = min_sizes.budgets_for(curr, root, try_cap);
            // TODO: Check whether dropping and rebuilding `prev` on every retry
            // is necessary for peak memory and whether it impacts performance too much.
            let prev = make_prev();
            let mut matches = enumerate_matches(curr, &prev, &try_budgets);
            drop(prev);

            let mut whole = LayerTable::new();
            let mut joint = LayerTable::new();

            let mut term_count = BigUint::ZERO;
            for size in 1..=try_budgets.limit() {
                whole::step(curr, &try_budgets, &mut whole, size);
                let root_non_novel = step_joint(curr, &matches, &try_budgets, &mut joint, size);

                let Some(whole_count) = whole.data.get(&root).and_then(|hist| hist.get(&size))
                else {
                    continue;
                };
                debug_assert!(*whole_count >= root_non_novel);
                let novel = whole_count - root_non_novel;

                if !novel.is_zero() {
                    term_count += novel;
                    if term_count >= threshold {
                        let mut whole_data = whole.into_data();
                        let mut joint_data = joint.into_data();
                        drop(try_budgets);
                        let final_budgets = min_sizes.budgets_for(curr, root, size);
                        prune_matches(curr, &mut matches, &final_budgets);
                        final_budgets.prune_histograms(&mut whole_data, |c| c);
                        final_budgets.prune_histograms(&mut joint_data, |(c, _)| c);
                        drop(final_budgets);
                        drop(min_sizes);
                        return Ok((size, Self::assemble(whole_data, joint_data, matches)));
                    }
                }
            }
            last_term_count = term_count;
        }

        Err(last_term_count)
    }

    fn assemble(
        whole: HashMap<Id, HashMap<usize, BigUint>>,
        joint: JointTable,
        matches: NodeMatches,
    ) -> Self {
        let mut cover: HashMap<Id, Vec<PrevId>> = HashMap::default();
        for &(c, pc) in joint.keys() {
            cover.entry(c).or_default().push(pc);
        }
        let data = derive_novel(whole, &joint);

        Self {
            joint,
            cover,
            matches,
            data,
        }
    }

    /// Shared-term histogram for a canonical current/previous class pair.
    pub(crate) fn joint_histogram(
        &self,
        curr_canon: Id,
        prev_id: PrevId,
    ) -> Option<&HashMap<usize, BigUint>> {
        self.joint.get(&(curr_canon, prev_id))
    }

    /// Novel histogram for a canonical current class.
    pub(crate) fn novel_histogram(&self, curr_canon: Id) -> Option<&HashMap<usize, BigUint>> {
        self.data.get(&curr_canon)
    }

    /// Previous matches of a canonical current class's e-node.
    pub(crate) fn matches_of(&self, curr_canon: Id, node_idx: usize) -> &[NodeMatch] {
        self.matches
            .get(&(curr_canon, node_idx))
            .map_or(&[], Vec::as_slice)
    }

    /// Previous classes sharing an extraction with canonical `curr_canon`.
    pub(crate) fn cover_of(&self, curr_canon: Id) -> &[PrevId] {
        self.cover.get(&curr_canon).map_or(&[], Vec::as_slice)
    }
}

// ============================================================================
// Phase 1: match enumeration.
// ============================================================================

/// Enumerate matches usable from the current root within its size limit.
/// Previous classes are not root-filtered.
fn enumerate_matches<L: Language, N: Analysis<L>, P: PreviousLookup<L>>(
    curr: &EGraph<L, N>,
    prev: &P,
    budgets: &RootBudgets<'_>,
) -> NodeMatches {
    assert!(curr.clean);
    let fitting_nodes = budgets
        .budgets()
        .keys()
        .flat_map(|&c| {
            curr[c]
                .nodes
                .iter()
                .enumerate()
                .filter(move |(_, node)| budgets.node_fits(curr, c, node))
                .map(move |(idx, node)| {
                    debug_assert!(
                        node.children()
                            .iter()
                            .all(|&child| curr.find(child) == child)
                    );
                    (c, idx, node)
                })
        })
        .collect::<Vec<_>>();

    let mut cover: HashMap<Id, HashSet<PrevId>> = HashMap::default();
    let mut matches = NodeMatches::default();

    let mut changed = true;
    while changed {
        changed = false;
        for &(c, idx, node) in &fitting_nodes {
            let Some(slots) = node
                .children()
                .iter()
                .map(|child| cover.get(child).map(|set| set.iter().copied()))
                .collect::<Option<SmallVec<[_; INLINE_ARITY]>>>()
            else {
                continue;
            };
            for combo in sampling::cartesian_product(slots) {
                let mut iter = combo.iter().copied();
                let translated = node.clone().map_children(|_| iter.next().unwrap().raw());
                if let Some(prev_class) = prev.lookup(translated) {
                    let witness = NodeMatch {
                        prev_class,
                        prev_children: combo,
                    };
                    let entry = matches.entry((c, idx)).or_default();
                    if !entry.contains(&witness) {
                        entry.push(witness);
                        cover.entry(c).or_default().insert(prev_class);
                        changed = true;
                    }
                }
            }
        }
    }
    matches
}

/// Tighten cap-scoped matches to a smaller final root budget.
fn prune_matches<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    matches: &mut NodeMatches,
    budgets: &RootBudgets<'_>,
) {
    matches.retain(|&(c, idx), _| {
        egraph[c]
            .nodes
            .get(idx)
            .is_some_and(|node| budgets.node_fits(egraph, c, node))
    });
    matches.shrink_to_fit();
}

// ============================================================================
// Phase 2: joint counts, layered by size.
// ============================================================================

/// Advance `joint` by one size layer directly over `matches`, returning the
/// total non-novel count at `(budgets.root(), size)`.
fn step_joint<L: Language, N: Analysis<L>>(
    curr: &EGraph<L, N>,
    matches: &NodeMatches,
    budgets: &RootBudgets<'_>,
    joint: &mut LayerTable<(Id, PrevId), (Id, usize, usize)>,
    size: usize,
) -> BigUint {
    assert!(curr.clean);
    let total = size - 1;
    // Collect per-match contributions into a flat Vec before publishing so
    // `joint.data` reads finish first. If many matches share one `(c, pc)` at
    // the same size, their counts stay live until merged below (switch back to
    // a per-layer `HashMap` if that transient overhead ever matters).
    let layer = matches
        .iter()
        .filter(|&(&(c, _), _)| budgets.budget(c).is_some_and(|budget| size <= budget))
        .flat_map(|(&(c, idx), ms)| {
            let curr_children = curr[c].nodes[idx].children();
            debug_assert!(curr_children.iter().all(|&child| curr.find(child) == child));
            ms.iter().enumerate().map(move |(match_idx, m)| {
                let children = curr_children
                    .iter()
                    .copied()
                    .zip(m.prev_children.iter().copied())
                    .collect::<SmallVec<[_; INLINE_ARITY]>>();
                (c, idx, match_idx, m.prev_class, children)
            })
        })
        .filter_map(|(c, idx, match_idx, pc, children)| {
            let count = joint.node_count((c, idx, match_idx), &children, total)?;
            Some(((c, pc), count))
        })
        .collect::<Vec<_>>();

    let root = budgets.root();
    let mut root_non_novel = BigUint::ZERO;
    for ((c, pc), count) in layer {
        if c == root {
            root_non_novel += &count;
        }
        joint
            .data
            .entry((c, pc))
            .or_default()
            .entry(size)
            .and_modify(|acc| *acc += &count)
            .or_insert(count);
    }
    root_non_novel
}

// ============================================================================
// Phase 3: derive novel histograms in place.
// ============================================================================

fn derive_novel(
    mut whole: HashMap<Id, HashMap<usize, BigUint>>,
    joint: &JointTable,
) -> HashMap<Id, HashMap<usize, BigUint>> {
    // No double-counting: `prev.lookup(t)` is unique once prev is rebuilt, so
    // each non-novel term contributes to exactly one `(c, pc)` pair and
    // `sum_pc joint[(c, pc)][s] <= whole[c][s]` always.
    for ((c, _pc), hist) in joint {
        if let Some(whole_hist) = whole.get_mut(c) {
            for (size, count) in hist {
                if let Some(total) = whole_hist.get_mut(size) {
                    debug_assert!(*total >= *count);
                    *total -= count;
                }
            }
        }
    }
    whole.retain(|_, hist| {
        hist.retain(|_, count| !count.is_zero());
        !hist.is_empty()
    });
    whole
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::langs::math::{self, Math};

    #[test]
    fn no_novelty_yields_empty() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(math::sym("a"));
        let b = graph.add(math::sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let found = NovelTermCount::build(&graph, || &graph, a, 5, 2, 1)
            .expect_err("expected zero novel terms");

        assert_eq!(found, BigUint::ZERO);
    }

    #[test]
    fn union_makes_alternate_extraction_novel() {
        // Build curr with a, b, ln(a) and clone -> prev (no union yet). Then
        // union a and b in curr so ln(b) becomes a new extraction from the
        // root class.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        let root = curr.add(Math::Ln(a));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.rebuild();

        let (used_size, novel) = NovelTermCount::build(&curr, || &prev, root, 5, 2, 1)
            .expect("expected one novel term at size 2");
        assert_eq!(used_size, 2);

        // The merged a/b class at size 1 has 2 extractions in curr; only "a"
        // is extractable from prev's a-class and only "b" from prev's b-class
        // -> non_novel = 2, novel = 0.
        let ab_class = curr.find(a);
        assert!(
            novel
                .novel_histogram(ab_class)
                .is_none_or(HashMap::is_empty)
        );
        let _ = b;

        // Root: 2 extractions in curr (ln(a), ln(b)). ln(a) is extractable
        // from prev's ln(a); ln(b) is not extractable from any prev class
        // (prev had no ln(b)). So novel = 1.
        let root_canon = curr.find(root);
        assert_eq!(
            novel.novel_histogram(root_canon).unwrap()[&2],
            BigUint::from(1u32)
        );
    }

    #[test]
    fn union_makes_self_term_novel() {
        // prev: Add(a, b)
        // curr: same, but a unioned with b. Now Add(merged, merged) extracts
        // 4 terms: aa, ab, ba, bb. Only ab was in prev (and possibly ba is
        // not since Math::Add is non-commutative). So 3 novel.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        let root = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.rebuild();

        let (used_size, novel) = NovelTermCount::build(&curr, || &prev, root, 5, 2, 1)
            .expect("expected novel terms at size 3");
        assert_eq!(used_size, 3);

        let root_canon = curr.find(root);
        // Whole at size 3 = 4 (aa, ab, ba, bb). Only Add(a, b) was in prev.
        // So novel = 4 - 1 = 3.
        assert_eq!(
            novel.novel_histogram(root_canon).unwrap()[&3],
            BigUint::from(3u32)
        );
    }

    #[test]
    fn rooted_match_fixpoint_builds_parent_from_child_cover() {
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(math::sym("a"));
        let ln = curr.add(Math::Ln(a));
        let root = curr.add(Math::Sqrt(ln));
        curr.rebuild();
        let prev = curr.clone();

        let min_sizes = MinSizes::new(&curr);
        let budgets = min_sizes.budgets_for(&curr, root, 3);
        let matches = enumerate_matches(&curr, &prev, &budgets);
        let root = curr.find(root);
        assert!(matches.keys().any(|(c, _)| *c == root));
    }

    #[test]
    fn rooted_matches_skip_unreachable_classes() {
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        let root = curr.add(Math::Ln(a));
        let unreachable = curr.add(Math::Sqrt(b));
        curr.rebuild();
        let prev = curr.clone();

        let min_sizes = MinSizes::new(&curr);
        let budgets = min_sizes.budgets_for(&curr, root, 2);
        let matches = enumerate_matches(&curr, &prev, &budgets);

        assert!(!matches.keys().any(|(c, _)| *c == curr.find(unreachable)));
        assert!(!matches.keys().any(|(c, _)| *c == curr.find(b)));
    }

    #[test]
    fn final_pruning_removes_oversized_node_matches() {
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(math::sym("a"));
        let ln = curr.add(Math::Ln(a));
        let oversized = curr.add(Math::Sqrt(ln));
        curr.rebuild();
        let prev = curr.clone();
        curr.union(a, oversized);
        curr.rebuild();
        let root = curr.find(a);

        let min_sizes = MinSizes::new(&curr);
        let cap_budgets = min_sizes.budgets_for(&curr, root, 3);
        let mut matches = enumerate_matches(&curr, &prev, &cap_budgets);
        let before = matches.values().map(Vec::len).sum::<usize>();
        let final_budgets = min_sizes.budgets_for(&curr, root, 1);
        prune_matches(&curr, &mut matches, &final_budgets);
        let after = matches.values().map(Vec::len).sum::<usize>();

        assert!(before > after);
        assert!(matches.keys().all(|&(c, idx)| final_budgets.node_fits(
            &curr,
            c,
            &curr[c].nodes[idx]
        )));
    }

    #[test]
    fn rooted_matches_keep_all_previous_witness_classes() {
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        curr.rebuild();
        let prev = curr.clone();
        curr.union(a, b);
        curr.rebuild();
        let root = curr.find(a);

        let min_sizes = MinSizes::new(&curr);
        let budgets = min_sizes.budgets_for(&curr, root, 1);
        let matches = enumerate_matches(&curr, &prev, &budgets);
        let previous_classes = matches
            .iter()
            .filter(|((c, _), _)| *c == root)
            .flat_map(|(_, ms)| ms.iter().map(|m| m.prev_class))
            .collect::<HashSet<_>>();

        assert_eq!(previous_classes.len(), 2);
        assert!(previous_classes.contains(&PreviousLookup::lookup(&prev, math::sym("a")).unwrap()));
        assert!(previous_classes.contains(&PreviousLookup::lookup(&prev, math::sym("b")).unwrap()));
    }

    #[test]
    fn ternary_joint_and_novel_counts_across_layers() {
        use egg::SymbolLang;

        // prev: a == g(a), b, c, root = h(a, b, c). Extracts h(g^k(a), b, c)
        // (1 non-novel term at each size >= 4).
        // curr: union(b, c). Now root extracts 4 terms at each size >= 4,
        // leaving 3 novel terms at size 4 and 3 at size 5 (6 total).
        let mut curr = EGraph::<SymbolLang, ()>::default();
        let a = curr.add(SymbolLang::leaf("a"));
        let ga = curr.add(SymbolLang::new("g", vec![a]));
        curr.union(a, ga);
        let b = curr.add(SymbolLang::leaf("b"));
        let c = curr.add(SymbolLang::leaf("c"));
        let root = curr.add(SymbolLang::new("h", vec![a, b, c]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(b, c);
        curr.rebuild();

        let (used_size, novel) = NovelTermCount::build(&curr, || &prev, root, 8, 2, 6)
            .expect("expected 6 novel terms by size 5");
        assert_eq!(used_size, 5);
        let hist = novel.novel_histogram(curr.find(root)).unwrap();
        assert_eq!(hist[&4], BigUint::from(3u32));
        assert_eq!(hist[&5], BigUint::from(3u32));
    }
}
