//! Novel-reachable term counts.
//!
//! A current term is novel when no previous e-class can extract it. This
//! module enumerates previous matches, counts shared terms, and subtracts them
//! from plain counts. See `docs/candidates/novel_candidates.md`.

use egg::{Analysis, EGraph, Id, Language};
use hashbrown::{HashMap, HashSet};
use num::BigUint;
use smallvec::SmallVec;

use crate::Counter;
use crate::candidates::count::budgets::RootBudgets;
#[cfg(test)]
use crate::candidates::count::count_histograms_rooted;
use crate::candidates::count::{LayeredDp, plain_dp_rooted};
use crate::previous::PreviousLookup;

/// Child class ids, inline for the usual arity of at most two.
pub type ChildIds = SmallVec<[Id; 2]>;

/// A current e-node's match in the previous e-graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeMatch {
    pub prev_class: Id,
    pub prev_children: ChildIds,
}

/// Per `(curr_class, node_idx)`: every match of that e-node in prev.
pub type NodeMatches = HashMap<(Id, usize), Vec<NodeMatch>>;

/// Previous classes sharing an extraction with each current class.
type MatchCover = HashMap<Id, HashSet<Id>>;

/// Dedup key for match enumeration: `(curr_class, node_idx, prev_class,
/// prev_children)`.
type MatchKeys = HashSet<(Id, usize, Id, ChildIds)>;

/// Shared extraction counts by class pair and size.
type JointTable<C> = HashMap<(Id, Id), HashMap<usize, C>>;

/// Joint and novel counts plus previous-node matches.
///
/// The plain counts are an input to [`derive_novel`] only, never a field: the
/// frontier drawer reads `data` and `joint` and re-derives child-size splits
/// per draw, so retaining the plain histograms and their suffix tables would
/// cost more than everything kept here put together.
#[derive(Debug)]
pub struct NovelTermCount<C>
where
    C: Counter,
{
    /// Shared extraction counts by class pair and size.
    joint: JointTable<C>,

    /// Previous classes sharing an extraction with each current class.
    cover: HashMap<Id, Vec<Id>>,

    /// Per `(curr_class, node_idx)`: every match of that e-node in prev.
    matches: NodeMatches,

    /// Novel extraction counts by current class and size.
    /// `data[c][s] = plain[c][s] - sum_pc joint[(c, pc)][s]`.
    data: HashMap<Id, HashMap<usize, C>>,
}

impl<C: Counter> NovelTermCount<C> {
    /// Test convenience around the production rooted pipeline.
    #[cfg(test)]
    pub(crate) fn rooted_for_tests<L: Language, N: Analysis<L>, P: PreviousLookup<L>>(
        max_size: usize,
        curr: &EGraph<L, N>,
        prev: &P,
        root: Id,
    ) -> Self {
        let budgets = crate::candidates::count::budgets::root_budgets(curr, root, max_size);
        let matches = enumerate_matches_rooted(curr, prev, &budgets);
        let plain = count_histograms_rooted(curr, &budgets);
        Self::from_rooted_matches(curr, &plain, matches, &budgets)
    }

    /// Run root-restricted joint counting using the same current-class
    /// budgets that produced `plain` and `matches`.
    ///
    /// `plain` is consumed by the novel subtraction and then dropped.
    #[must_use]
    pub(crate) fn from_rooted_matches<L: Language, N: Analysis<L>>(
        curr: &EGraph<L, N>,
        plain: &HashMap<Id, HashMap<usize, C>>,
        matches: NodeMatches,
        budgets: &RootBudgets,
    ) -> Self {
        let joint = compute_joint_rooted(curr, &matches, budgets);
        let cover = build_cover(&joint);
        let data = derive_novel(plain, &joint);

        Self {
            joint,
            cover,
            matches,
            data,
        }
    }

    /// Novel histograms keyed by canonical current ids.
    #[must_use]
    pub const fn data(&self) -> &HashMap<Id, HashMap<usize, C>> {
        &self.data
    }

    /// Shared-term histogram for a current/previous class pair.
    pub(crate) fn joint_histogram<L: Language, N: Analysis<L>>(
        &self,
        curr: &EGraph<L, N>,
        curr_id: Id,
        prev_id: Id,
    ) -> Option<&HashMap<usize, C>> {
        let curr_canon = curr.find(curr_id);
        self.joint.get(&(curr_canon, prev_id))
    }

    /// Novel histogram for a current class.
    pub(crate) fn novel_histogram<L: Language, N: Analysis<L>>(
        &self,
        curr: &EGraph<L, N>,
        curr_id: Id,
    ) -> Option<&HashMap<usize, C>> {
        let canon = curr.find(curr_id);
        self.data.get(&canon)
    }

    pub(crate) fn matches_of<L: Language, N: Analysis<L>>(
        &self,
        curr: &EGraph<L, N>,
        curr_class: Id,
        node_idx: usize,
    ) -> &[NodeMatch] {
        let canon = curr.find(curr_class);
        self.matches
            .get(&(canon, node_idx))
            .map_or(&[][..], Vec::as_slice)
    }

    /// Previous classes sharing an extraction with `curr_id`.
    pub(crate) fn cover_of<L: Language, N: Analysis<L>>(
        &self,
        curr: &EGraph<L, N>,
        curr_id: Id,
    ) -> &[Id] {
        let canon = curr.find(curr_id);
        self.cover.get(&canon).map_or(&[][..], Vec::as_slice)
    }
}

// The exposed `cover` is built from the *joint* keys, not from match
// enumeration's internal `cover`. The two can differ: a `(c, pc)` pair whose
// matches all involve some child with empty `joint` within its root budget
// ends up dropped by rooted joint counting. That's fine for exact candidate construction: a
// missing `pc` had joint count 0 anyway, so neither slot enumeration nor
// `completes_some_match`
// can be fooled by its absence.
fn build_cover<C: Counter>(joint: &JointTable<C>) -> HashMap<Id, Vec<Id>> {
    let mut out: HashMap<Id, Vec<Id>> = HashMap::new();
    for (c, pc) in joint.keys() {
        let entry = out.entry(*c).or_default();
        if !entry.contains(pc) {
            entry.push(*pc);
        }
    }
    out
}

// ============================================================================
// Phase 1: match enumeration.
// ============================================================================

/// Enumerate matches usable from the current root within its size limit.
/// Previous classes are not root-filtered.
pub(crate) fn enumerate_matches_rooted<L: Language, N: Analysis<L>, P: PreviousLookup<L>>(
    curr: &EGraph<L, N>,
    prev: &P,
    budgets: &RootBudgets,
) -> NodeMatches {
    let mut cover = MatchCover::new();
    let mut matches = NodeMatches::new();
    let mut seen = MatchKeys::new();

    loop {
        let before = seen.len();
        for &c in budgets.budgets().keys() {
            for (idx, node) in curr[c].nodes.iter().enumerate() {
                if !budgets.node_fits(curr, c, node) {
                    continue;
                }
                let child_canons = node
                    .children()
                    .iter()
                    .map(|&child| curr.find(child))
                    .collect::<ChildIds>();
                let combos = child_combinations(&child_canons, &cover);
                for combo in combos {
                    let mut translated = node.clone();
                    let mut iter = combo.iter().copied();
                    translated.for_each_mut(|child| {
                        if let Some(pc) = iter.next() {
                            *child = pc;
                        }
                    });
                    if let Some(pc_class) = prev.lookup(translated)
                        && seen.insert((c, idx, pc_class, combo.clone()))
                    {
                        matches.entry((c, idx)).or_default().push(NodeMatch {
                            prev_class: pc_class,
                            prev_children: combo,
                        });
                        cover.entry(c).or_default().insert(pc_class);
                    }
                }
            }
        }

        let discovered = seen.len() - before;
        if discovered == 0 {
            break;
        }
    }
    matches
}

/// Tighten cap-scoped matches to a smaller final root budget.
pub(crate) fn prune_matches<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    matches: &mut NodeMatches,
    budgets: &RootBudgets,
) {
    matches.retain(|&(c, idx), _| {
        budgets.budget(c).is_some()
            && egraph[c]
                .nodes
                .get(idx)
                .is_some_and(|node| budgets.node_fits(egraph, c, node))
    });
}

/// Cartesian product of `cover[child_i]` over `i`. For zero-arity nodes,
/// returns `[[]]` (a single empty combination).
fn child_combinations(children: &[Id], cover: &MatchCover) -> Vec<ChildIds> {
    let mut combos = vec![ChildIds::new()];
    for child in children {
        let Some(opts) = cover.get(child) else {
            return Vec::new();
        };
        if opts.is_empty() {
            return Vec::new();
        }
        combos = combos
            .iter()
            .flat_map(|prefix| {
                opts.iter().map(|opt| {
                    let mut p = prefix.clone();
                    p.push(*opt);
                    p
                })
            })
            .collect();
    }
    combos
}

// ============================================================================
// Phase 2: joint counts, layered by size.
// ============================================================================

/// Compute rooted `joint[(c, pc)]` counts with each pair capped by the budget
/// of its current class.
fn compute_joint_rooted<C: Counter, L: Language, N: Analysis<L>>(
    curr: &EGraph<L, N>,
    matches: &NodeMatches,
    rooted: &RootBudgets,
) -> JointTable<C> {
    let children_of = joint_children_of(curr, matches);
    let budgets = children_of
        .keys()
        .filter_map(|&(c, pc)| rooted.budget(c).map(|budget| ((c, pc), budget)))
        .collect();
    let mut dp = LayeredDp::new(children_of, budgets);
    for _ in 0..rooted.limit() {
        dp.step();
    }
    dp.into_parts().0
}

/// Joint-DP nodes grouped by current/previous class pair.
type PairChildren = HashMap<(Id, Id), Vec<Vec<(Id, Id)>>>;

fn joint_children_of<L: Language, N: Analysis<L>>(
    curr: &EGraph<L, N>,
    matches: &NodeMatches,
) -> PairChildren {
    let mut out = PairChildren::new();
    for ((c, idx), ms) in matches {
        let node = &curr[*c].nodes[*idx];
        for m in ms {
            let child_pairs = node
                .children()
                .iter()
                .zip(m.prev_children.iter())
                .map(|(child, prev_child)| (curr.find(*child), *prev_child))
                .collect();
            out.entry((*c, m.prev_class)).or_default().push(child_pairs);
        }
    }
    out
}

// ============================================================================
// Exact root-size scan.
// ============================================================================

/// Find the first novel root sizes that ensures enough are extractable within `rooted`.
pub(crate) fn find_novel_root_sizes<L: Language, N: Analysis<L>>(
    curr: &EGraph<L, N>,
    root: Id,
    matches: &NodeMatches,
    min_extractable: usize,
    rooted: &RootBudgets,
) -> Result<usize, BigUint> {
    let root = curr.find(root);
    let mut plain: LayeredDp<Id, BigUint> = plain_dp_rooted(curr, rooted);

    // Each pair inherits its curr class's rooted budget: joint terms are
    // plain terms of that class, and the budget recurrence relaxes with
    // plain minima, which lower-bound joint subterm sizes too — so every
    // cell a root query depends on stays within budget. Pairs of classes
    // unreachable within `max_size` can never be depended on and are
    // skipped entirely.
    let children_of = joint_children_of(curr, matches);
    let budgets: HashMap<(Id, Id), usize> = children_of
        .keys()
        .filter_map(|&(c, pc)| plain.budgets().get(&c).map(|&b| ((c, pc), b)))
        .collect();
    let root_pairs = budgets
        .keys()
        .copied()
        .filter(|&(c, _)| c == root)
        .collect::<Vec<_>>();
    let mut joint: LayeredDp<(Id, Id), BigUint> = LayeredDp::new(children_of, budgets);

    let mut max_size;
    let mut term_count = BigUint::ZERO;
    for _ in 0..rooted.limit() {
        let size = plain.step();
        eprintln!(
            "DEBUG: PEAK RSS IN NOVEL ROOT DP AFTER STEP {size}: {}",
            crate::utils::peak_rss_bytes()
        );
        joint.step();

        // Final as of this layer. Zero-count entries are absent and read as 0.
        let mut novel = plain
            .data()
            .get(&root)
            .and_then(|hist| hist.get(&size))
            .cloned()
            .unwrap_or(BigUint::ZERO);
        for pair in &root_pairs {
            if let Some(count) = joint.data().get(pair).and_then(|hist| hist.get(&size)) {
                novel -= count;
            }
        }

        if novel != BigUint::ZERO {
            max_size = size;
            term_count += novel;
            if term_count >= min_extractable.into() {
                return Ok(max_size);
            }
        }
    }

    Err(term_count)
}

// ============================================================================
// Phase 3: derive novel histograms.
// ============================================================================

fn derive_novel<C: Counter>(
    plain: &HashMap<Id, HashMap<usize, C>>,
    joint: &JointTable<C>,
) -> HashMap<Id, HashMap<usize, C>> {
    // Aggregate sum_pc joint[(c, pc)] per curr class. No double-counting:
    // `prev.lookup(t)` is unique once prev is rebuilt, so each non-novel term
    // contributes to exactly one `(c, pc)` pair. Hence
    // `non_novel[c][s] <= plain[c][s]` always (every non-novel term is also
    // a plain term).
    let mut non_novel: HashMap<Id, HashMap<usize, C>> = HashMap::new();
    for ((c, _pc), hist) in joint {
        let entry = non_novel.entry(*c).or_default();
        for (size, count) in hist {
            entry
                .entry(*size)
                .and_modify(|x| *x += count)
                .or_insert_with(|| count.clone());
        }
    }

    let mut out = HashMap::with_capacity(plain.len());
    for (c, plain_hist) in plain {
        let nn = non_novel.get(c);
        let mut hist = HashMap::with_capacity(plain_hist.len());
        for (&size, total) in plain_hist {
            let novel = match nn.and_then(|h| h.get(&size)) {
                Some(non_novel_count) => {
                    debug_assert!(non_novel_count <= total);
                    let mut t = total.clone();
                    t -= non_novel_count;
                    t
                }
                None => total.clone(),
            };
            if novel != C::zero() {
                hist.insert(size, novel);
            }
        }
        if !hist.is_empty() {
            out.insert(*c, hist);
        }
    }
    out
}

// ============================================================================
// Tests.
// ============================================================================

#[cfg(test)]
mod tests {
    use egg::EGraph;
    use num::BigUint;

    use super::*;
    use crate::candidates::count::budgets::root_budgets;
    use crate::langs::math::Math;
    use crate::utils::sym;

    fn rooted_novel(
        curr: &EGraph<Math, ()>,
        prev: &EGraph<Math, ()>,
        root: Id,
        max_size: usize,
    ) -> NovelTermCount<BigUint> {
        let budgets = root_budgets(curr, root, max_size);
        let matches = enumerate_matches_rooted(curr, prev, &budgets);
        let plain = count_histograms_rooted(curr, &budgets);
        NovelTermCount::from_rooted_matches(curr, &plain, matches, &budgets)
    }

    #[test]
    fn no_novelty_yields_empty() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let novel = rooted_novel(&graph, &graph, a, 5);

        assert!(novel.data().is_empty(), "expected empty novel data");
    }

    #[test]
    fn union_makes_alternate_extraction_novel() {
        // Build curr with a, b, ln(a) and clone -> prev (no union yet). Then
        // union a and b in curr so ln(b) becomes a new extraction from the
        // root class.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let root = curr.add(Math::Ln(a));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.rebuild();

        let novel = rooted_novel(&curr, &prev, root, 5);

        // The merged a/b class at size 1 has 2 extractions in curr; only "a"
        // is extractable from prev's a-class and only "b" from prev's b-class
        // -> non_novel = 2, novel = 0.
        let ab_class = curr.find(a);
        assert!(novel.data().get(&ab_class).is_none_or(HashMap::is_empty));
        let _ = b;

        // Root: 2 extractions in curr (ln(a), ln(b)). ln(a) is extractable
        // from prev's ln(a); ln(b) is not extractable from any prev class
        // (prev had no ln(b)). So novel = 1.
        let root_canon = curr.find(root);
        assert_eq!(novel.data()[&root_canon][&2], BigUint::from(1u32));
    }

    #[test]
    fn union_makes_self_term_novel() {
        // prev: Add(a, b)
        // curr: same, but a unioned with b. Now Add(merged, merged) extracts
        // 4 terms: aa, ab, ba, bb. Only ab was in prev (and possibly ba is
        // not since Math::Add is non-commutative). So 3 novel.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let root = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.rebuild();

        let novel = rooted_novel(&curr, &prev, root, 5);

        let root_canon = curr.find(root);
        // Plain at size 3 = 4 (aa, ab, ba, bb). Only Add(a, b) was in prev.
        // So novel = 4 - 1 = 3.
        assert_eq!(novel.data()[&root_canon][&3], BigUint::from(3u32));
    }

    #[test]
    fn rooted_match_fixpoint_builds_parent_from_child_cover() {
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let ln = curr.add(Math::Ln(a));
        let root = curr.add(Math::Sqrt(ln));
        curr.rebuild();
        let prev = curr.clone();

        let budgets = root_budgets(&curr, root, 3);
        let matches = enumerate_matches_rooted(&curr, &prev, &budgets);
        let root = curr.find(root);
        assert!(matches.keys().any(|(c, _)| *c == root));
    }

    #[test]
    fn rooted_matches_skip_unreachable_classes() {
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let root = curr.add(Math::Ln(a));
        let unreachable = curr.add(Math::Sqrt(b));
        curr.rebuild();
        let prev = curr.clone();

        let budgets = root_budgets(&curr, root, 2);
        let matches = enumerate_matches_rooted(&curr, &prev, &budgets);

        assert!(!matches.keys().any(|(c, _)| *c == curr.find(unreachable)));
        assert!(!matches.keys().any(|(c, _)| *c == curr.find(b)));
    }

    #[test]
    fn final_pruning_removes_oversized_node_matches() {
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let ln = curr.add(Math::Ln(a));
        let oversized = curr.add(Math::Sqrt(ln));
        curr.rebuild();
        let prev = curr.clone();
        curr.union(a, oversized);
        curr.rebuild();
        let root = curr.find(a);

        let cap_budgets = root_budgets(&curr, root, 3);
        let mut matches = enumerate_matches_rooted(&curr, &prev, &cap_budgets);
        let before = matches.values().map(Vec::len).sum::<usize>();
        let final_budgets = root_budgets(&curr, root, 1);
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
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        curr.rebuild();
        let prev = curr.clone();
        curr.union(a, b);
        curr.rebuild();
        let root = curr.find(a);

        let budgets = root_budgets(&curr, root, 1);
        let matches = enumerate_matches_rooted(&curr, &prev, &budgets);
        let previous_classes = matches
            .iter()
            .filter(|((c, _), _)| *c == root)
            .flat_map(|(_, ms)| ms.iter().map(|m| m.prev_class))
            .collect::<HashSet<_>>();

        assert_eq!(previous_classes.len(), 2);
        assert!(previous_classes.contains(&prev.find(a)));
        assert!(previous_classes.contains(&prev.find(b)));
    }
}
