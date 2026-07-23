//! Novel-reachable term counts.
//!
//! A term extracted from `curr` is novel iff it is not extractable from any
//! e-class in `prev`. This module computes:
//!
//! - `joint[(c, pc)](s)`: for each (curr-class, prev-class) pair, the number
//!   of size-`s` extractions rooted at curr's `c` that are also extractable
//!   from prev's `pc`.
//! - `matches[(c, idx)]`: for each curr e-node, the list of prev matches
//!   (a prev e-class plus the canonical prev-class ids of the matching prev
//!   node's children).
//! - `data[c](s) = plain[c](s) - sum_pc joint[(c, pc)](s)`: the per-class
//!   histogram of novel-reachable extractions.
//!
//! Two prev classes cannot share a term (`prev.lookup(t)` is unique once
//! prev has been rebuilt), so the sum over `pc` does not double-count.

use egg::{Analysis, EGraph, Id, Language};
use hashbrown::{HashMap, HashSet};
use num::BigUint;
use smallvec::SmallVec;

use crate::Counter;
use crate::sampling::count::{LayeredDp, PlainTermCount, plain_dp};

/// Inline-allocated list of child class ids. Sized for the typical e-node
/// arity (0–2); higher-arity nodes spill to the heap transparently.
pub type ChildIds = SmallVec<[Id; 2]>;

/// One match of a curr e-node in prev. The `prev_children` are canonical prev
/// class ids of the matching prev node's children, in the same order as the
/// curr node's children.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeMatch {
    pub prev_class: Id,
    pub prev_children: ChildIds,
}

/// Per `(curr_class, node_idx)`: every match of that e-node in prev.
pub type NodeMatches = HashMap<(Id, usize), Vec<NodeMatch>>;

/// Per curr class: the set of prev classes that share at least one extraction
/// with it. Used internally during match enumeration; the exposed `cover`
/// field on [`NovelTermCount`] is a `Vec<Id>`-valued variant derived from the
/// joint table.
type MatchCover = HashMap<Id, HashSet<Id>>;

/// Dedup key for match enumeration: `(curr_class, node_idx, prev_class,
/// prev_children)`.
type MatchKeys = HashSet<(Id, usize, Id, ChildIds)>;

/// Per `(curr_class, prev_class)` pair: histogram of size -> count of
/// extractions rooted at curr's class that are also extractable from prev's
/// class.
type JointTable<C> = HashMap<(Id, Id), HashMap<usize, C>>;

/// Joint extractability table + per-node prev matches + derived novel
/// histograms.
#[derive(Debug)]
pub struct NovelTermCount<'g, C, L, N>
where
    C: Counter,
    L: Language,
    N: Analysis<L>,
{
    curr: &'g EGraph<L, N>,
    prev: &'g EGraph<L, N>,
    plain: PlainTermCount<C>,

    /// Per `(curr_class, prev_class)` pair: histogram of size -> count of
    /// extractions rooted at curr's class that are also extractable from
    /// prev's class.
    joint: JointTable<C>,

    /// Per curr class: the set of prev classes that share at least one
    /// extraction with it (i.e., the second-coordinate keys of `joint`).
    cover: HashMap<Id, Vec<Id>>,

    /// Per `(curr_class, node_idx)`: every match of that e-node in prev.
    matches: NodeMatches,

    /// Per curr class: histogram of novel-reachable extraction sizes.
    /// `data[c][s] = plain[c][s] - sum_pc joint[(c, pc)][s]`.
    data: HashMap<Id, HashMap<usize, C>>,
}

impl<'g, C: Counter, L: Language, N: Analysis<L>> NovelTermCount<'g, C, L, N> {
    /// Convenience constructor that enumerates the matches itself; the
    /// production path goes through [`with_matches`](Self::with_matches).
    #[cfg(test)]
    #[must_use]
    pub fn new(
        max_size: usize,
        curr: &'g EGraph<L, N>,
        prev: &'g EGraph<L, N>,
        plain: PlainTermCount<C>,
    ) -> Self {
        Self::with_matches(max_size, curr, prev, plain, enumerate_matches(curr, prev))
    }

    /// Run the joint counting analysis, with the match enumeration
    /// precomputed by the caller. The matches are independent of `max_size`,
    /// so callers that run several analyses on the same egraph pair (see
    /// `PrecomputePackage::backoff_precompute`) can share one enumeration.
    #[must_use]
    pub(crate) fn with_matches(
        max_size: usize,
        curr: &'g EGraph<L, N>,
        prev: &'g EGraph<L, N>,
        plain: PlainTermCount<C>,
        matches: NodeMatches,
    ) -> Self {
        let joint = compute_joint(max_size, curr, &matches);
        let cover = build_cover(&joint);
        let data = derive_novel(plain.data(), &joint);

        Self {
            curr,
            prev,
            plain,
            joint,
            cover,
            matches,
            data,
        }
    }

    /// Per-class novel histograms. Keyed by **canonical** curr ids. Callers
    /// looking up by an `Id` that hasn't been through `curr.find` may miss.
    #[must_use]
    pub const fn data(&self) -> &HashMap<Id, HashMap<usize, C>> {
        &self.data
    }

    #[must_use]
    pub const fn plain(&self) -> &PlainTermCount<C> {
        &self.plain
    }

    #[must_use]
    pub const fn curr(&self) -> &'g EGraph<L, N> {
        self.curr
    }

    #[must_use]
    pub const fn prev(&self) -> &'g EGraph<L, N> {
        self.prev
    }

    /// Joint histogram for a `(curr_class, prev_class)` pair. `None` if the
    /// two classes share no extraction.
    pub(crate) fn joint_histogram(&self, curr_id: Id, prev_id: Id) -> Option<&HashMap<usize, C>> {
        let curr_canon = self.curr.find(curr_id);
        let prev_canon = self.prev.find(prev_id);
        self.joint.get(&(curr_canon, prev_canon))
    }

    /// Novel histogram for a curr class. `None` if every extraction is
    /// representable in some prev class.
    pub(crate) fn novel_histogram(&self, curr_id: Id) -> Option<&HashMap<usize, C>> {
        let canon = self.curr.find(curr_id);
        self.data.get(&canon)
    }

    pub(crate) fn matches_of(&self, curr_class: Id, node_idx: usize) -> &[NodeMatch] {
        let canon = self.curr.find(curr_class);
        self.matches
            .get(&(canon, node_idx))
            .map_or(&[][..], Vec::as_slice)
    }

    /// Set of prev classes that share at least one extraction with the given
    /// curr class. Used by the sampler to enumerate per-slot agreement
    /// options (any prev class in this set, plus NOVEL).
    pub(crate) fn cover_of(&self, curr_class: Id) -> &[Id] {
        let canon = self.curr.find(curr_class);
        self.cover.get(&canon).map_or(&[][..], Vec::as_slice)
    }
}

// The exposed `cover` is built from the *joint* keys, not from match
// enumeration's internal `cover`. The two can differ: a `(c, pc)` pair whose
// matches all involve some child with empty `joint` within `max_size` ends up
// dropped in `compute_joint`. That's fine for sampling: a missing `pc` had
// joint count 0 anyway, so neither slot enumeration nor `completes_some_match`
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

/// Enumerate all matches of every curr e-node in `prev`.
///
/// Bottom-up fixpoint: per curr e-node `n` with children `c_1..c_k`, try every
/// combination `(pc_1, .., pc_k)` from `cover[c_i]` (the set of prev classes
/// that share a term with `c_i`). For each combo, look up the translated node
/// in `prev`; if found, record the match and add the discovered prev class to
/// `cover[curr_class_of_n]`. Iterate until no new matches/cover entries.
///
/// The result is independent of any size limit and can be shared across
/// several counting runs on the same egraph pair.
pub fn enumerate_matches<L: Language, N: Analysis<L>>(
    curr: &EGraph<L, N>,
    prev: &EGraph<L, N>,
) -> NodeMatches {
    let mut cover = MatchCover::new();
    let mut matches = NodeMatches::new();
    let mut seen = MatchKeys::new();

    let mut changed = true;
    while changed {
        changed = false;

        for class in curr.classes() {
            let c = curr.find(class.id);
            for (idx, node) in class.nodes.iter().enumerate() {
                let children = node.children();
                let child_canons = children
                    .iter()
                    .map(|cc| curr.find(*cc))
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
                    if let Some(pc_class) = prev.lookup(translated) {
                        let pc_canon = prev.find(pc_class);
                        if seen.insert((c, idx, pc_canon, combo.clone())) {
                            matches.entry((c, idx)).or_default().push(NodeMatch {
                                prev_class: pc_canon,
                                prev_children: combo,
                            });
                            // Newly discovered cover entry; another pass
                            // might find more matches via this class.
                            cover.entry(c).or_default().insert(pc_canon);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    matches
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

/// Compute `joint[(c, pc)]` for every pair appearing in matches, as a
/// size-layered DP over `(curr_class, prev_class)` pairs: the matched e-node
/// contributes 1 to a term's size, so a pair's count at `size` depends only
/// on child-pair counts at sizes strictly below it — the same stratification
/// [`count_terms`](super::count_terms) uses for the plain counts, and the
/// same [`LayeredDp`] kernel (see `docs/counting/novel_size_search.md`). Each
/// `(pair, size)` cell is computed exactly once, even on cyclic e-graphs.
fn compute_joint<C: Counter, L: Language, N: Analysis<L>>(
    max_size: usize,
    curr: &EGraph<L, N>,
    matches: &NodeMatches,
) -> JointTable<C> {
    let children_of = joint_children_of(curr, matches);
    let budgets = children_of.keys().map(|&pair| (pair, max_size)).collect();
    let mut dp = LayeredDp::new(children_of, budgets);
    for _ in 0..max_size {
        dp.step();
    }
    dp.into_parts().0
}

/// The joint DP's structure in [`LayeredDp`] shape: per `(curr_class,
/// prev_class)` pair, one node per match with that prev class, holding the
/// child pair keys `(curr_child, prev_child)` in slot order.
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

/// The first `stop_after` sizes (ascending) at which `root` has at least one
/// novel term, up to `max_size`.
///
/// This runs the same plain/joint recurrence as
/// [`NovelTermCount::with_matches`] with exact [`BigUint`] counts, but
/// restricts both DPs to what `root` can reach. The DPs advance in lockstep;
/// after layer `s`,
/// `plain[root](s) - sum_pc joint[(root, pc)](s)` is final. The scan can
/// therefore stop at the requested number of sizes without computing any
/// larger layers or sampler-only caches. See `docs/counting/novel_size_search.md`.
pub fn find_novel_root_sizes<L: Language, N: Analysis<L>>(
    max_size: usize,
    curr: &EGraph<L, N>,
    root: Id,
    matches: &NodeMatches,
    stop_after: usize,
) -> Vec<usize> {
    let root = curr.find(root);
    let mut plain: LayeredDp<Id, BigUint> = plain_dp(max_size, curr, Some(&[root]));

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

    let mut sizes = Vec::new();
    for _ in 0..max_size {
        let size = plain.step();
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
            sizes.push(size);
            if sizes.len() >= stop_after {
                break;
            }
        }
    }
    sizes
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
    use crate::langs::math::Math;
    use crate::test_utils::sym;

    #[test]
    fn no_novelty_yields_empty() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let plain = PlainTermCount::<BigUint>::new(5, &graph);
        let novel = NovelTermCount::new(5, &graph, &graph, plain);

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

        let plain = PlainTermCount::<BigUint>::new(5, &curr);
        let novel = NovelTermCount::new(5, &curr, &prev, plain);

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

    /// The root-restricted scan must agree with the full novel histogram for
    /// every class taken as root.
    fn assert_size_scan_agrees(curr: &EGraph<Math, ()>, prev: &EGraph<Math, ()>, max_size: usize) {
        let plain = PlainTermCount::<BigUint>::new(max_size, curr);
        let novel = NovelTermCount::new(max_size, curr, prev, plain);
        let matches = enumerate_matches(curr, prev);

        for class in curr.classes() {
            let mut expected = novel
                .data()
                .get(&curr.find(class.id))
                .map(|hist| hist.keys().copied().collect::<Vec<_>>())
                .unwrap_or_default();
            expected.sort_unstable();
            let found = find_novel_root_sizes(max_size, curr, class.id, &matches, usize::MAX);
            assert_eq!(
                found, expected,
                "novel sizes diverge for class {}",
                class.id
            );
        }
    }

    #[test]
    fn size_scan_agrees_with_novel_histogram() {
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let _root = curr.add(Math::Ln(a));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.rebuild();

        assert_size_scan_agrees(&curr, &prev, 5);
    }

    #[test]
    fn size_scan_agrees_when_nothing_is_novel() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        assert_size_scan_agrees(&graph, &graph, 5);
    }

    #[test]
    fn size_scan_agrees_on_cyclic_graph() {
        // Unioning the root of (+ a b) with `a` creates a cycle, so novel
        // terms exist at unboundedly many sizes.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let apb = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, apb);
        curr.rebuild();

        assert_size_scan_agrees(&curr, &prev, 11);
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

        let plain = PlainTermCount::<BigUint>::new(5, &curr);
        let novel = NovelTermCount::new(5, &curr, &prev, plain);

        let root_canon = curr.find(root);
        // Plain at size 3 = 4 (aa, ab, ba, bb). Only Add(a, b) was in prev.
        // So novel = 4 - 1 = 3.
        assert_eq!(novel.data()[&root_canon][&3], BigUint::from(3u32));
    }
}
