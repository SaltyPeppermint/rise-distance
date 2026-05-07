//! Novel-reachable term counts.
//!
//! A term extracted from `curr` is novel iff it is not extractable from any
//! e-class in `prev`. This module computes:
//!
//! - `joint[(c, pc)](s)` — for each (curr-class, prev-class) pair, the number
//!   of size-`s` extractions rooted at curr's `c` that are also extractable
//!   from prev's `pc`.
//! - `matches[(c, idx)]` — for each curr e-node, the list of prev matches
//!   (a prev e-class plus the canonical prev-class ids of the matching prev
//!   node's children).
//! - `data[c](s) = plain[c](s) - sum_pc joint[(c, pc)](s)` — the per-class
//!   histogram of novel-reachable extractions.
//!
//! Two prev classes cannot share a term (`prev.lookup(t)` is unique once
//! prev has been rebuilt), so the sum over `pc` does not double-count.

use std::borrow::Cow;

use egg::{EGraph, Id};
use hashbrown::{HashMap, HashSet};

use super::{Counter, PlainTermCount};
use crate::utils::UniqueQueue;
use crate::{MyAnalysis, MyLanguage};

/// One match of a curr e-node in prev. The `prev_children` are canonical prev
/// class ids of the matching prev node's children, in the same order as the
/// curr node's children.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeMatch {
    pub prev_class: Id,
    pub prev_children: Vec<Id>,
}

/// Joint extractability table + per-node prev matches + derived novel
/// histograms.
pub struct NovelTermCount<'p, 'g, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    curr: &'g EGraph<L, N>,
    prev: &'g EGraph<L, N>,
    plain: &'p PlainTermCount<C>,

    /// Per `(curr_class, prev_class)` pair: histogram of size -> count of
    /// extractions rooted at curr's class that are also extractable from
    /// prev's class.
    joint: HashMap<(Id, Id), HashMap<usize, C>>,

    /// Per curr class: the set of prev classes that share at least one
    /// extraction with it (i.e., the second-coordinate keys of `joint`).
    cover: HashMap<Id, Vec<Id>>,

    /// Per `(curr_class, node_idx)`: every match of that e-node in prev.
    matches: HashMap<(Id, usize), Vec<NodeMatch>>,

    /// Per curr class: histogram of novel-reachable extraction sizes.
    /// `data[c][s] = plain[c][s] - sum_pc joint[(c, pc)][s]`.
    data: HashMap<Id, HashMap<usize, C>>,
}

impl<'p, 'g, C, L, N> NovelTermCount<'p, 'g, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    #[must_use]
    pub fn new(
        max_size: usize,
        curr: &'g EGraph<L, N>,
        prev: &'g EGraph<L, N>,
        plain: &'p PlainTermCount<C>,
    ) -> Self {
        let matches = enumerate_matches(curr, prev);
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

    #[must_use]
    pub fn data(&self) -> &HashMap<Id, HashMap<usize, C>> {
        &self.data
    }

    #[must_use]
    pub fn plain(&self) -> &'p PlainTermCount<C> {
        self.plain
    }

    #[must_use]
    pub fn curr(&self) -> &'g EGraph<L, N> {
        self.curr
    }

    #[must_use]
    pub fn prev(&self) -> &'g EGraph<L, N> {
        self.prev
    }

    #[must_use]
    pub fn joint(&self) -> &HashMap<(Id, Id), HashMap<usize, C>> {
        &self.joint
    }

    /// Joint histogram for a `(curr_class, prev_class)` pair. Empty if the
    /// two classes share no extraction.
    pub(crate) fn joint_histogram(&self, curr_id: Id, prev_id: Id) -> Cow<'_, HashMap<usize, C>> {
        let curr_canon = self.curr.find(curr_id);
        let prev_canon = self.prev.find(prev_id);
        match self.joint.get(&(curr_canon, prev_canon)) {
            Some(h) => Cow::Borrowed(h),
            None => Cow::Owned(HashMap::default()),
        }
    }

    /// Novel histogram for a curr class. Empty if every extraction is
    /// representable in some prev class.
    pub(crate) fn novel_histogram(&self, curr_id: Id) -> Cow<'_, HashMap<usize, C>> {
        let canon = self.curr.find(curr_id);
        match self.data.get(&canon) {
            Some(h) => Cow::Borrowed(h),
            None => Cow::Owned(HashMap::default()),
        }
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

fn build_cover<C: Counter>(joint: &HashMap<(Id, Id), HashMap<usize, C>>) -> HashMap<Id, Vec<Id>> {
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
// Phase 1 — match enumeration.
// ============================================================================

/// Enumerate all matches of every curr e-node in `prev`.
///
/// Bottom-up fixpoint: per curr e-node `n` with children `c_1..c_k`, try every
/// combination `(pc_1, .., pc_k)` from `cover[c_i]` (the set of prev classes
/// that share a term with `c_i`). For each combo, look up the translated node
/// in `prev`; if found, record the match and add the discovered prev class to
/// `cover[curr_class_of_n]`. Iterate until no new matches/cover entries.
fn enumerate_matches<L, N>(
    curr: &EGraph<L, N>,
    prev: &EGraph<L, N>,
) -> HashMap<(Id, usize), Vec<NodeMatch>>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    let mut cover: HashMap<Id, HashSet<Id>> = HashMap::new();
    let mut matches: HashMap<(Id, usize), Vec<NodeMatch>> = HashMap::new();
    let mut seen: HashSet<(Id, usize, Id, Vec<Id>)> = HashSet::new();

    loop {
        let mut changed = false;

        for class in curr.classes() {
            let c = curr.find(class.id);
            for (idx, node) in class.nodes.iter().enumerate() {
                let children = node.children();
                let child_canons: Vec<Id> = children.iter().map(|cc| curr.find(*cc)).collect();

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
                        let key = (c, idx, pc_canon, combo.clone());
                        if seen.insert(key) {
                            matches.entry((c, idx)).or_default().push(NodeMatch {
                                prev_class: pc_canon,
                                prev_children: combo,
                            });
                            if cover.entry(c).or_default().insert(pc_canon) {
                                // Newly discovered cover entry; another pass
                                // might find more matches via this class.
                            }
                            changed = true;
                        }
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    matches
}

/// Cartesian product of `cover[child_i]` over `i`. For zero-arity nodes,
/// returns `[[]]` (a single empty combination).
fn child_combinations(children: &[Id], cover: &HashMap<Id, HashSet<Id>>) -> Vec<Vec<Id>> {
    let mut combos: Vec<Vec<Id>> = vec![Vec::new()];
    for child in children {
        let Some(opts) = cover.get(child) else {
            return Vec::new();
        };
        if opts.is_empty() {
            return Vec::new();
        }
        let mut next = Vec::with_capacity(combos.len() * opts.len());
        for prefix in &combos {
            for opt in opts {
                let mut p = prefix.clone();
                p.push(*opt);
                next.push(p);
            }
        }
        combos = next;
    }
    combos
}

// ============================================================================
// Phase 2 — joint count fixpoint.
// ============================================================================

/// Compute `joint[(c, pc)]` for every pair appearing in matches, via a
/// bottom-up fixpoint similar to `PlainTermCount::new`.
fn compute_joint<C, L, N>(
    max_size: usize,
    curr: &EGraph<L, N>,
    matches: &HashMap<(Id, usize), Vec<NodeMatch>>,
) -> HashMap<(Id, Id), HashMap<usize, C>>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    // Collect all (c, pc) pairs we need to compute.
    let mut pairs: HashSet<(Id, Id)> = HashSet::new();
    for ((c, _), ms) in matches {
        for m in ms {
            pairs.insert((*c, m.prev_class));
        }
    }

    // Group matches by (curr_class, prev_class) for efficient per-pair update.
    let mut by_pair: HashMap<(Id, Id), Vec<(usize, &NodeMatch)>> = HashMap::new();
    for ((c, idx), ms) in matches {
        for m in ms {
            by_pair
                .entry((*c, m.prev_class))
                .or_default()
                .push((*idx, m));
        }
    }

    let mut joint: HashMap<(Id, Id), HashMap<usize, C>> = HashMap::new();

    let mut pending: UniqueQueue<(Id, Id)> = pairs.iter().copied().collect();

    // Reverse dependency: given (c, pc), which pairs depend on its histogram?
    // A pair (c', pc') depends on (c, pc) iff some match of c' (with
    // prev_class pc') has a child position whose curr class is c and prev
    // class is pc.
    let mut deps: HashMap<(Id, Id), HashSet<(Id, Id)>> = HashMap::new();
    for ((c_prime, _), ms) in matches {
        for m in ms {
            // For each child slot, we depend on (curr_child_class, m.prev_children[i]).
            // But we don't know curr_child_class without looking at the node.
            // So we need to iterate nodes too. Defer: iterate all nodes here.
            let _ = (c_prime, m);
        }
    }
    for class in curr.classes() {
        let c_prime = curr.find(class.id);
        for (idx, node) in class.nodes.iter().enumerate() {
            let Some(ms) = matches.get(&(c_prime, idx)) else {
                continue;
            };
            for m in ms {
                for (child, prev_child) in node.children().iter().zip(m.prev_children.iter()) {
                    let cc = curr.find(*child);
                    deps.entry((cc, *prev_child))
                        .or_default()
                        .insert((c_prime, m.prev_class));
                }
            }
        }
    }

    while let Some((c, pc)) = pending.pop() {
        let new_hist = compute_pair_histogram::<C, L, N>(
            max_size,
            curr,
            c,
            pc,
            by_pair.get(&(c, pc)).map_or(&[][..], Vec::as_slice),
            &joint,
        );

        if joint.get(&(c, pc)).is_none_or(|v| *v != new_hist) {
            if new_hist.is_empty() {
                joint.remove(&(c, pc));
            } else {
                joint.insert((c, pc), new_hist);
            }
            if let Some(dependents) = deps.get(&(c, pc)) {
                pending.extend(dependents.iter().copied());
            }
        }
    }

    joint
}

/// Compute the histogram for a single `(c, pc)` pair from its matches.
fn compute_pair_histogram<C, L, N>(
    max_size: usize,
    curr: &EGraph<L, N>,
    c: Id,
    _pc: Id,
    pair_matches: &[(usize, &NodeMatch)],
    joint: &HashMap<(Id, Id), HashMap<usize, C>>,
) -> HashMap<usize, C>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    let mut acc: HashMap<usize, C> = HashMap::new();

    let eclass = &curr[c];

    for (idx, m) in pair_matches {
        let node = &eclass.nodes[*idx];
        let children = node.children();

        let base = 1_usize;
        if children.is_empty() {
            if base <= max_size {
                acc.entry(base)
                    .and_modify(|x| *x += &C::one())
                    .or_insert_with(C::one);
            }
            continue;
        }

        let Some(budget) = max_size.checked_sub(base) else {
            continue;
        };

        // Build histograms for each child, looking up
        // `joint[(curr.find(child_i), m.prev_children[i])]`.
        let histograms: Vec<HashMap<usize, C>> = children
            .iter()
            .zip(m.prev_children.iter())
            .map(|(child, prev_child)| {
                let cc = curr.find(*child);
                joint.get(&(cc, *prev_child)).cloned().unwrap_or_default()
            })
            .collect();

        if histograms.iter().any(HashMap::is_empty) && !children.is_empty() {
            continue;
        }

        let conv = PlainTermCount::<C>::convolve(&histograms, budget);
        for (s, count) in conv {
            let total = s + base;
            acc.entry(total)
                .and_modify(|x| *x += &count)
                .or_insert(count);
        }
    }

    acc
}

// ============================================================================
// Phase 3 — derive novel histograms.
// ============================================================================

fn derive_novel<C: Counter>(
    plain: &HashMap<Id, HashMap<usize, C>>,
    joint: &HashMap<(Id, Id), HashMap<usize, C>>,
) -> HashMap<Id, HashMap<usize, C>> {
    // Aggregate sum_pc joint[(c, pc)] per curr class.
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

    let mut out: HashMap<Id, HashMap<usize, C>> = HashMap::with_capacity(plain.len());
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
    use crate::egg::Math;

    fn sym(name: &str) -> Math {
        Math::Symbol(name.into())
    }

    #[test]
    fn no_novelty_yields_empty() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let plain = PlainTermCount::<BigUint>::new(5, &graph);
        let novel = NovelTermCount::new(5, &graph, &graph, &plain);

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
        let novel = NovelTermCount::new(5, &curr, &prev, &plain);

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

        let plain = PlainTermCount::<BigUint>::new(5, &curr);
        let novel = NovelTermCount::new(5, &curr, &prev, &plain);

        let root_canon = curr.find(root);
        // Plain at size 3 = 4 (aa, ab, ba, bb). Only Add(a, b) was in prev.
        // So novel = 4 - 1 = 3.
        assert_eq!(novel.data()[&root_canon][&3], BigUint::from(3u32));
    }
}
