//! Novelty-constrained sampler. See [`NovelSampler`].

use egg::{EGraph, Id, RecExpr};
use hashbrown::HashMap;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::count::{NodeMatch, NovelTermCount, PlainTermCount};
use crate::sampling::Weigher;
use crate::{Counter, MyAnalysis, MyLanguage, OriginLang, Sampler, lower, stack_children};

/// Sampler that draws size-targeted terms which are *not* extractable from
/// any e-class in `prev` — i.e., terms that carry information learned in
/// the iteration that produced `curr` from `prev`.
///
/// Recursion has two modes:
/// - [`Mode::AgreeWith`] — the subtree must equal an extraction of a specific
///   prev e-class (drawn from the joint table).
/// - [`Mode::Novel`] — the subtree must not be extractable from any prev
///   e-class.
///
/// In `Novel` mode we enumerate, for each curr e-node `n`, all *agreement
/// profiles* `(a_1, …, a_k)` over `n`'s children. Each `a_i` is either
/// `Novel` or a specific prev class drawn from one of `n`'s matches. The
/// profile is "novel-via-`n`" iff it doesn't fully match any of `n`'s
/// matches. We sample `(n, profile)` weighted by the count of size-`s`
/// extractions consistent with that profile, then recurse children
/// accordingly.
pub struct NovelSampler<'a, 'g, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    novel: &'a NovelTermCount<'g, C, L, N>,
    root: Id,
    weigher: W,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    AgreeWith(Id),
    Novel,
}

// /// One sampling candidate in [`Mode::AgreeWith`]: the chosen e-node index in
// /// the curr class, the matching prev node, the candidate's weighted count,
// /// and the per-child histograms used to draw child sizes.
// type AgreeCandidate<'h, C> = (usize, &'h NodeMatch, C, Vec<&'h HashMap<usize, C>>);

// /// One sampling candidate in [`Mode::Novel`]: the chosen e-node index, the
// /// agreement profile, the candidate's weighted count, and the per-child
// /// histograms.
// type NovelCandidate<'h, C> = (usize, Vec<Option<Id>>, C, Vec<&'h HashMap<usize, C>>);

impl<'a, 'g, C, L, N, W> NovelSampler<'a, 'g, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    #[must_use]
    pub fn new(novel: &'a NovelTermCount<'g, C, L, N>, root: Id, weigher: W) -> Self {
        Self {
            novel,
            root,
            weigher,
        }
    }

    fn graph(&self) -> &'g EGraph<L, N> {
        self.novel.curr()
    }

    fn sample_with_mode(
        &self,
        id: Id,
        size: usize,
        mode: Mode,
        rng: &mut ChaCha12Rng,
    ) -> RecExpr<OriginLang<L>> {
        match mode {
            Mode::AgreeWith(pc) => self.sample_agree(id, size, pc, rng),
            Mode::Novel => self.sample_novel(id, size, rng),
        }
    }

    fn sample_agree(
        &self,
        id: Id,
        size: usize,
        pc: Id,
        rng: &mut ChaCha12Rng,
    ) -> RecExpr<OriginLang<L>> {
        let graph = self.graph();
        let canon_id = graph.find(id);
        let eclass = &graph[canon_id];
        let child_budget = size - 1;

        // Per (node_idx, match) where match.prev_class == pc, compute the
        // count of size-`size` extractions through that node-match pair.
        let candidates = eclass
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(idx, node)| {
                self.novel
                    .matches_of(canon_id, idx)
                    .iter()
                    .filter(|m| m.prev_class == pc)
                    .filter_map(move |m| {
                        let child_hists =
                            self.agree_child_histograms(node.children(), &m.prev_children)?;
                        let count = convolve_at::<C>(&child_hists, child_budget)?;
                        Some((idx, m, self.weigher.node_weight(&count), child_hists))
                    })
            })
            .collect::<Vec<_>>();

        let dist = WeightedIndex::new(candidates.iter().map(|(_, _, w, _)| w))
            .expect("AgreeWith: at least one candidate match (precondition)");
        let pick = dist.sample(rng);
        let (idx, m, _, child_hists) = &candidates[pick];
        let node = &eclass.nodes[*idx];

        let children = self.sample_children_with_modes(
            node.children(),
            child_hists,
            &m.prev_children
                .iter()
                .map(|&p| Mode::AgreeWith(p))
                .collect::<Vec<_>>(),
            child_budget,
            rng,
        );

        stack_children(&children, OriginLang::new(node.clone(), canon_id))
    }

    fn sample_novel(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let graph = self.graph();
        let canon_id = graph.find(id);
        let eclass = &graph[canon_id];
        let child_budget = size - 1;

        // Per node, enumerate agreement profiles. Each profile is a vector
        // of slot agreements: `None` = NOVEL, `Some(pc)` = agree with prev
        // class pc. Profile is novel-via-n iff it doesn't equal any
        // match.prev_children.
        let candidates = eclass
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(idx, node)| {
                let n_matches = self.novel.matches_of(canon_id, idx);
                let children = node.children();

                // Per child slot, the prev classes that share at least one
                // extraction with that child class — these are the valid
                // "agree-with" options. NOVEL is also always an option.
                let slot_options = children
                    .iter()
                    .map(|child| {
                        let mut opts = vec![None];
                        opts.extend(self.novel.cover_of(*child).iter().copied().map(Some));
                        opts
                    })
                    .collect::<Vec<_>>();

                enumerate_profiles(&slot_options)
                    .into_iter()
                    .filter(|profile| !completes_some_match(profile, n_matches))
                    .filter_map(move |profile| {
                        let child_hists = self.novel_child_histograms(children, &profile)?;
                        let count = convolve_at(&child_hists, child_budget)?;
                        Some((idx, profile, self.weigher.node_weight(&count), child_hists))
                    })
            })
            .collect::<Vec<_>>();

        let dist = WeightedIndex::new(candidates.iter().map(|(_, _, w, _)| w)).expect(
            "Novel: at least one non-completing profile (precondition: novel data nonempty)",
        );
        let pick = dist.sample(rng);
        let (idx, profile, _, child_hists) = &candidates[pick];
        let node = &eclass.nodes[*idx];

        let modes = profile
            .iter()
            .map(|a| match a {
                None => Mode::Novel,
                Some(pc) => Mode::AgreeWith(*pc),
            })
            .collect::<Vec<_>>();

        let children = self.sample_children_with_modes(
            node.children(),
            child_hists,
            &modes,
            child_budget,
            rng,
        );

        stack_children(&children, OriginLang::new(node.clone(), canon_id))
    }

    /// Per-child histograms when sampling under [`Mode::AgreeWith`]. `None`
    /// if any slot has no shared extraction — that candidate's count would
    /// be zero anyway.
    fn agree_child_histograms(
        &self,
        children: &[Id],
        prev_children: &[Id],
    ) -> Option<Vec<&'a HashMap<usize, C>>> {
        children
            .iter()
            .zip(prev_children.iter())
            .map(|(child, &pc)| self.novel.joint_histogram(*child, pc))
            .collect()
    }

    /// Per-child histograms for an agreement profile in [`Mode::Novel`].
    /// `None` if any slot is unsatisfiable.
    fn novel_child_histograms(
        &self,
        children: &[Id],
        profile: &[Option<Id>],
    ) -> Option<Vec<&'a HashMap<usize, C>>> {
        children
            .iter()
            .zip(profile.iter())
            .map(|(child, slot)| match slot {
                None => self.novel.novel_histogram(*child),
                Some(pc) => self.novel.joint_histogram(*child, *pc),
            })
            .collect()
    }

    fn enumerate_with_mode(&self, id: Id, size: usize, mode: Mode) -> Vec<RecExpr<OriginLang<L>>> {
        match mode {
            Mode::AgreeWith(pc) => self.enumerate_agree(id, size, pc),
            Mode::Novel => self.enumerate_novel(id, size),
        }
    }

    /// Enumerate all distinct novel terms of exactly `size` rooted at `id`.
    fn enumerate_novel(&self, id: Id, size: usize) -> Vec<RecExpr<OriginLang<L>>> {
        let graph = self.graph();
        let canon_id = graph.find(id);
        let Some(child_budget) = size.checked_sub(1) else {
            return Vec::new();
        };
        let eclass = &graph[canon_id];

        let mut results = Vec::new();

        for (idx, node) in eclass.nodes.iter().enumerate() {
            let n_matches = self.novel.matches_of(canon_id, idx);
            let children = node.children();

            let slot_options = children
                .iter()
                .map(|child| {
                    let mut opts = vec![None];
                    opts.extend(self.novel.cover_of(*child).iter().copied().map(Some));
                    opts
                })
                .collect::<Vec<_>>();

            for profile in enumerate_profiles(&slot_options) {
                if completes_some_match(&profile, n_matches) {
                    continue;
                }
                let modes = profile
                    .iter()
                    .map(|a| match a {
                        None => Mode::Novel,
                        Some(pc) => Mode::AgreeWith(*pc),
                    })
                    .collect::<Vec<_>>();

                for combo in self.enumerate_children_with_modes(children, &modes, child_budget) {
                    results.push(stack_children(
                        &combo,
                        OriginLang::new(node.clone(), canon_id),
                    ));
                }
            }
        }

        results
    }

    /// Enumerate all distinct terms of exactly `size` rooted at `id` that
    /// agree with prev class `pc`.
    fn enumerate_agree(&self, id: Id, size: usize, pc: Id) -> Vec<RecExpr<OriginLang<L>>> {
        let graph = self.graph();
        let canon_id = graph.find(id);
        let Some(child_budget) = size.checked_sub(1) else {
            return Vec::new();
        };
        let eclass = &graph[canon_id];

        let mut results = Vec::new();

        for (idx, node) in eclass.nodes.iter().enumerate() {
            let children = node.children();
            for m in self
                .novel
                .matches_of(canon_id, idx)
                .iter()
                .filter(|m| m.prev_class == pc)
            {
                let modes = m
                    .prev_children
                    .iter()
                    .map(|&p| Mode::AgreeWith(p))
                    .collect::<Vec<_>>();
                for combo in self.enumerate_children_with_modes(children, &modes, child_budget) {
                    results.push(stack_children(
                        &combo,
                        OriginLang::new(node.clone(), canon_id),
                    ));
                }
            }
        }

        results
    }

    /// Cartesian product over every `(child_size_tuple, child_term_tuple)`
    /// such that child sizes sum to `budget` and each child term has the
    /// chosen mode at the chosen size.
    fn enumerate_children_with_modes(
        &self,
        children_ids: &[Id],
        modes: &[Mode],
        budget: usize,
    ) -> Vec<Vec<RecExpr<OriginLang<L>>>> {
        let mut acc: Vec<(usize, Vec<RecExpr<OriginLang<L>>>)> = vec![(budget, Vec::new())];

        for (i, &c_id) in children_ids.iter().enumerate() {
            let mode = modes[i];
            let mut next = Vec::new();
            for (remaining, partial) in acc {
                for s in 1..=remaining {
                    for term in self.enumerate_with_mode(c_id, s, mode) {
                        let mut combo = partial.clone();
                        combo.push(term);
                        next.push((remaining - s, combo));
                    }
                }
            }
            acc = next;
        }

        acc.into_iter()
            .filter(|(remaining, _)| *remaining == 0)
            .map(|(_, combo)| combo)
            .collect()
    }

    /// Sample child sizes via suffix convolution and recurse with the given
    /// per-child modes.
    fn sample_children_with_modes(
        &self,
        children_ids: &[Id],
        child_hists: &[&HashMap<usize, C>],
        modes: &[Mode],
        child_budget: usize,
        rng: &mut ChaCha12Rng,
    ) -> Vec<RecExpr<OriginLang<L>>> {
        let suffix = PlainTermCount::<C>::suffix_convolutions(child_hists, child_budget);

        let mut remaining = child_budget;
        let mut sampled = Vec::with_capacity(children_ids.len());

        for (i, &c_id) in children_ids.iter().enumerate() {
            let candidates = child_hists[i]
                .iter()
                .filter_map(|(&s, count)| {
                    let r = remaining.checked_sub(s)?;
                    let rest = suffix[i + 1].get(&r)?;
                    (*rest != C::zero()).then(|| (s, self.weigher.child_weight(count, rest)))
                })
                .collect::<Vec<_>>();

            let dist = WeightedIndex::new(candidates.iter().map(|(_, w)| w))
                .expect("at least one valid child size for chosen profile");
            let s = candidates[dist.sample(rng)].0;
            remaining -= s;
            sampled.push(self.sample_with_mode(c_id, s, modes[i], rng));
        }

        sampled
    }
}

impl<C, L, N, W> Sampler<C, L, N> for NovelSampler<'_, '_, C, L, N, W>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
    W: Weigher<C>,
{
    fn root(&self) -> Id {
        self.root
    }

    fn possible_size(&self, id: Id, size: usize, samples: u64) -> bool {
        let canon_id = self.graph().find(id);
        let Some(count) = self.novel.data().get(&canon_id).and_then(|h| h.get(&size)) else {
            return false;
        };
        samples.try_into().is_ok_and(|s: C| count > &s)
    }

    fn term_sizes(&self, id: Id) -> Vec<usize> {
        let canon_id = self.graph().find(id);
        self.novel
            .data()
            .get(&canon_id)
            .map(|h| h.keys().copied().collect())
            .unwrap_or_default()
    }

    fn enumerate_size(&self, id: Id, size: usize) -> Vec<RecExpr<OriginLang<L>>> {
        self.enumerate_with_mode(id, size, Mode::Novel)
    }

    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let sample = self.sample_with_mode(id, size, Mode::Novel, rng);
        debug_assert!(
            self.novel
                .prev()
                .lookup_expr(&lower(sample.clone()))
                .is_none(),
            "SOMEHOW THE NOVEL SAMPLER PRODUCED A NON-NOVEL TERM"
        );
        sample
    }
}

// ============================================================================
// Helpers.
// ============================================================================

/// Cartesian product over `slot_options` returning every combination as a
/// vector. `slot_options[i]` lists the choices available for slot `i`.
fn enumerate_profiles<T: Clone>(slot_options: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut combos = vec![Vec::new()];
    for slot in slot_options {
        let mut next = Vec::with_capacity(combos.len() * slot.len());
        for prefix in &combos {
            for opt in slot {
                let mut p = prefix.clone();
                p.push(opt.clone());
                next.push(p);
            }
        }
        combos = next;
    }
    combos
}

/// True iff the profile `(a_1, …, a_k)` exactly matches some
/// `match.prev_children` — i.e., the term would be extractable from prev's
/// `match.prev_class`.
fn completes_some_match(profile: &[Option<Id>], matches: &[NodeMatch]) -> bool {
    matches.iter().any(|m| {
        profile.len() == m.prev_children.len()
            && profile
                .iter()
                .zip(m.prev_children.iter())
                .all(|(slot, &pc)| matches!(slot, Some(p) if *p == pc))
    })
}

/// Convolve histograms and read out the count at exactly `budget`. Returns
/// `None` if any histogram is empty (the convolution is then empty too) or
/// if the budget has no entry.
fn convolve_at<C: Counter>(histograms: &[&HashMap<usize, C>], budget: usize) -> Option<C> {
    if histograms.iter().any(|h| h.is_empty()) {
        return None;
    }
    let conv = PlainTermCount::<C>::convolve(histograms, budget);
    conv.get(&budget).cloned()
}

// ============================================================================
// Tests.
// ============================================================================

#[cfg(test)]
mod tests {
    use egg::EGraph;
    use num::BigUint;

    use super::*;
    use crate::count::PlainTermCount;
    use crate::langs::math::Math;
    use crate::lower;
    use crate::utils::combined_rng;

    fn sym(name: &str) -> Math {
        Math::Symbol(name.into())
    }

    #[test]
    fn novel_sample_picks_only_novel_term() {
        // prev: a, b, ln(a) (no union).
        // curr: same plus union(a, b). Now ln(b) is extractable from curr's
        // root but not from any prev class.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let root = curr.add(Math::Ln(a));
        curr.rebuild();
        let prev = curr.clone();
        let _ = b;

        curr.union(a, b);
        curr.rebuild();

        let plain = PlainTermCount::<BigUint>::new(5, &curr);
        let novel = NovelTermCount::new(5, &curr, &prev, plain);

        let sampler = NovelSampler::new(&novel, root, super::super::CountWeigher);

        for s in 0..50_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(sampler.sample(root, 2, &mut rng)).to_string();
            assert_eq!(term, "(ln b)", "got non-novel sample: {term}");
        }
    }

    #[test]
    fn novel_sample_union_diagonal() {
        // prev: Add(a, b)
        // curr: same plus union(a, b). Add(merged, merged) extracts 4 terms;
        // only Add(a, b) is in prev. Sampler should never produce "(+ a b)".
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

        let sampler = NovelSampler::new(&novel, root, super::super::CountWeigher);

        for s in 0..100_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(sampler.sample(root, 3, &mut rng)).to_string();
            assert_ne!(term, "(+ a b)", "produced non-novel term");
            assert!(
                ["(+ a a)", "(+ b a)", "(+ b b)"].contains(&term.as_str()),
                "unexpected term: {term}"
            );
        }
    }

    #[test]
    fn novel_possible_size_excludes_old_terms() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        graph.rebuild();

        let plain = PlainTermCount::<BigUint>::new(5, &graph);
        let novel = NovelTermCount::new(5, &graph, &graph, plain);

        let sampler = NovelSampler::new(&novel, a, super::super::CountWeigher);
        // Nothing is novel, so possible_size should be false everywhere.
        assert!(!sampler.possible_size(a, 1, 0));
    }
}
