//! Term counting analysis for e-graphs.
//!
//! Counts the number of terms up to a given size that can be extracted from each e-class.

#[cfg(test)]
use egg::RecExpr;
use egg::{Analysis, EGraph, Id, Language};
use hashbrown::HashMap;
#[cfg(test)]
use indicatif::{ProgressBar, ProgressIterator};

use crate::Counter;
use crate::sampling::count::{CountData, count_terms};
use crate::{MyAnalysis, MyLanguage};
#[cfg(test)]
use crate::{OriginLang, stack_children};

/// Map from e-class ID to a map of (size -> count) (histogram).
#[derive(Debug, Clone)]
pub struct PlainTermCount<C: Counter> {
    data: HashMap<Id, HashMap<usize, C>>,
    /// Per e-class, per node index: precomputed suffix convolution tables.
    /// `suffix_cache[eclass][node_idx][i]` = convolution of children `i..n`,
    /// mapping budget -> count.
    suffix_cache: HashMap<Id, Vec<Vec<HashMap<usize, C>>>>,
}

impl<C: Counter> PlainTermCount<C> {
    /// Run the term counting analysis restricted to what extractions of size
    /// <= `max_size` from `roots` can reach: classes unreachable from the
    /// roots are absent, and every other class's histogram is capped at the
    /// largest subterm size it can take in such an extraction. Root-driven
    /// sampling and histogram queries are unaffected by the restriction;
    /// direct queries against deeper classes see the capped data.
    #[must_use]
    pub fn rooted<L, N>(max_size: usize, graph: &EGraph<L, N>, roots: &[Id]) -> Self
    where
        L: Language,
        N: Analysis<L>,
    {
        Self::from_counts(count_terms(max_size, graph, Some(roots)))
    }

    fn from_counts(counts: CountData<C>) -> Self {
        Self {
            data: counts.data,
            suffix_cache: counts.suffix,
        }
    }

    /// Get the histogram for a child (size -> count). `None` means the child
    /// has no extractions — callers should treat it as an empty histogram.
    pub(crate) fn child_histogram<L: MyLanguage, N: MyAnalysis<L>>(
        &self,
        child_id: Id,
        graph: &EGraph<L, N>,
    ) -> Option<&HashMap<usize, C>> {
        self.data.get(&graph.find(child_id))
    }

    #[must_use]
    pub const fn data(&self) -> &HashMap<Id, HashMap<usize, C>> {
        &self.data
    }

    #[must_use]
    pub const fn suffix_cache(&self) -> &HashMap<Id, Vec<Vec<HashMap<usize, C>>>> {
        &self.suffix_cache
    }
}

/// Test-only helpers: the unrestricted constructor and exhaustive term
/// enumeration, used to cross-check the counting DP.
#[cfg(test)]
impl<C: Counter> PlainTermCount<C> {
    /// Run the term counting analysis on an e-graph, counting every class.
    ///
    /// # Arguments
    /// * `max_size` - Maximum term size to count
    #[must_use]
    pub fn new<L, N>(max_size: usize, graph: &EGraph<L, N>) -> Self
    where
        L: Language,
        N: Analysis<L>,
    {
        Self::from_counts(count_terms(max_size, graph, None))
    }

    /// Enumerate all terms from an e-class with sizes in `1..=max_size`.
    #[must_use]
    pub fn enumerate<L: MyLanguage, N: MyAnalysis<L>>(
        &self,
        graph: &EGraph<L, N>,
        id: Id,
        max_size: usize,
        progress: Option<ProgressBar>,
    ) -> Vec<RecExpr<OriginLang<L>>> {
        let canon_id = graph.find(id);
        let Some(histogram) = self.data.get(&canon_id) else {
            return Vec::new();
        };
        let sum = histogram.values().sum::<C>().to_u64().unwrap();

        let mut cache = HashMap::new();
        let iter = (1..=max_size)
            .flat_map(|size| self.enumerate_class_inner(graph, canon_id, size, &mut cache));

        if let Some(pb) = progress {
            pb.set_length(sum);
            iter.progress_with(pb).collect()
        } else {
            iter.collect()
        }
    }

    /// Enumerate all terms of exactly `size` from an e-class, using a shared cache.
    fn enumerate_class_cached<L: MyLanguage, N: MyAnalysis<L>>(
        &self,
        graph: &EGraph<L, N>,
        id: Id,
        size: usize,
        cache: &mut HashMap<(Id, usize), Vec<RecExpr<OriginLang<L>>>>,
    ) -> Vec<RecExpr<OriginLang<L>>> {
        let canon_id = graph.find(id);
        let key = (canon_id, size);

        // Cache hit
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }

        let result = self.enumerate_class_inner(graph, canon_id, size, cache);
        cache.insert(key, result.clone());
        result
    }

    /// Inner logic for enumerating all terms of exactly `size` from a canonical e-class.
    fn enumerate_class_inner<L: MyLanguage, N: MyAnalysis<L>>(
        &self,
        graph: &EGraph<L, N>,
        canon_id: Id,
        size: usize,
        cache: &mut HashMap<(Id, usize), Vec<RecExpr<OriginLang<L>>>>,
    ) -> Vec<RecExpr<OriginLang<L>>> {
        // Check if this class has any terms at this size
        let Some(histogram) = self.data.get(&canon_id) else {
            return Vec::new();
        };
        if !histogram.contains_key(&size) {
            return Vec::new();
        }

        let eclass = &graph[canon_id];
        // let type_overhead = self.type_overhead(&canon_id);

        // Bail if type size overhead is too big
        let Some(child_budget) = size.checked_sub(1) else {
            //+ type_overhead) else {
            return Vec::new();
        };

        // let ty = OriginTree::from_eclass(graph, canon_id);

        let mut results = Vec::new();

        for node in &eclass.nodes {
            let children = node.children();

            for child_combo in self.enumerate_children(graph, children, child_budget, cache) {
                results.push(stack_children(
                    &child_combo,
                    OriginLang::new(node.clone(), canon_id),
                ));
            }
        }

        results
    }

    /// Enumerate all ways to fill `children` with exactly `budget` total size,
    /// returning the cartesian product of child terms for each valid size tuple.
    fn enumerate_children<L: MyLanguage, N: MyAnalysis<L>>(
        &self,
        graph: &EGraph<L, N>,
        children: &[Id],
        budget: usize,
        cache: &mut HashMap<(Id, usize), Vec<RecExpr<OriginLang<L>>>>,
    ) -> impl Iterator<Item = Vec<RecExpr<OriginLang<L>>>> + use<C, L, N> {
        // Accumulate via left-fold: start with the empty tuple at budget=`budget`,
        // then for each child, expand every (remaining_budget, partial_combo) by
        // enumerating that child at each feasible size.

        let mut acc = vec![(budget, Vec::new())];

        for &child_id in children {
            let next_acc = acc
                .into_iter()
                .flat_map(|(remaining, partial)| {
                    self.expand_child(graph, child_id, remaining, &partial, cache)
                })
                .collect();

            acc = next_acc;
        }

        // Only keep combos that used the entire budget
        acc.into_iter()
            .filter(|(remaining, _)| *remaining == 0)
            .map(|(_, combo)| combo)
    }

    /// Expand a single partial combo by one child, returning all valid extensions.
    fn expand_child<L: MyLanguage, N: MyAnalysis<L>>(
        &self,
        graph: &EGraph<L, N>,
        child_id: Id,
        remaining: usize,
        partial: &[RecExpr<OriginLang<L>>],
        cache: &mut HashMap<(Id, usize), Vec<RecExpr<OriginLang<L>>>>,
    ) -> Vec<(usize, Vec<RecExpr<OriginLang<L>>>)> {
        let canonical_child = graph.find(child_id);
        let Some(child_histogram) = self.data.get(&canonical_child) else {
            return Vec::new();
        };

        let mut results = Vec::new();
        for (&child_size, _) in child_histogram {
            if child_size > remaining {
                continue;
            }
            let child_exprs =
                self.enumerate_class_cached(graph, canonical_child, child_size, cache);
            for expr in child_exprs {
                let mut combo = partial.to_vec();
                combo.push(expr.clone());
                results.push((remaining - child_size, combo));
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, Language};
    use hashbrown::HashSet;
    use num::BigUint;

    use super::*;
    use crate::langs::math::Math;
    use crate::test_utils::sym;

    #[test]
    fn enumerate_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let terms = tc.enumerate(&graph, root, 10, None);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0][terms[0].root()].inner(), &sym("a"));
    }

    #[test]
    fn enumerate_two_leaves() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let terms = tc.enumerate(&graph, a, 10, None);
        assert_eq!(terms.len(), 2);
        let labels: HashSet<_> = terms.iter().map(|t| t[t.root()].inner().clone()).collect();
        assert!(labels.contains(&sym("a")));
        assert!(labels.contains(&sym("b")));
    }

    #[test]
    fn enumerate_parent_child() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let terms = tc.enumerate(&graph, root, 10, None);
        assert_eq!(terms.len(), 1);
        let term = &terms[0];
        let root_node = &term[term.root()];
        assert!(matches!(root_node.inner(), Math::Ln(_)));
        let child_id = root_node.children()[0];
        assert_eq!(term[child_id].inner(), &sym("a"));
    }

    #[test]
    fn enumerate_combinatorial() {
        // root: (+ left right)
        // left:  "a1", "a2"
        // right: "b1", "b2", "b3"
        let mut graph = EGraph::<Math, ()>::new(());
        let a1 = graph.add(sym("a1"));
        let a2 = graph.add(sym("a2"));
        graph.union(a1, a2);

        let b1 = graph.add(sym("b1"));
        let b2 = graph.add(sym("b2"));
        let b3 = graph.add(sym("b3"));
        graph.union(b1, b2);
        graph.union(b1, b3);

        let root = graph.add(Math::Add([a1, b1]));
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        let terms = tc.enumerate(&graph, root, 10, None);
        // 2 * 3 = 6 combinations
        assert_eq!(terms.len(), 6);
    }

    #[test]
    fn enumerate_respects_max_size() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let tc = PlainTermCount::<BigUint>::new(10, &graph);
        // max_size=1 should not include ln(a) which is size 2
        let terms = tc.enumerate(&graph, root, 1, None);
        assert_eq!(terms.len(), 0);
    }

    #[test]
    fn enumerate_count_matches_term_count() {
        // root: (+ child child)  -- same child twice
        // child: "a", "b", ln(c)
        // c: "c"
        let mut graph = EGraph::<Math, ()>::new(());
        let c = graph.add(sym("c"));
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        let ln_c = graph.add(Math::Ln(c));
        graph.union(a, b);
        graph.union(a, ln_c);

        let root = graph.add(Math::Add([a, a]));
        graph.rebuild();

        let max_size = 10;
        let tc = PlainTermCount::<BigUint>::new(max_size, &graph);

        let terms = tc.enumerate(&graph, root, max_size, None);
        let expected_total: BigUint = tc
            .data
            .get(&graph.find(root))
            .unwrap()
            .iter()
            .filter(|&(s, _)| *s <= max_size)
            .map(|(_, count)| count.clone())
            .sum();
        assert_eq!(BigUint::from(terms.len()), expected_total);
    }

    #[test]
    fn single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let term_count = PlainTermCount::<BigUint>::new(10, &graph);

        let root_data = &term_count.data[&graph.find(root)];
        assert_eq!(root_data.len(), 1);
        assert_eq!(root_data[&1], BigUint::from(1u32));
    }

    #[test]
    fn two_choices() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let term_count = PlainTermCount::<BigUint>::new(10, &graph);

        let root_data = &term_count.data[&graph.find(a)];
        assert_eq!(root_data[&1], BigUint::from(2u32));
    }

    #[test]
    fn parent_child() {
        // Class 0: ln(class 1)
        // Class 1: leaf "a"
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let term_count = PlainTermCount::<BigUint>::new(10, &graph);

        // Class a: one term of size 1
        assert_eq!(term_count.data[&graph.find(a)][&1], BigUint::from(1u32));

        // Class root: one term of size 2 (ln + a)
        assert_eq!(term_count.data[&graph.find(root)][&2], BigUint::from(1u32));
    }

    #[test]
    fn parent_with_multiple_child_choices() {
        // root: ln(child)
        // child: two leaves "a" and "b"
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let term_count = PlainTermCount::<BigUint>::new(10, &graph);

        // child: two terms of size 1
        assert_eq!(term_count.data[&graph.find(a)][&1], BigUint::from(2u32));

        // root: two terms of size 2 (ln(a), ln(b))
        assert_eq!(term_count.data[&graph.find(root)][&2], BigUint::from(2u32));
    }

    #[test]
    fn two_children() {
        // root: (+ a b)
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        let root = graph.add(Math::Add([a, b]));
        graph.rebuild();

        let term_count = PlainTermCount::<BigUint>::new(10, &graph);

        // root: one term of size 3 (+ + a + b)
        assert_eq!(term_count.data[&graph.find(root)][&3], BigUint::from(1u32));
    }

    #[test]
    fn combinatorial_explosion() {
        // root: (+ left right)
        // left:  two leaves "a1", "a2"
        // right: three leaves "b1", "b2", "b3"
        let mut graph = EGraph::<Math, ()>::new(());
        let a1 = graph.add(sym("a1"));
        let a2 = graph.add(sym("a2"));
        graph.union(a1, a2);

        let b1 = graph.add(sym("b1"));
        let b2 = graph.add(sym("b2"));
        let b3 = graph.add(sym("b3"));
        graph.union(b1, b2);
        graph.union(b1, b3);

        let root = graph.add(Math::Add([a1, b1]));
        graph.rebuild();

        let term_count = PlainTermCount::<BigUint>::new(10, &graph);

        // root: 2 * 3 = 6 terms of size 3
        assert_eq!(term_count.data[&graph.find(root)][&3], BigUint::from(6u32));
    }

    #[test]
    fn max_size_filters() {
        // root: ln(a)
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        // max_size = 1, so ln(a) with size 2 should be filtered out
        let term_count = PlainTermCount::<BigUint>::new(1, &graph);

        // a should have data (size 1)
        assert!(term_count.data.contains_key(&graph.find(a)));
        assert_eq!(term_count.data[&graph.find(a)][&1], BigUint::from(1u32));

        // root should be empty (size 2 exceeds max_size)
        assert!(
            term_count
                .data
                .get(&graph.find(root))
                .is_none_or(|d| d.is_empty())
        );
    }
}
