use egg::{EGraph, Id, RecExpr};
use hashbrown::HashMap;
use num::BigUint;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use smallvec::SmallVec;

use crate::candidates::count::{
    Histograms, RootBudgets, count_histograms_rooted, find_plain_root_size, root_budgets,
};
use crate::candidates::draw::{
    CountWeigher, Drawer, DrawerPackage, DrawingError, UniformWeigher, Weigher,
};
use crate::candidates::{convolve_at, greedy_distribute_alloc, suffix_convolutions};
use crate::cli::Policy;
use crate::eqsat::EqsatResult;
use crate::{MyAnalysis, MyLanguage, OriginLang, stack_children};

pub struct PlainDrawer<'a, 'b, L: MyLanguage, N: MyAnalysis<L>, W: Weigher> {
    counts: &'a Histograms<Id, BigUint>,
    graph: &'b EGraph<L, N>,
    root: Id,
    weigher: W,
}

impl<'a, 'b, L: MyLanguage, N: MyAnalysis<L>, W: Weigher> PlainDrawer<'a, 'b, L, N, W> {
    #[must_use]
    pub const fn new(
        counts: &'a Histograms<Id, BigUint>,
        graph: &'b EGraph<L, N>,
        root: Id,
        weigher: W,
    ) -> Self {
        Self {
            counts,
            graph,
            root,
            weigher,
        }
    }

    /// The counted histograms of a nodes children or `None` if any
    /// child were not counted => 0
    fn child_hists(&self, node: &L) -> Option<SmallVec<[&HashMap<usize, BigUint>; 2]>> {
        node.children()
            .iter()
            .map(|&c| self.counts.get(&self.graph.find(c)))
            .collect()
    }
}

impl<L: MyLanguage, N: MyAnalysis<L>, W: Weigher> Drawer<L, N> for PlainDrawer<'_, '_, L, N, W> {
    fn root(&self) -> Id {
        self.root
    }

    fn find(&self, id: Id) -> Id {
        self.graph.find(id)
    }

    fn size_histogram(&self, id: Id) -> Option<&HashMap<usize, BigUint>> {
        self.counts.get(&id)
    }

    fn draw(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let canon_id = self.graph.find(id);
        let eclass = &self.graph[canon_id];
        let child_budget = size - 1;

        let weights = eclass
            .nodes
            .iter()
            .map(|node| {
                self.child_hists(node)
                    .and_then(|hists| convolve_at(&hists, child_budget))
                    .map_or_else(|| BigUint::ZERO, |count| self.weigher.node_weight(&count))
            })
            .collect::<Vec<_>>();
        let pick_idx = WeightedIndex::new(&weights).unwrap().sample(rng);

        let pick = &eclass.nodes[pick_idx];
        // The pick has positive weight, so `convolve_at` saw every child.
        let hists = self
            .child_hists(pick)
            .expect("a weighted node has counted children");
        let suffix = suffix_convolutions(&hists, child_budget);

        let mut remaining = child_budget;
        let children = pick
            .children()
            .iter()
            .enumerate()
            .map(|(i, &c_id)| {
                let candidates = hists[i]
                    .iter()
                    .filter_map(|(&s, count)| {
                        let rest = remaining.checked_sub(s)?;
                        let rest_count = suffix[i + 1].get(&rest)?;
                        Some((s, self.weigher.child_weight(count, rest_count)))
                    })
                    .collect::<Vec<_>>();

                let dist = WeightedIndex::new(candidates.iter().map(|(_, w)| w)).unwrap();
                let chosen_size = candidates[dist.sample(rng)].0;
                remaining -= chosen_size;
                self.draw(c_id, chosen_size, rng)
            })
            .collect::<Vec<_>>();

        stack_children(&children, OriginLang::new(pick.clone(), canon_id))
    }
}

/// Final e-graph and complete count tables for whole-graph candidate
/// construction.
///
/// Construction consumes [`EqsatResult`] and discards its run metadata.
pub struct PlainPackage<L: MyLanguage, N: MyAnalysis<L>> {
    egraph: EGraph<L, N>,
    counts: Histograms<Id, BigUint>,
    min_size: usize,
    max_size: usize,
    root: Id,
}

impl<L: MyLanguage, N: MyAnalysis<L>> PlainPackage<L, N> {
    /// Build counts through `max_size` over the whole e-graph.
    /// Returns `None` if the root has no terms within `max_size`.
    #[must_use]
    pub fn build(result: EqsatResult<L, N>, max_size: usize) -> Option<PlainPackage<L, N>> {
        let curr = result.curr();
        let root = curr.find(result.root());
        let budgets = root_budgets(curr, root, max_size);

        Self::from_root_budget(result, max_size, &budgets)
    }

    fn from_root_budget(
        result: EqsatResult<L, N>,
        max_size: usize,
        budgets: &RootBudgets,
    ) -> Option<PlainPackage<L, N>> {
        let (egraph, root) = result.into_curr();
        let counts = count_histograms_rooted(&egraph, budgets);

        let root = egraph.find(root);
        let histogram = counts.get(&root)?;

        let min_size = histogram.keys().min().copied().unwrap_or(1);
        Some(PlainPackage {
            egraph,
            counts,
            min_size,
            max_size,
            root,
        })
    }

    /// Build a package ending at the smallest root size that makes at least
    /// `min_extractable` terms available.
    ///
    /// The exact scan stops at the cap `start_size + search_steps`.
    /// Unlike the frontier package there is no previous boundary to subtract,
    /// so every term the root can extract counts toward the threshold.
    /// See `docs/counting/novel_size_search.md`.
    ///
    /// # Errors
    ///
    /// Returns the cap if the scan or package finds too few terms.
    pub fn build_through_sizes(
        result: EqsatResult<L, N>,
        start_size: usize,
        search_steps: usize,
        min_extractable: usize,
    ) -> Result<(usize, Self), usize> {
        let cap = start_size + search_steps;

        let curr = result.curr();
        let root = curr.find(result.root());
        let cap_budgets = root_budgets(curr, root, cap);

        let max_size = match find_plain_root_size(curr, root, min_extractable, &cap_budgets) {
            Ok(max_size) => max_size,
            Err(term_count) => {
                eprintln!(
                    "found insufficient ({term_count}) extractable terms with cap={cap} when {min_extractable} were expected"
                );
                return Err(cap);
            }
        };

        let final_budgets = root_budgets(curr, root, max_size);

        let Some(package) = Self::from_root_budget(result, max_size, &final_budgets) else {
            eprintln!("package construction found no terms (max_size={max_size})");
            return Err(cap);
        };

        Ok((max_size, package))
    }

    #[must_use]
    pub const fn root(&self) -> Id {
        self.root
    }
}

impl<L: MyLanguage, N: MyAnalysis<L>> DrawerPackage<L, N> for PlainPackage<L, N> {
    /// Root-term counts by size.
    ///
    /// # Panics
    ///
    /// Panics if package construction violated its root-histogram invariant.
    fn root_histogram(&self) -> &HashMap<usize, BigUint> {
        self.counts
            .get(&self.root)
            .expect("root histogram present iff build returned Some")
    }

    /// Draw exact root candidates from the whole e-graph.
    fn draw_candidates(
        &self,
        count: usize,
        policy: Policy,
        seed: [u64; 2],
    ) -> Result<Vec<RecExpr<OriginLang<L>>>, DrawingError> {
        let histogram = self.root_histogram();

        let requests = greedy_distribute_alloc(self.min_size, self.max_size, count, histogram);

        match policy {
            Policy::Uniform => {
                PlainDrawer::new(&self.counts, &self.egraph, self.root, UniformWeigher)
                    .draw_root_batch(&requests, seed)
            }
            Policy::Count => PlainDrawer::new(&self.counts, &self.egraph, self.root, CountWeigher)
                .draw_root_batch(&requests, seed),
            Policy::Smallest => Ok(vec![
                PlainDrawer::new(&self.counts, &self.egraph, self.root, UniformWeigher)
                    .smallest_root(),
            ]),
        }
    }

    fn root(&self) -> Id {
        self.root
    }
}

#[cfg(test)]
mod tests {
    use egg::EGraph;

    use super::*;
    use crate::candidates::count::{count_histograms_rooted, root_budgets};
    use crate::candidates::draw::{CountWeigher, UniformWeigher};
    use crate::langs::math::Math;
    use crate::lower;
    use crate::utils::combined_rng;
    use crate::utils::sym;

    fn rooted_counts(
        max_size: usize,
        graph: &EGraph<Math, ()>,
        root: Id,
    ) -> Histograms<Id, BigUint> {
        let budgets = root_budgets(graph, root, max_size);
        count_histograms_rooted(graph, &budgets)
    }

    #[test]
    fn build_through_sizes_stops_at_min_extractable() {
        // Unioning `a` with the root of (+ a b) creates a cycle: the root
        // class extracts a, (+ a b), (+ (+ a b) b), ... — one term each at
        // sizes 1, 3, 5, ... The naive package ignores prev entirely, so a
        // threshold of 3 terms must yield max_size = 5 (where the frontier
        // would yield 9).
        let mut curr = EGraph::<Math, ()>::new(());
        curr.enable_union_event_recording();
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let apb = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev_raw_node_count = curr.nodes().len();
        let prev_union_event_count = curr.union_event_count();

        curr.union(a, apb);
        curr.rebuild();

        let result =
            EqsatResult::new_for_tests(curr, apb, prev_raw_node_count, prev_union_event_count);
        let (used_max_size, package) = PlainPackage::build_through_sizes(result, 30, 2, 3)
            .expect("build_through_sizes should succeed");

        assert_eq!(used_max_size, 5);
        assert_eq!(package.max_size, 5);
        assert_eq!(package.min_size, 1);
        let mut keys = package.root_histogram().keys().copied().collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, vec![1, 3, 5]);
    }

    #[test]
    fn build_through_sizes_reports_cap_when_short() {
        // A single leaf has exactly one extractable term, so a threshold of
        // 3 can never be met and the scan must report the cap.
        let mut graph = EGraph::<Math, ()>::new(());
        graph.enable_union_event_recording();
        let root = graph.add(sym("a"));
        graph.rebuild();
        let prev_raw_node_count = graph.nodes().len();
        let prev_union_event_count = graph.union_event_count();

        let result =
            EqsatResult::new_for_tests(graph, root, prev_raw_node_count, prev_union_event_count);
        let Err(cap) = PlainPackage::build_through_sizes(result, 3, 20, 3) else {
            panic!("a single term cannot satisfy a threshold of 3");
        };

        assert_eq!(cap, 23, "cap = start_size + search_steps");
    }

    #[test]
    fn naive_draw_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let counts = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&counts, &graph, root, UniformWeigher);

        let mut rng = combined_rng([42]);
        let term = drawer.draw(root, 1, &mut rng);
        assert_eq!(lower(term).to_string(), "a");
    }

    #[test]
    fn naive_draw_picks_valid_choice() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let counts = rooted_counts(10, &graph, a);
        let drawer = PlainDrawer::new(&counts, &graph, a, UniformWeigher);

        for s in 0..50_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(drawer.draw(a, 1, &mut rng)).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn naive_possible_size_correct() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let counts = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&counts, &graph, root, UniformWeigher);

        assert!(!drawer.possible_size(root, 1, 0));
        assert!(!drawer.possible_size(root, 3, 0));
        assert!(drawer.possible_size(root, 2, 0));
        assert!(!drawer.possible_size(root, 2, 1));
    }

    #[test]
    fn naive_draw_batch_finds_all_unique() {
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

        let counts = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&counts, &graph, root, UniformWeigher);

        let result = drawer.draw_root_batch(&[(3, 5)], [1, 2]).unwrap();
        assert!(result.len() <= 6);
    }

    #[test]
    fn count_weighted_draw_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(sym("a"));
        graph.rebuild();

        let counts = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&counts, &graph, root, CountWeigher);

        let mut rng = combined_rng([42]);
        let term = drawer.draw(root, 1, &mut rng);
        assert_eq!(lower(term).to_string(), "a");
    }

    #[test]
    fn count_weighted_draw_picks_valid_choice() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let counts = rooted_counts(10, &graph, a);
        let drawer = PlainDrawer::new(&counts, &graph, a, CountWeigher);

        for s in 0..50_u64 {
            let mut rng = combined_rng([s]);
            let term = lower(drawer.draw(a, 1, &mut rng)).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn count_draw_batch_finds_unique() {
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

        let counts = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&counts, &graph, root, CountWeigher);

        let result = drawer.draw_root_batch(&[(3, 5)], [1, 2]).unwrap();

        assert!(result.len() <= 6);
    }

    #[test]
    fn draw_batch_errors_when_size_undersupplied() {
        // Root Add([a-class, b-class]) has 2 * 3 = 6 distinct terms of size 3.
        // A batch is all-or-nothing: exactly 6 is satisfiable, and asking for
        // more than the size holds fails instead of returning a short draw.
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

        let counts = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&counts, &graph, root, CountWeigher);

        let exact = drawer
            .draw_root_batch(&[(3, 6)], [1, 2])
            .expect("the size holds exactly the requested 6 terms");
        assert_eq!(exact.len(), 6, "should return exactly the 6 distinct terms");

        assert!(
            matches!(
                drawer.draw_root_batch(&[(3, 1000)], [1, 2]),
                Err(DrawingError::InsufficientTerms)
            ),
            "a size that cannot supply the request fails the batch"
        );
    }

    #[test]
    fn draw_batch_errors_when_any_size_is_short() {
        // Root Ln(a) has exactly one term of size 2 and nothing at size 5.
        // The satisfiable size alone succeeds, but pairing it with an empty
        // size fails the whole batch rather than silently dropping it.
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let counts = rooted_counts(10, &graph, root);
        let drawer = PlainDrawer::new(&counts, &graph, root, UniformWeigher);

        let satisfiable = drawer
            .draw_root_batch(&[(2, 1)], [1, 2])
            .expect("the lone size-2 term satisfies a request for one");
        assert_eq!(satisfiable.len(), 1);

        assert!(
            matches!(
                drawer.draw_root_batch(&[(2, 1), (5, 5)], [1, 2]),
                Err(DrawingError::InsufficientTerms)
            ),
            "an empty size poisons an otherwise satisfiable batch"
        );

        assert!(
            matches!(
                drawer.draw_root_batch(&[(5, 5)], [1, 2]),
                Err(DrawingError::InsufficientTerms)
            ),
            "a wholly empty request returns an Error"
        );
    }
}
