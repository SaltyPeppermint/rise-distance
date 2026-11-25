use egg::{Analysis, EGraph, Id, Language, RecExpr};
use num::BigUint;
use rand_chacha::ChaCha12Rng;
use smallvec::SmallVec;

use crate::cli::Policy;
use crate::eqsat::EqsatResult;
use crate::origin::OriginLang;
use crate::sampling::{self, INLINE_ARITY};
use crate::sampling::count::whole;
use crate::sampling::draw::{self, SamplePackage, DrawCursor, Drawer, DrawingError, SampledTrie};
use crate::sampling::weigher::{Choice, Weigher};
use crate::utils::{self, HashMap};

struct WholeDrawer<'a, 'b, L: Language, N: Analysis<L>> {
    /// Histogram of the counts for each class, indexed by `node_idx`
    counts: &'a HashMap<Id, HashMap<usize, BigUint>>,
    graph: &'b EGraph<L, N>,
    root: Id,
    weigher: Weigher,
}

impl<'a, 'b, L: Language, N: Analysis<L>> WholeDrawer<'a, 'b, L, N> {
    fn new(
        counts: &'a HashMap<Id, HashMap<usize, BigUint>>,
        graph: &'b EGraph<L, N>,
        root: Id,
        weigher: Weigher,
    ) -> Self {
        Self {
            counts,
            graph,
            root: graph.find(root),
            weigher,
        }
    }

    /// The counted histograms of a nodes children or `None` if any
    /// child were not counted => 0
    fn child_hists(&self, node: &L) -> Option<SmallVec<[&HashMap<usize, BigUint>; INLINE_ARITY]>> {
        node.children()
            .iter()
            .map(|&c| self.counts.get(&self.graph.find(c)))
            .collect()
    }

    fn draw_rec(
        &self,
        id: Id,
        size: usize,
        rest_total: Option<&BigUint>,
        cursor: &mut DrawCursor<'_>,
    ) -> RecExpr<OriginLang<L>> {
        let canon_id = self.graph.find(id);
        let eclass = &self.graph[canon_id];
        let child_budget = size - 1;
        let active_rest = cursor.active_rest(rest_total);

        let feasible_nodes = eclass
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(node_idx, node)| {
                let hists = self.child_hists(node)?;
                let count = sampling::convolve_at(&hists, child_budget)?;
                Some((node_idx, count, hists))
            })
            .collect::<Vec<_>>();
        let choices = feasible_nodes
            .iter()
            .map(|(node_idx, count, _)| Choice::branch(*node_idx, count))
            .collect::<Vec<_>>();
        let (pick_idx, _, hists) =
            &feasible_nodes[self.weigher.pick(&choices, active_rest, cursor)];

        cursor.step(*pick_idx);
        let pick = &eclass.nodes[*pick_idx];
        let children_ids = pick.children();
        let children = draw::draw_children(
            self.weigher,
            hists,
            child_budget,
            active_rest,
            cursor,
            |i, child_size, child_rest, cur| {
                self.draw_rec(children_ids[i], child_size, child_rest, cur)
            },
        );

        utils::stack_children(&children, OriginLang::new(pick.clone(), canon_id))
    }
}

impl<L: Language, N: Analysis<L>> Drawer<L> for WholeDrawer<'_, '_, L, N> {
    fn root_histogram(&self) -> Option<&HashMap<usize, BigUint>> {
        self.counts.get(&self.root)
    }

    fn draw_next(
        &self,
        size: usize,
        sampled: &mut SampledTrie,
        rng: &mut ChaCha12Rng,
    ) -> RecExpr<OriginLang<L>> {
        let mut cursor = DrawCursor::new(sampled, rng);
        self.draw_rec(self.root, size, Some(&BigUint::ONE), &mut cursor)
    }
}

/// Final e-graph and complete count tables for whole-graph sample
/// construction.
///
/// Construction consumes [`EqsatResult`] and discards its run metadata.
pub struct WholePackage<L: Language, N: Analysis<L>> {
    egraph: EGraph<L, N>,
    counts: HashMap<Id, HashMap<usize, BigUint>>,
    root: Id,
}

impl<L: Language, N: Analysis<L>> WholePackage<L, N> {
    fn from_counts(
        egraph: EGraph<L, N>,
        root: Id,
        counts: HashMap<Id, HashMap<usize, BigUint>>,
    ) -> Option<WholePackage<L, N>> {
        counts.get(&root)?;
        Some(WholePackage {
            egraph,
            counts,
            root,
        })
    }

    /// Build a package ending at the smallest root size that makes at least
    /// `min_extractable` terms available.
    ///
    /// The exact scan grows budgets geometrically starting at
    /// `min_size(root) + search_step` and stops at the cap
    /// `start_size + search_steps`. Unlike the frontier package there is no
    /// previous boundary to subtract, so every term the root can extract
    /// counts toward the threshold.
    ///
    /// # Errors
    ///
    /// Returns the cap if the scan or package finds too few terms.
    pub fn build(
        result: EqsatResult<L, N>,
        start_size: usize,
        search_step: usize,
        search_steps: usize,
        min_extractable: usize,
    ) -> Result<(usize, Self), usize> {
        let cap = start_size + search_steps;
        let (egraph, root) = result.into_curr();
        let root = egraph.find(root);

        let (max_size, counts) = match whole::count(
            &egraph,
            root,
            cap,
            search_step,
            min_extractable,
        ) {
            Ok(found) => found,
            Err(term_count) => {
                eprintln!(
                    "found insufficient ({term_count}) extractable terms with cap={cap} when {min_extractable} were expected"
                );
                return Err(cap);
            }
        };

        let Some(package) = Self::from_counts(egraph, root, counts) else {
            eprintln!("package construction found no terms (max_size={max_size})");
            return Err(cap);
        };

        Ok((max_size, package))
    }
}

impl<L: Language, N: Analysis<L>> SamplePackage<L> for WholePackage<L, N> {
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

    /// Draw exact root samples from the whole e-graph.
    fn draw_samples(
        &self,
        count: usize,
        policy: Policy,
        seed: [u64; 2],
    ) -> Result<Vec<RecExpr<OriginLang<L>>>, DrawingError> {
        WholeDrawer::new(&self.counts, &self.egraph, self.root, policy.into())
            .draw_for_policy(count, policy, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::langs::math::{self, Math};
    use crate::origin;
    use Weigher::{Count, Uniform};

    fn rooted_counts(
        cap: usize,
        min_extractable: usize,
        graph: &EGraph<Math, ()>,
        root: Id,
    ) -> HashMap<Id, HashMap<usize, BigUint>> {
        whole::count(graph, root, cap, 2, min_extractable)
            .expect("test graph has enough terms")
            .1
    }

    #[test]
    fn build_stops_at_min_extractable() {
        // Unioning `a` with the root of (+ a b) creates a cycle: the root
        // class extracts a, (+ a b), (+ (+ a b) b), ... — one term each at
        // sizes 1, 3, 5, ... The whole package ignores prev entirely, so a
        // threshold of 3 terms must yield max_size = 5 (where the frontier
        // would yield 9).
        let mut curr = EGraph::<Math, ()>::new(());
        curr.enable_union_event_recording();
        let a = curr.add(math::sym("a"));
        let b = curr.add(math::sym("b"));
        let apb = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev_raw_node_count = curr.nodes().len();
        let prev_union_event_count = curr.union_event_count();

        curr.union(a, apb);
        curr.rebuild();

        let result =
            EqsatResult::new_for_tests(curr, apb, prev_raw_node_count, prev_union_event_count);
        let (used_max_size, package) = WholePackage::build(result, 30, 2, 2, 3)
            .expect("build should succeed");

        assert_eq!(used_max_size, 5);
        let mut keys = package.root_histogram().keys().copied().collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(keys, vec![1, 3, 5]);
    }

    #[test]
    fn build_reports_cap_when_short() {
        // A single leaf has exactly one extractable term, so a threshold of
        // 3 can never be met and the scan must report the cap.
        let mut graph = EGraph::<Math, ()>::new(());
        graph.enable_union_event_recording();
        let root = graph.add(math::sym("a"));
        graph.rebuild();
        let prev_raw_node_count = graph.nodes().len();
        let prev_union_event_count = graph.union_event_count();

        let result =
            EqsatResult::new_for_tests(graph, root, prev_raw_node_count, prev_union_event_count);
        let Err(cap) = WholePackage::build(result, 3, 2, 20, 3) else {
            panic!("a single term cannot satisfy a threshold of 3");
        };

        assert_eq!(cap, 23, "cap = start_size + search_steps");
    }

    #[test]
    fn uniform_draw_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(math::sym("a"));
        graph.rebuild();

        let counts = rooted_counts(10, 1, &graph, root);
        let drawer = WholeDrawer::new(&counts, &graph, root, Uniform);

        let terms = drawer.draw_batch(&[(1, 1)], [42, 0]).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(origin::lower(terms[0].clone()).to_string(), "a");
    }

    #[test]
    fn uniform_draw_picks_valid_choice() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(math::sym("a"));
        let b = graph.add(math::sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let counts = rooted_counts(10, 1, &graph, a);
        let drawer = WholeDrawer::new(&counts, &graph, a, Uniform);

        for s in 0..50_u64 {
            let terms = drawer.draw_batch(&[(1, 1)], [s, 0]).unwrap();
            let term = origin::lower(terms[0].clone()).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn uniform_possible_size_correct() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(math::sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let counts = rooted_counts(10, 1, &graph, root);
        let drawer = WholeDrawer::new(&counts, &graph, root, Uniform);

        assert!(drawer.draw_batch(&[(1, 1)], [0, 0]).is_err());
        assert!(drawer.draw_batch(&[(3, 1)], [0, 0]).is_err());
        assert!(drawer.draw_batch(&[(2, 1)], [0, 0]).is_ok());
        assert!(drawer.draw_batch(&[(2, 2)], [0, 0]).is_err());
    }

    #[test]
    fn uniform_draw_batch_finds_all_unique() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a1 = graph.add(math::sym("a1"));
        let a2 = graph.add(math::sym("a2"));
        graph.union(a1, a2);
        let b1 = graph.add(math::sym("b1"));
        let b2 = graph.add(math::sym("b2"));
        let b3 = graph.add(math::sym("b3"));
        graph.union(b1, b2);
        graph.union(b1, b3);
        let root = graph.add(Math::Add([a1, b1]));
        graph.rebuild();

        let counts = rooted_counts(10, 1, &graph, root);
        let drawer = WholeDrawer::new(&counts, &graph, root, Uniform);

        let result = drawer.draw_batch(&[(3, 5)], [1, 2]).unwrap();
        assert!(result.len() <= 6);
    }

    #[test]
    fn count_weighted_draw_single_leaf() {
        let mut graph = EGraph::<Math, ()>::new(());
        let root = graph.add(math::sym("a"));
        graph.rebuild();

        let counts = rooted_counts(10, 1, &graph, root);
        let drawer = WholeDrawer::new(&counts, &graph, root, Count);

        let terms = drawer.draw_batch(&[(1, 1)], [42, 0]).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(origin::lower(terms[0].clone()).to_string(), "a");
    }

    #[test]
    fn count_weighted_draw_picks_valid_choice() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(math::sym("a"));
        let b = graph.add(math::sym("b"));
        graph.union(a, b);
        graph.rebuild();

        let counts = rooted_counts(10, 1, &graph, a);
        let drawer = WholeDrawer::new(&counts, &graph, a, Count);

        for s in 0..50_u64 {
            let terms = drawer.draw_batch(&[(1, 1)], [s, 0]).unwrap();
            let term = origin::lower(terms[0].clone()).to_string();
            assert!(term == "a" || term == "b", "got unexpected: {term}");
        }
    }

    #[test]
    fn count_draw_batch_finds_unique() {
        let mut graph = EGraph::<Math, ()>::new(());
        let a1 = graph.add(math::sym("a1"));
        let a2 = graph.add(math::sym("a2"));
        graph.union(a1, a2);
        let b1 = graph.add(math::sym("b1"));
        let b2 = graph.add(math::sym("b2"));
        let b3 = graph.add(math::sym("b3"));
        graph.union(b1, b2);
        graph.union(b1, b3);
        let root = graph.add(Math::Add([a1, b1]));
        graph.rebuild();

        let counts = rooted_counts(10, 1, &graph, root);
        let drawer = WholeDrawer::new(&counts, &graph, root, Count);

        let result = drawer.draw_batch(&[(3, 5)], [1, 2]).unwrap();

        assert!(result.len() <= 6);
    }

    #[test]
    fn draw_batch_errors_when_size_undersupplied() {
        // Root Add([a-class, b-class]) has 2 * 3 = 6 distinct terms of size 3.
        // A batch is all-or-nothing: exactly 6 is satisfiable, and asking for
        // more than the size holds fails instead of returning a short draw.
        let mut graph = EGraph::<Math, ()>::new(());
        let a1 = graph.add(math::sym("a1"));
        let a2 = graph.add(math::sym("a2"));
        graph.union(a1, a2);
        let b1 = graph.add(math::sym("b1"));
        let b2 = graph.add(math::sym("b2"));
        let b3 = graph.add(math::sym("b3"));
        graph.union(b1, b2);
        graph.union(b1, b3);
        let root = graph.add(Math::Add([a1, b1]));
        graph.rebuild();

        let counts = rooted_counts(10, 6, &graph, root);
        let drawer = WholeDrawer::new(&counts, &graph, root, Count);

        let exact = drawer
            .draw_batch(&[(3, 6)], [1, 2])
            .expect("the size holds exactly the requested 6 terms");
        assert_eq!(exact.len(), 6, "should return exactly the 6 distinct terms");

        assert!(
            matches!(
                drawer.draw_batch(&[(3, 1000)], [1, 2]),
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
        let a = graph.add(math::sym("a"));
        let root = graph.add(Math::Ln(a));
        graph.rebuild();

        let counts = rooted_counts(10, 1, &graph, root);
        let drawer = WholeDrawer::new(&counts, &graph, root, Uniform);

        let satisfiable = drawer
            .draw_batch(&[(2, 1)], [1, 2])
            .expect("the lone size-2 term satisfies a request for one");
        assert_eq!(satisfiable.len(), 1);

        assert!(
            matches!(
                drawer.draw_batch(&[(2, 1), (5, 5)], [1, 2]),
                Err(DrawingError::InsufficientTerms)
            ),
            "an empty size poisons an otherwise satisfiable batch"
        );

        assert!(
            matches!(
                drawer.draw_batch(&[(5, 5)], [1, 2]),
                Err(DrawingError::InsufficientTerms)
            ),
            "a wholly empty request returns an Error"
        );
    }

    #[test]
    fn skewed_uniform_draw_succeeds_without_rejection() {
        // Build a chain of 12 unary splits where each level has one 1-term
        // branch (`sin(leaf)`) and one continuation (`cos(next)`), ending in a
        // class with 16 leaves. Under local `Uniform`, any single term in the
        // deep 16-leaf branch has probability 2^-12 * 1/16 = 2^-16, so 32x
        // rejection sampling would fail to draw all 28 size-13 terms (or even
        // just 10 of them). Exact trie exclusion must return every term.
        let mut graph = EGraph::<Math, ()>::new(());
        let mut leaves = Vec::new();
        for i in 0..16 {
            leaves.push(graph.add(math::sym(&format!("x{i}"))));
        }
        let base = leaves[0];
        for &leaf in &leaves[1..] {
            graph.union(base, leaf);
        }

        // Pad `base` down a 12-level spine where each level also has a
        // single-term branch of matching size.
        let mut cur = base;
        let mut single_chain = graph.add(math::sym("s"));
        for _ in 0..12 {
            let wide = graph.add(Math::Cos(cur));
            let narrow = graph.add(Math::Sin(single_chain));
            graph.union(wide, narrow);
            cur = wide;
            single_chain = graph.add(Math::Ln(single_chain));
        }
        graph.rebuild();

        let target_size = 13;
        let counts = rooted_counts(target_size, 28, &graph, cur);
        let root = graph.find(cur);
        // 16 terms through the deep `cos^12` spine + 12 narrow `cos^k(sin(ln^...))`
        // branches = 28 total terms of size 13.
        assert_eq!(counts[&root][&target_size], BigUint::from(28u32));

        let drawer = WholeDrawer::new(&counts, &graph, root, Uniform);
        let first = drawer
            .draw_batch(&[(target_size, 28)], [7, 9])
            .expect("exhaustive uniform draw over skewed graph must succeed");
        let second = drawer
            .draw_batch(&[(target_size, 28)], [7, 9])
            .expect("repeated draw must succeed");

        assert_eq!(first.len(), 28);
        assert_eq!(first, second, "output order must be stable across runs");

        let lowered = first
            .into_iter()
            .map(|t| origin::lower(t).to_string())
            .collect::<crate::utils::HashSet<_>>();
        assert_eq!(lowered.len(), 28, "all 28 lowered terms must be distinct");
    }

    #[test]
    fn exhaustive_cyclic_draw_both_policies() {
        // `a == b == (+ a b)` has 2 terms of size 1, 4 of size 3, and 16 of size 5 (22 total).
        let mut graph = EGraph::<Math, ()>::new(());
        let a = graph.add(math::sym("a"));
        let b = graph.add(math::sym("b"));
        let apb = graph.add(Math::Add([a, b]));
        graph.union(a, apb);
        graph.union(b, apb);
        graph.rebuild();

        let root = graph.find(apb);
        let counts = rooted_counts(5, 22, &graph, root);
        assert_eq!(counts[&root][&5], BigUint::from(16u32));

        let count_terms = WholeDrawer::new(&counts, &graph, root, Count)
            .draw_batch(&[(5, 16)], [11, 13])
            .expect("count exhaustive cyclic draw");
        let uniform_terms = WholeDrawer::new(&counts, &graph, root, Uniform)
            .draw_batch(&[(5, 16)], [11, 13])
            .expect("uniform exhaustive cyclic draw");

        assert_eq!(count_terms.len(), 16);
        assert_eq!(uniform_terms.len(), 16);
        // Both drew all 16 distinct terms and sorted them identically.
        assert_eq!(count_terms, uniform_terms);
    }

    #[test]
    fn exhaustive_ternary_variable_split_draw() {
        use egg::SymbolLang;

        // `f(x, y, z)` where x = {a, g(a)}, y = {b, g(b)}, z = {c, g(c)}.
        // At root size 5, child sizes sum to 4 across 3 children:
        // (2,1,1), (1,2,1), (1,1,2) -> 3 splits.
        // Add a second leaf to each class so each size 1 has 2 terms and
        // size 2 has 2 terms: size 4 has 8 terms and size 5 has 24 (32 total).
        let mut graph = EGraph::<SymbolLang, ()>::new(());
        let mk_class = |eg: &mut EGraph<SymbolLang, ()>, p: &str, q: &str| {
            let l1 = eg.add(SymbolLang::leaf(p));
            let l2 = eg.add(SymbolLang::leaf(q));
            let g = eg.add(SymbolLang::new("g", vec![l1]));
            eg.union(l1, l2);
            eg.union(l1, g);
            l1
        };
        let x = mk_class(&mut graph, "a1", "a2");
        let y = mk_class(&mut graph, "b1", "b2");
        let z = mk_class(&mut graph, "c1", "c2");
        let root = graph.add(SymbolLang::new("f", vec![x, y, z]));
        graph.rebuild();

        let root = graph.find(root);
        let (_, counts) = whole::count(&graph, root, 5, 2, 32).unwrap();
        assert_eq!(counts[&root][&5], BigUint::from(24u32));

        let count_terms = WholeDrawer::new(&counts, &graph, root, Count)
            .draw_batch(&[(5, 24)], [3, 5])
            .expect("ternary count exhaustive draw");
        let uniform_terms = WholeDrawer::new(&counts, &graph, root, Uniform)
            .draw_batch(&[(5, 24)], [3, 5])
            .expect("ternary uniform exhaustive draw");

        assert_eq!(count_terms.len(), 24);
        assert_eq!(count_terms, uniform_terms);

        let lowered = count_terms
            .into_iter()
            .map(|t| origin::lower(t).to_string())
            .collect::<crate::utils::HashSet<_>>();
        assert_eq!(lowered.len(), 24);
    }
}
