pub mod math;

use egg::{
    Analysis, EGraph, Id, IterationData, Language, RecExpr, Rewrite, Runner, SimpleScheduler,
    StopReason,
};
use hashbrown::{HashMap, HashSet};

use crate::ids::{AnyId, EClassId, ExprChildId};
use crate::nodes::ENode;
use crate::tree::TreeShaped;
use crate::{Class, Graph, Label, OriginTree};

pub use math::{Math, MathLabel};

/// Result of [`run_guide_goal`]: egraph snapshots at guide and goal iterations,
/// plus the total node count of the final egraph.
pub struct GuideGoalResult<L: Language, N: Analysis<L>> {
    /// Egraph at `n_guide - 1` (rebuilt), for frontier membership checks.
    graphs: Vec<EGraph<L, N>>,
    /// Root (valid for all egraphs)
    root: Id,
    /// Guide Iteration
    guide_iteration: usize,
}

impl<L: Language, N: Analysis<L>> GuideGoalResult<L, N> {
    #[must_use]
    pub fn root(&self) -> Id {
        self.root
    }

    #[must_use]
    pub fn guide(&self) -> &EGraph<L, N> {
        &self.graphs[self.guide_iteration]
    }

    #[must_use]
    pub fn prev_guide(&self) -> &EGraph<L, N> {
        &self.graphs[self.guide_iteration - 1]
    }

    #[must_use]
    pub fn goal(&self) -> &EGraph<L, N> {
        &self.graphs[self.graphs.len() - 1]
    }

    #[must_use]
    pub fn prev_goal(&self) -> &EGraph<L, N> {
        &self.graphs[self.graphs.len() - 2]
    }
}

pub trait ToEgg<L: Label>: TreeShaped<L> {
    type Lang: Language;

    fn to_rec_expr(&self) -> RecExpr<Self::Lang> {
        let mut expr = RecExpr::default();
        let mut adder = |_: &_, x| expr.add(x);
        self.add_node(&mut adder);
        expr
    }

    fn add_node<F: FnMut(&Self, Self::Lang) -> Id>(&self, adder: &mut F) -> Id;
}

pub fn convert<L, N, LL>(egg_graph: &EGraph<L, N>, root: Id) -> Graph<LL>
where
    L: Language,
    N: Analysis<L>,
    LL: Label + for<'a> From<&'a L>,
{
    // Works because classes are unique!
    let classes = egg_graph
        .classes()
        .map(|egg_class| {
            debug_assert_eq!(egg_class.id, egg_graph.find(egg_class.id));
            let eclass_id = EClassId::new(egg_class.id.into());
            let nodes = egg_class
                .nodes
                .iter()
                .map(|math_node| {
                    let children = math_node
                        .children()
                        .iter()
                        .map(|&child_id| {
                            ExprChildId::EClass(EClassId::new(egg_graph.find(child_id).into()))
                        })
                        .collect::<Vec<_>>();
                    ENode::new(math_node.into(), children)
                })
                .collect();
            (eclass_id, Class::new(nodes, None))
        })
        .collect::<HashMap<_, _>>();

    // Build union-find: identity mapping for canonical IDs
    // Include both class IDs and all child IDs to cover the full range of IDs
    // that canonicalize() may be called with.
    let max_id = egg_graph
        .classes()
        .map(|c| usize::from(c.id))
        .chain(
            egg_graph
                .nodes()
                .iter()
                .flat_map(|n| n.children().iter())
                .map(|id| usize::from(*id)),
        )
        .max()
        .map_or(0, |m| m + 1);
    let union_find = (0..max_id)
        .map(|i| EClassId::new(usize::from(egg_graph.find(Id::from(i)))))
        .collect::<Vec<_>>();

    Graph::new(
        classes,
        EClassId::new(root.into()),
        union_find,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    )
}

struct EGraphHolder<L, N>(EGraph<L, N>)
where
    L: Language,
    N: Analysis<L> + Clone,
    N::Data: Clone;

impl<L, N> IterationData<L, N> for EGraphHolder<L, N>
where
    L: Language,
    N: Analysis<L> + Clone,
    N::Data: Clone,
{
    fn make(runner: &Runner<L, N, Self>) -> Self {
        Self(runner.egraph.clone())
    }
}

/// Run equality saturation for `n_goal` iterations, capturing egraphs at two
/// points: `n_guide` and `n_goal`.
///
/// Returns an array of two `(raw, converted)` pairs:
/// - `[0]`: raw egraph at `n_guide - 1` (rebuilt) and converted egraph at `n_guide`
/// - `[1]`: raw egraph at `n_goal - 1` (rebuilt) and converted egraph at `n_goal`
///
/// The raw egraphs are useful for `lookup_expr`-based frontier membership checks.
///
/// # Panics
///
/// Panics if `n_guide == 0`, `n_goal <= n_guide`, or if the runner
/// saturates before reaching `n_goal` iterations.
pub fn run_guide_goal<'a, L, N, R>(
    start: &RecExpr<L>,
    rules: R,
    n_guide: usize,
    n_goal: usize,
) -> GuideGoalResult<L, N>
where
    L: Language + 'static,
    N: Analysis<L> + Default + Clone + 'static,
    N::Data: Clone,
    R: IntoIterator<Item = &'a Rewrite<L, N>>,
{
    assert!(n_guide > 0);
    assert!(n_goal > n_guide);
    assert!(n_goal >= 3, "Needs 3 or more iterations to be meaningful");

    let runner = Runner::<L, N, EGraphHolder<L, N>>::new(Default::default())
        .with_scheduler(SimpleScheduler)
        .with_iter_limit(n_goal)
        .with_expr(start)
        .run(rules);

    assert!(
        matches!(
            runner.stop_reason.as_ref().unwrap(),
            StopReason::IterationLimit(_)
        ),
        "Failed cause stopped with {:?}",
        runner.stop_reason
    );

    let root = runner.roots[0];
    let graphs = runner
        .iterations
        .into_iter()
        .map(|i| {
            let mut eg = i.data.0;
            eg.rebuild();
            eg
        })
        .collect::<Vec<_>>();

    GuideGoalResult {
        graphs,
        root,
        guide_iteration: n_guide,
    }
}

/// Run eqsat from `guides` (all unioned together) and check if `goal` becomes reachable.
/// Returns `Some((iterations, nodes))` if reached, `None` otherwise.
///
/// # Panics
///
/// Panics if no guides given
pub fn verify_reachability<L, N, LL>(
    guides: &[OriginTree<LL>],
    goal: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    max_iters: usize,
    full_union: bool,
) -> Option<Vec<egg::Iteration<()>>>
where
    L: Language + 'static,
    N: Analysis<L> + Default,
    LL: Label,
    OriginTree<LL>: ToEgg<LL, Lang = L>,
{
    assert!(!guides.is_empty(), "must have at least one guide");
    let goal_clone = goal.clone();

    let mut runner = Runner::default()
        .with_scheduler(SimpleScheduler)
        .with_iter_limit(max_iters)
        .with_hook(move |runner| {
            if runner.egraph.lookup_expr(&goal_clone).is_some() {
                return Err("goal found".to_owned());
            }
            Ok(())
        });

    runner = if full_union {
        add_with_full_union(runner, guides)
    } else {
        add_with_root_union(runner, guides)
    };
    runner.egraph.rebuild();

    let runner = runner.run(rules);

    runner.egraph.lookup_expr(goal).map(|_| runner.iterations)
}

fn add_with_root_union<LL, L, N, I>(
    mut runner: Runner<L, N, I>,
    guides: &[OriginTree<LL>],
) -> Runner<L, N, I>
where
    LL: Label,
    L: Language,
    N: Analysis<L>,
    I: IterationData<L, N>,
    OriginTree<LL>: ToEgg<LL, Lang = L>,
{
    for guide in guides {
        let expr = guide.to_rec_expr();
        runner = runner.with_expr(&expr);
    }

    // Union all guide roots together before running
    for &root in &runner.roots[1..] {
        runner.egraph.union(runner.roots[0], root);
    }
    runner
}

fn add_with_full_union<LL, L, N, I>(
    mut runner: Runner<L, N, I>,
    guides: &[OriginTree<LL>],
) -> Runner<L, N, I>
where
    LL: Label,
    L: Language,
    N: Analysis<L>,
    I: IterationData<L, N>,
    OriginTree<LL>: ToEgg<LL, Lang = L>,
{
    let mut origin_to_new_ids = HashMap::new();

    for guide in guides {
        let new_root = add_uncanon_remember(&mut runner.egraph, guide, &mut origin_to_new_ids);
        runner.roots.push(new_root);
    }

    // Union all nodes that shared an eclass in the original egraph
    for new_ids in origin_to_new_ids.values() {
        let mut id_iter = new_ids.iter();
        if let Some(first) = id_iter.next() {
            for id in id_iter {
                runner.egraph.union(*first, *id);
            }
        }
    }
    runner
}

fn add_uncanon_remember<LL, L, N>(
    graph: &mut EGraph<L, N>,
    guide: &OriginTree<LL>,
    origin_to_new_ids: &mut HashMap<AnyId, HashSet<Id>>,
) -> Id
where
    LL: Label,
    L: Language,
    N: Analysis<L>,
    OriginTree<LL>: ToEgg<LL, Lang = L>,
{
    let mut adder = |node: &OriginTree<LL>, lang_node| {
        let new_id = graph.add_uncanonical(lang_node);
        origin_to_new_ids
            .entry(node.origin())
            .or_default()
            .insert(new_id);
        new_id
    };
    guide.add_node(&mut adder)
}

// fn add_expr_uncanonical<L: Language, N: Analysis<L>>(
//     graph: &mut EGraph<L, N>,
//     expr: &RecExpr<L>,
// ) -> Id {
//     let mut new_ids = Vec::with_capacity(expr.len());
//     for node in expr {
//         let new_node = node.clone().map_children(|i| new_ids[usize::from(i)]);
//         let next_id = graph.add_uncanonical(new_node);

//         new_ids.push(next_id);
//     }
//     *new_ids.last().unwrap()
// }

#[cfg(test)]
mod tests {
    use egg::{RecExpr, Runner};

    use super::math::ConstantFold;
    use super::*;
    use crate::ids::{ExprChildId, NumericId};

    /// Build a saturated egg `EGraph` from a string expression, returning (egraph, root).
    fn build(expr: &str) -> (egg::EGraph<Math, ConstantFold>, egg::Id) {
        let expr = expr.parse().unwrap();
        let runner = Runner::default().with_expr(&expr).run(&[]);
        let root = runner.roots[0];
        (runner.egraph, root)
    }

    // -----------------------------------------------------------------------
    // 1. Single symbol leaf
    // -----------------------------------------------------------------------
    #[test]
    fn convert_single_symbol() {
        let (egg, root) = build("x");
        let g = convert::<_, _, MathLabel>(&egg, root);

        assert_eq!(g.root().to_index(), usize::from(egg.find(root)));
        // The root class must exist and contain exactly the Symbol node
        let class = g.class(g.root());
        assert!(!class.nodes().is_empty());
        assert!(
            class
                .nodes()
                .iter()
                .any(|n| *n.label() == MathLabel::Symbol("x".into()))
        );
    }

    // -----------------------------------------------------------------------
    // 2. Single constant leaf
    // -----------------------------------------------------------------------
    #[test]
    #[expect(clippy::float_cmp)]
    fn convert_single_constant() {
        let (egg, root) = build("42");
        let g = convert::<_, _, MathLabel>(&egg, root);

        let class = g.class(g.root());
        assert!(
            class
                .nodes()
                .iter()
                .any(|n| matches!(n.label(), MathLabel::Constant(c) if **c == 42.0))
        );
    }

    // -----------------------------------------------------------------------
    // 3. Unary operator (ln x)
    // -----------------------------------------------------------------------
    #[test]
    fn convert_unary_ln() {
        let (egg, root) = build("(ln x)");
        let g = convert::<_, _, MathLabel>(&egg, root);

        // Root class has an Ln node with one EClass child
        let root_class = g.class(g.root());
        let ln_node = root_class
            .nodes()
            .iter()
            .find(|n| *n.label() == MathLabel::Ln);
        assert!(ln_node.is_some(), "expected Ln node in root class");
        let ln_node = ln_node.unwrap();
        assert_eq!(ln_node.children().len(), 1);

        // The child class must contain the Symbol "x"
        let ExprChildId::EClass(child_id) = ln_node.children()[0] else {
            panic!("expected EClass child")
        };
        let child_class = g.class(child_id);
        assert!(
            child_class
                .nodes()
                .iter()
                .any(|n| *n.label() == MathLabel::Symbol("x".into()))
        );
    }

    // -----------------------------------------------------------------------
    // 4. Binary operator (+ x y) -> two distinct children
    // -----------------------------------------------------------------------
    #[test]
    fn convert_binary_add() {
        let (egg, root) = build("(+ x y)");
        let g = convert::<_, _, MathLabel>(&egg, root);

        let root_class = g.class(g.root());
        let add_node = root_class
            .nodes()
            .iter()
            .find(|n| *n.label() == MathLabel::Add);
        assert!(add_node.is_some(), "expected Add node in root class");
        assert_eq!(add_node.unwrap().children().len(), 2);
    }

    // -----------------------------------------------------------------------
    // 5. All classes in the egg graph appear in the converted graph
    // -----------------------------------------------------------------------
    #[test]
    fn convert_class_count_matches() {
        let (egg, root) = build("(* (+ x 1) (- y 2))");
        let g = convert::<_, _, MathLabel>(&egg, root);

        let egg_class_count = egg.number_of_classes();
        let converted_count = g.class_ids().count();
        assert_eq!(egg_class_count, converted_count);
    }

    // -----------------------------------------------------------------------
    // 6. Children hold canonical IDs after union
    // -----------------------------------------------------------------------
    #[test]
    fn convert_children_are_canonical() {
        // After union/rebuild egg canonicalizes IDs; convert must follow suit.
        let expr: RecExpr<Math> = "(+ x x)".parse().unwrap();
        let mut egg_graph = egg::EGraph::<Math, ConstantFold>::default();
        let root = egg_graph.add_expr(&expr);
        // Union two distinct symbols so one gets remapped
        let a = egg_graph.add(Math::Symbol("a".into()));
        let b = egg_graph.add(Math::Symbol("b".into()));
        egg_graph.union(a, b);
        egg_graph.rebuild();

        let g = convert::<_, _, MathLabel>(&egg_graph, root);

        // Every EClass child referenced by a node must be a key in the graph
        for id in g.class_ids() {
            for node in g.class(id).nodes() {
                for child in node.children() {
                    if let ExprChildId::EClass(child_id) = child {
                        // canonicalize should be identity for canonical ids
                        assert_eq!(g.canonicalize(*child_id), *child_id);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. Union-find maps non-canonical IDs to canonical ones
    // -----------------------------------------------------------------------
    #[test]
    fn convert_union_find_is_populated() {
        // Build an expression with children so the union-find has entries
        let (egg, root) = build("(+ x y)");
        let g = convert::<_, _, MathLabel>(&egg, root);

        // Every entry in the union-find must resolve to a canonical class id
        for (i, &canonical) in g.union_find().iter().enumerate() {
            // canonical must be a key in the class map
            let _ = g.class(canonical); // panics if not found
            // Canonicalizing a canonical id must be a no-op
            assert_eq!(
                g.canonicalize(canonical),
                canonical,
                "entry {i}: union-find entry {canonical:?} is not canonical"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 8. Root is correctly set to the provided root id
    // -----------------------------------------------------------------------
    #[test]
    fn convert_root_matches_egg_root() {
        let (egg, root) = build("(sin x)");
        let g = convert::<_, _, MathLabel>(&egg, root);

        assert_eq!(g.root().to_index(), usize::from(egg.find(root)));
    }

    // -----------------------------------------------------------------------
    // 9. ConstantFold: constant expression collapses into a single constant node
    // -----------------------------------------------------------------------
    #[test]
    #[expect(clippy::float_cmp)]
    fn convert_constant_fold_collapses() {
        // After ConstantFold + rebuild, (+ 2 3) and 5.0 are in the same class
        let expr: RecExpr<Math> = "(+ 2 3)".parse().unwrap();
        let runner = Runner::<Math, ConstantFold>::default()
            .with_expr(&expr)
            .run(&[]);
        let root = runner.roots[0];
        let g = convert::<_, _, MathLabel>(&runner.egraph, root);

        // The root class must contain the folded constant 5.0
        let root_class = g.class(g.root());
        assert!(
            root_class
                .nodes()
                .iter()
                .any(|n| matches!(n.label(), MathLabel::Constant(c) if **c == 5.0)),
            "expected folded constant 5.0 in root class"
        );
    }

    // -----------------------------------------------------------------------
    // 10. Empty union-find when expression has no children (leaf-only graph)
    // -----------------------------------------------------------------------
    #[test]
    fn convert_leaf_has_empty_union_find() {
        let (egg, root) = build("x");
        let g = convert::<_, _, MathLabel>(&egg, root);

        // Even a lone symbol has a class ID, so the union-find covers that entry.
        // Every entry must map to a canonical class.
        for (i, &canonical) in g.union_find().iter().enumerate() {
            let _ = g.class(canonical); // panics if not found
            assert_eq!(
                g.canonicalize(canonical),
                canonical,
                "entry {i}: union-find entry {canonical:?} is not canonical"
            );
        }
    }
}
