pub mod lambda;
pub mod math;
pub mod mini_rise;
pub mod prop;
pub mod sampler;

pub mod origin;

use std::cell::RefCell;
use std::fmt::Display;
use std::rc::Rc;
use std::time::Duration;

use egg::{
    Analysis, BackoffScheduler, CostFunction, EGraph, Extractor, FromOp, Id, Iteration,
    IterationData, Language, LpCostFunction, LpExtractor, RecExpr, Rewrite, Runner,
    SimpleScheduler, StopReason,
};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use crate::cli::argparse::EqsatConfig;
use crate::cli::{EqsatMetadata, GuideError};
use crate::tee_println;

pub use origin::{OriginLang, lower};

/// Trait for node labels in e-graphs and exprs.
pub trait MyLanguage:
    Serialize
    + for<'de> Deserialize<'de>
    + Send
    + Sync
    + Display
    + Language<Discriminant: Send + Sync>
    + FromOp<Error: Display>
    + 'static
{
    /// Returns the label used for type annotations (e.g., "typeOf").
    fn type_of() -> Self;

    /// Returns true if this label is the type annotation label.
    fn is_type_of(&self) -> bool {
        &Self::type_of() == self
    }
}

/// Trait for node labels in e-graphs and exprs.
pub trait MyAnalysis<L: MyLanguage>:
    Serialize
    + for<'de> Deserialize<'de>
    + Send
    + Sync
    + std::fmt::Debug
    + Analysis<L, Data: Send + Sync + Clone + Eq + Default>
    + Default
    + Clone
    + 'static
{
    fn is_typed(id: Id) -> bool;

    fn ty(id: Id) -> Option<RecExpr<OriginLang<L>>>;
}

impl<L: MyLanguage> MyAnalysis<L> for () {
    fn is_typed(_id: Id) -> bool {
        false
    }

    fn ty(_id: Id) -> Option<RecExpr<OriginLang<L>>> {
        None
    }
}

pub fn stack_children<L: MyLanguage>(children: &[RecExpr<L>], root: L) -> RecExpr<L> {
    let mut i = 0;
    root.map_children(|_c| {
        let new_id = Id::from(i);
        i += 1;
        new_id
    })
    .join_recexprs(|c_id| children[usize::from(c_id)].clone())
}

pub fn typed_stack_children<L: MyLanguage>(
    children: &[RecExpr<L>],
    root: L,
    ty: Option<RecExpr<L>>,
) -> RecExpr<L> {
    let untyped = stack_children(children, root);
    if let Some(ty) = ty {
        stack_children(&[untyped, ty], L::type_of())
    } else {
        untyped
    }
}

/// Result of running eqsat. Holds only the last two egraphs (`prev` and
/// `curr`), plus per-iteration metadata in `iter_data` (timings,
/// `egraph_nodes`, etc. — no egraphs). `root` is the id returned by the
/// initial `add`, so it may not be canonical in later iterations —
/// canonicalize with `egraph.find(root)` before using it as a `HashMap` key.
pub struct EqsatResult<L, N>
where
    L: Language,
    N: Analysis<L> + Clone,
    N::Data: Clone,
{
    iter_data: Vec<Iteration<()>>,
    prev: EGraph<L, N>,
    curr: EGraph<L, N>,
    root: Id,
    stop_reason: StopReason,
}

impl<L, N> EqsatResult<L, N>
where
    L: Language,
    N: Analysis<L> + Clone,
    N::Data: Clone,
{
    #[must_use]
    pub fn root(&self) -> Id {
        self.root
    }

    #[must_use]
    pub fn curr(&self) -> &EGraph<L, N> {
        &self.curr
    }

    #[must_use]
    pub fn prev(&self) -> &EGraph<L, N> {
        &self.prev
    }

    /// Per-iteration metadata only (timings, `egraph_nodes`, `egraph_classes`,
    /// `applied`, etc.). Index `0` is the iteration that started from the
    /// initial egraph. `iters()` is the last applied iteration index.
    #[must_use]
    pub fn data(&self) -> &[Iteration<()>] {
        &self.iter_data
    }

    /// Index of the last applied iteration (`iter_data.len() - 1`).
    #[must_use]
    pub fn iters(&self) -> usize {
        self.iter_data.len() - 1
    }

    #[must_use]
    pub fn stop_reason(&self) -> &StopReason {
        &self.stop_reason
    }

    /// Split this run into guide- and goal-phase metadata. The guide phase is
    /// the first half of the applied iterations (`iters() / 2`); the goal phase
    /// is the whole run.
    ///
    /// egg's `Iteration` records `egraph_nodes`/`egraph_classes` at the *start*
    /// of each iteration, so iter K+1's start equals iter K's end. The guide
    /// node/class counts therefore read from `data()[guide_iters + 1]`.
    #[must_use]
    pub fn split_metadata(&self) -> SplitMetadata {
        let goal_iters = self.iters();
        let guide_iters = goal_iters / 2;

        let guide_time = self.iter_data[..=guide_iters]
            .iter()
            .map(|i| i.total_time)
            .sum();
        let goal_time = self.iter_data.iter().map(|i| i.total_time).sum();

        let guide_iter_end = &self.iter_data[guide_iters + 1];
        SplitMetadata {
            guide: EqsatMetadata {
                nodes: guide_iter_end.egraph_nodes,
                classes: guide_iter_end.egraph_classes,
                time: guide_time,
                iters: guide_iters,
            },
            goal: EqsatMetadata {
                nodes: self.curr.total_number_of_nodes(),
                classes: self.curr.classes().len(),
                time: goal_time,
                iters: goal_iters,
            },
        }
    }
}

/// Guide- and goal-phase metadata for a single eqsat run. See
/// [`EqsatResult::split_metadata`].
pub struct SplitMetadata {
    pub guide: EqsatMetadata,
    pub goal: EqsatMetadata,
}

/// Holds the latest two *distinct* egraph snapshots seen by the hook. A
/// snapshot is taken only when the egraph differs from the previous snapshot
/// (`same_egraph` lineage check), so trailing no-op iterations don't shift
/// the slots.
#[derive(Debug)]
struct DistinctSlots<L, N>
where
    L: Language,
    N: Analysis<L>,
{
    /// Latest distinct egraph snapshot.
    distinct: Option<EGraph<L, N>>,
    /// The one before that.
    prev_distinct: Option<EGraph<L, N>>,
}

/// Minimum number of iterations the runner must complete for `run_eqsat` to
/// return `Some`. Lower than this means we don't have enough distinct egraph
/// states for a meaningful guide/goal split.
const MIN_ITERS: usize = 3;

/// Run equality saturation up to `config.max_iters` iterations and return the
/// final egraph (`curr`) together with the last meaningfully different
/// earlier egraph (`prev`).
///
/// Returns `None` if fewer than [`MIN_ITERS`] iterations completed or if the
/// runner never produced a distinct earlier egraph (e.g. saturated with no
/// effective changes).
///
/// # Panics
///
/// Panics if egg's `Runner` returns without a `stop_reason` set, which it
/// documents as impossible.
pub fn run_eqsat<'a, L, N, R>(
    start: &RecExpr<L>,
    rules: R,
    config: &EqsatConfig,
) -> Option<EqsatResult<L, N>>
where
    L: MyLanguage + 'static,
    N: MyAnalysis<L> + Default + Clone + 'static,
    N::Data: Clone,
    R: IntoIterator<Item = &'a Rewrite<L, N>>,
{
    let slots = Rc::new(RefCell::new(DistinctSlots {
        distinct: None,
        prev_distinct: None,
    }));
    let hook_slots = Rc::clone(&slots);

    let mut runner = Runner::default()
        .with_time_limit(Duration::try_from_secs_f64(config.max_time).unwrap_or(Duration::MAX))
        .with_node_limit(config.max_nodes)
        .with_iter_limit(config.max_iters)
        .with_expr(start)
        .with_hook(move |runner| {
            let mut s = hook_slots.borrow_mut();
            let unchanged = s
                .distinct
                .as_ref()
                .is_some_and(|d| same_egraph(d, &runner.egraph));
            if !unchanged {
                s.prev_distinct = s.distinct.take();
                s.distinct = Some(runner.egraph.clone());
            }
            Ok(())
        });

    runner = if config.backoff_scheduler {
        runner.with_scheduler(BackoffScheduler::default())
    } else {
        runner.with_scheduler(SimpleScheduler)
    }
    .run(rules);

    let stop_reason = runner.stop_reason.unwrap();
    tee_println!("Stopped with stop reason: {stop_reason:?}");

    let root = runner.roots[0];
    // Drop hook closures so the slot's Rc has only our local clone left.
    runner.hooks.clear();
    let iter_data = runner.iterations;
    let mut curr = runner.egraph;

    if iter_data.len() < MIN_ITERS {
        tee_println!("Not enough iterations ({} < {MIN_ITERS})", iter_data.len());
        return None;
    }

    let DistinctSlots {
        distinct,
        prev_distinct,
    } = Rc::try_unwrap(slots)
        .expect("hooks cleared, slot Rc should be unique")
        .into_inner();

    let prev = match distinct {
        Some(d) if same_egraph(&d, &curr) => prev_distinct,
        d => d,
    };

    let Some(mut prev) = prev else {
        tee_println!("Egraph never produced a distinct earlier state");
        return None;
    };

    debug_assert!(
        !same_egraph(&prev, &curr),
        "prev/curr should be distinct after selection"
    );

    prev.rebuild();
    curr.rebuild();

    Some(EqsatResult {
        iter_data,
        prev,
        curr,
        root,
        stop_reason,
    })
}

/// Run eqsat from `guides` (all unioned together) and check if `goal` becomes reachable.
/// Returns `Some((iterations, nodes))` if reached, `None` otherwise.
///
/// # Errors
///
/// Errors either if the guide is unrachable or we have a panic
///
/// # Panics
///
/// Panics if not at least one guide is given
pub fn verify_reachability<L, N>(
    guides: &[RecExpr<OriginLang<L>>],
    goal: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    eqsat: &EqsatConfig,
    full_union: bool,
) -> Result<Vec<egg::Iteration<()>>, GuideError>
where
    L: MyLanguage + Language + Display + 'static,
    N: MyAnalysis<L> + Default,
{
    assert!(!guides.is_empty(), "must have at least one guide");
    let goal_clone = goal.clone();

    let mut runner = Runner::default()
        .with_time_limit(Duration::try_from_secs_f64(eqsat.max_time).unwrap_or(Duration::MAX))
        .with_node_limit(eqsat.max_nodes)
        .with_iter_limit(eqsat.max_iters)
        .with_hook(move |runner| {
            if runner.egraph.lookup_expr(&goal_clone).is_some() {
                return Err("goal found".to_owned());
            }
            Ok(())
        });

    runner = if eqsat.backoff_scheduler {
        runner.with_scheduler(BackoffScheduler::default())
    } else {
        runner.with_scheduler(SimpleScheduler)
    };

    runner = if full_union {
        add_with_full_union(runner, guides)
    } else {
        add_with_root_union(runner, guides)
    };

    runner.egraph.rebuild();

    let Ok(r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules))) else {
        println!("Panic caught verify_reachability for guide/goal pair: {guides:?}/{goal}");
        return Err(GuideError::PanicWhileAttempt);
    };
    // let runner = runner.run(rules);

    r.egraph
        .lookup_expr(goal)
        .map(|_| r.iterations)
        .ok_or(GuideError::Unreached(r.stop_reason.clone().unwrap()))
}

fn add_with_root_union<'a, L, N, D, I>(mut runner: Runner<L, N, D>, guides: I) -> Runner<L, N, D>
where
    L: MyLanguage + 'a + Language,
    N: MyAnalysis<L>,
    D: IterationData<L, N>,
    I: IntoIterator<Item = &'a RecExpr<OriginLang<L>>>,
{
    for guide in guides {
        let expr = lower(guide.clone());
        runner = runner.with_expr(&expr);
    }

    // Union all guide roots together before running
    for &root in &runner.roots[1..] {
        runner.egraph.union(runner.roots[0], root);
    }
    runner
}

fn add_with_full_union<'a, L, N, D, I>(mut runner: Runner<L, N, D>, guides: I) -> Runner<L, N, D>
where
    L: MyLanguage + 'a,
    N: MyAnalysis<L>,
    D: IterationData<L, N>,
    I: IntoIterator<Item = &'a RecExpr<OriginLang<L>>>,
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

fn add_uncanon_remember<L: MyLanguage, N: MyAnalysis<L>>(
    graph: &mut EGraph<L, N>,
    guide: &RecExpr<OriginLang<L>>,
    origin_to_new_ids: &mut HashMap<Id, HashSet<Id>>,
) -> Id {
    fn rec<LL: MyLanguage, NN: MyAnalysis<LL>>(
        graph: &mut EGraph<LL, NN>,
        guide: &RecExpr<OriginLang<LL>>,
        origin_to_new_ids: &mut HashMap<Id, HashSet<Id>>,
        id: Id,
    ) -> Id {
        let node = &guide[id]
            .clone()
            .map_children(|c_id| rec(graph, guide, origin_to_new_ids, c_id));
        let new_id = graph.add_uncanonical(node.inner().clone());
        origin_to_new_ids
            .entry(node.origin())
            .or_default()
            .insert(new_id);
        new_id
    }
    rec(graph, guide, origin_to_new_ids, guide.root())
}

/// Check whether two egraphs from the same lineage (one cloned from the other,
/// possibly with further `add` / `union` calls) are still identical.
///
/// Egg only ever grows an egraph: `add` increases the node count, `union`
/// decreases the class count (never the other way around). So for a shared
/// lineage, equal class count *and* equal node count implies no rewrite took
/// effect — the canonical ids in `a` and `b` agree on every class, and the
/// node sets coincide.
///
/// Not valid for comparing independent egraphs: those need a full e-class
/// isomorphism check, since canonical ids depend on union-find history.
#[must_use]
pub fn same_egraph<L, N>(a: &EGraph<L, N>, b: &EGraph<L, N>) -> bool
where
    L: Language,
    N: Analysis<L>,
{
    a.number_of_classes() == b.number_of_classes()
        && a.total_number_of_nodes() == b.total_number_of_nodes()
}

#[must_use]
pub fn id0() -> Id {
    Id::from(0)
}

pub fn cheapest<CF, L, N, I>(runner: &Runner<L, N, I>, cf: CF) -> usize
where
    CF: CostFunction<L, Cost = usize>,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    Extractor::new(&runner.egraph, cf).find_best_cost(runner.roots[0])
}

pub fn cheapest_ilp<CF, L, N, I>(runner: &Runner<L, N, I>, cf: CF) -> RecExpr<L>
where
    CF: LpCostFunction<L, N>,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    let root = runner.egraph.find(runner.roots[0]);
    LpExtractor::new(&runner.egraph, cf).solve(root)
}
