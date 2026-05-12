pub mod lambda;
pub mod math;

pub mod origin;

use std::fmt::Display;
use std::time::Duration;

use egg::{
    Analysis, BackoffScheduler, EGraph, Id, Iteration, IterationData, Language, RecExpr, Rewrite,
    Runner, SimpleScheduler, StopReason,
};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use crate::cli::GuideError;
use crate::cli::argparse::EqsatConfig;
use crate::tee_println;

pub use lambda::{Lambda, LambdaAnalysis};
pub use math::{ConstantFold, Math};
pub use origin::{OriginLang, lower};

/// Trait for node labels in e-graphs and exprs.
pub trait MyLanguage:
    Serialize + for<'de> Deserialize<'de> + Send + Sync + Display + Language<Discriminant: Send + Sync>
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

/// Result of [`run_guide_goal`]: egraph snapshots at guide and goal iterations,
/// plus the total node count of the final egraph.
pub struct GuideGoalResult<L, N>
where
    L: Language,
    N: Analysis<L> + Clone,
    N::Data: Clone,
{
    // /// Iteration Data at `n_guide - 1` (rebuilt), for frontier membership checks.
    iter_data: Vec<Iteration<EGraphHolder<L, N>>>,
    /// Root (valid for all egraphs). Note: this is the Id returned by the
    /// initial `add`, so it may not be canonical in later iterations'
    /// egraphs — canonicalize with `egraph.find(root)` before using as a
    /// `HashMap` key.
    root: Id,
    /// Guide Iteration
    guide_iters: usize,
    /// Goal Iterations
    goal_iters: usize,
    /// Stop reason
    stop_reason: StopReason,
}

impl<L, N> GuideGoalResult<L, N>
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
    pub fn curr_guide(&self) -> &EGraph<L, N> {
        &self.iter_data[self.guide_iters].data.0
    }

    #[must_use]
    pub fn prev_guide(&self) -> &EGraph<L, N> {
        &self.iter_data[self.guide_iters - 1].data.0
    }

    #[must_use]
    pub fn guide_data(&self) -> &[Iteration<EGraphHolder<L, N>>] {
        &self.iter_data[..self.guide_iters]
    }

    #[must_use]
    pub fn curr_goal(&self) -> &EGraph<L, N> {
        &self.iter_data[self.goal_iters].data.0
    }

    #[must_use]
    pub fn prev_goal(&self) -> &EGraph<L, N> {
        &self.iter_data[self.goal_iters - 1].data.0
    }

    #[must_use]
    pub fn goal_data(&self) -> &[Iteration<EGraphHolder<L, N>>] {
        &self.iter_data
    }

    #[must_use]
    pub fn guide_iters(&self) -> usize {
        self.guide_iters
    }

    #[must_use]
    pub fn goal_iters(&self) -> usize {
        self.goal_iters
    }

    #[must_use]
    pub fn stop_reason(&self) -> &StopReason {
        &self.stop_reason
    }
}

pub struct EGraphHolder<L, N>(pub EGraph<L, N>)
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
pub fn big_eqsat<'a, L, N, R>(
    start: &RecExpr<L>,
    rules: R,
    eqsat: &EqsatConfig,
) -> Option<GuideGoalResult<L, N>>
where
    L: Language + 'static,
    N: Analysis<L> + Default + Clone + 'static,
    N::Data: Clone,
    R: IntoIterator<Item = &'a Rewrite<L, N>>,
{
    let mut runner = Runner::<L, N, EGraphHolder<L, N>>::new(Default::default())
        .with_time_limit(Duration::from_secs_f64(eqsat.max_time))
        .with_node_limit(eqsat.max_nodes)
        .with_iter_limit(eqsat.max_iters)
        .with_expr(start);

    runner = if eqsat.backoff_scheduler {
        runner.with_scheduler(BackoffScheduler::default())
    } else {
        runner.with_scheduler(SimpleScheduler)
    }
    .run(rules);

    if !matches!(
        runner.stop_reason.as_ref().unwrap(),
        StopReason::TimeLimit(_) | StopReason::IterationLimit(_) | StopReason::NodeLimit(_)
    ) {
        tee_println!("Failed cause stopped with {:?}", runner.stop_reason);
        return None;
    }

    let stop_reason = runner.stop_reason.unwrap();
    tee_println!("Stopped with stop reason: {stop_reason:?}");

    let root = runner.roots[0];
    let mut iter_data = runner.iterations;

    // Drop trailing iterations whose egraph is identical to its predecessor:
    // `curr_goal` / `prev_goal` would not differ and novelty would be empty by
    // definition. Partial-apply iterations (mid-apply stops) typically produce
    // a changed egraph and are kept.
    while iter_data.len() >= 2
        && same_egraph(
            &iter_data[iter_data.len() - 2].data.0,
            &iter_data[iter_data.len() - 1].data.0,
        )
    {
        iter_data.pop();
    }

    if iter_data.len() < 3 {
        tee_println!("Not enough iterations!");
        return None;
    }

    let goal_iters = iter_data.len() - 1;
    let guide_iters = goal_iters / 2;

    for i in &mut iter_data {
        i.data.0.rebuild();
    }

    Some(GuideGoalResult {
        iter_data,
        root,
        guide_iters,
        goal_iters,
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
        .with_time_limit(Duration::from_secs_f64(eqsat.max_time))
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
        .ok_or(GuideError::Unreached)
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
