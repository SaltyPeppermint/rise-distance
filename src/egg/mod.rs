pub mod lambda;
pub mod math;

use std::fmt::Display;
use std::time::Duration;

use egg::{
    Analysis, BackoffScheduler, EGraph, Id, Iteration, IterationData, Language, RecExpr, Rewrite,
    Runner, SimpleScheduler, StopReason,
};
use hashbrown::{HashMap, HashSet};

use crate::cli::GuideError;
use crate::ids::AnyId;
use crate::tree::TreeShaped;
use crate::{Label, OriginTree, tee_println};

pub use lambda::{Lambda, LambdaAnalysis};
pub use math::{ConstantFold, Math};

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
    /// Root (valid for all egraphs)
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
    pub fn guide(&self) -> &EGraph<L, N> {
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
    pub fn goal(&self) -> &EGraph<L, N> {
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
    time_limit: Duration,
    node_limit: usize,
    backoff_scheduler: bool,
) -> Option<GuideGoalResult<L, N>>
where
    L: Language + 'static,
    N: Analysis<L> + Default + Clone + 'static,
    N::Data: Clone,
    R: IntoIterator<Item = &'a Rewrite<L, N>>,
{
    let mut runner = Runner::<L, N, EGraphHolder<L, N>>::new(Default::default())
        .with_time_limit(time_limit)
        .with_node_limit(node_limit)
        .with_expr(start);

    runner = if backoff_scheduler {
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
    iter_data.pop();

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

pub trait ToEgg<L: Label + Language>: TreeShaped<L> {
    fn to_rec_expr(&self) -> RecExpr<L> {
        let mut expr = RecExpr::default();
        let mut adder = |_: &_, x| expr.add(x);
        self.add_node(&mut adder);
        expr
    }

    fn add_node<F: FnMut(&Self, L) -> Id>(&self, adder: &mut F) -> Id;
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
    guides: &[OriginTree<L>],
    goal: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    time_limit: Duration,
    node_limit: usize,
    full_union: bool,
    backoff_scheduler: bool,
) -> Result<Vec<egg::Iteration<()>>, GuideError>
where
    L: Label + Language + Display + 'static,
    N: Analysis<L> + Default,
    OriginTree<L>: ToEgg<L>,
{
    assert!(!guides.is_empty(), "must have at least one guide");
    let goal_clone = goal.clone();

    let mut runner = Runner::default()
        .with_time_limit(time_limit)
        .with_node_limit(node_limit)
        .with_hook(move |runner| {
            if runner.egraph.lookup_expr(&goal_clone).is_some() {
                return Err("goal found".to_owned());
            }
            Ok(())
        });

    runner = if backoff_scheduler {
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
    L: Label + 'a + Language,
    N: Analysis<L>,
    D: IterationData<L, N>,
    OriginTree<L>: ToEgg<L>,
    I: IntoIterator<Item = &'a OriginTree<L>>,
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

fn add_with_full_union<'a, L, N, D, I>(mut runner: Runner<L, N, D>, guides: I) -> Runner<L, N, D>
where
    L: Label + 'a + Language,
    N: Analysis<L>,
    D: IterationData<L, N>,
    OriginTree<L>: ToEgg<L>,
    I: IntoIterator<Item = &'a OriginTree<L>>,
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

fn add_uncanon_remember<L, N>(
    graph: &mut EGraph<L, N>,
    guide: &OriginTree<L>,
    origin_to_new_ids: &mut HashMap<AnyId, HashSet<Id>>,
) -> Id
where
    L: Language + Label,
    N: Analysis<L>,
    OriginTree<L>: ToEgg<L>,
{
    let mut adder = |node: &OriginTree<L>, lang_node| {
        let new_id = graph.add_uncanonical(lang_node);
        origin_to_new_ids
            .entry(node.origin())
            .or_default()
            .insert(new_id);
        new_id
    };
    guide.add_node(&mut adder)
}

#[must_use]
pub fn id0() -> Id {
    Id::from(0)
}
