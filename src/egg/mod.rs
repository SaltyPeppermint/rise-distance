pub mod lambda;
pub mod math;

use std::fmt::Display;
use std::mem;
use std::time::{Duration, Instant};

use egg::{
    Analysis, EClass, EGraph, Id, Iteration, IterationData, Language, RecExpr, Rewrite, Runner,
    SimpleScheduler, StopReason,
};
use hashbrown::{HashMap, HashSet};
use memory_stats::memory_stats;

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
) -> Option<GuideGoalResult<L, N>>
where
    L: Language + 'static,
    N: Analysis<L> + Default + Clone + 'static,
    N::Data: Clone,
    R: IntoIterator<Item = &'a Rewrite<L, N>>,
{
    let runner = Runner::<L, N, EGraphHolder<L, N>>::new(Default::default())
        .with_scheduler(SimpleScheduler)
        .with_time_limit(time_limit)
        .with_node_limit(node_limit)
        .with_expr(start)
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
) -> Result<Vec<egg::Iteration<()>>, GuideError>
where
    L: Label + Language + Display + 'static,
    N: Analysis<L> + Default,
    OriginTree<L>: ToEgg<L>,
{
    assert!(!guides.is_empty(), "must have at least one guide");
    let goal_clone = goal.clone();

    let mut runner = Runner::default()
        .with_scheduler(SimpleScheduler)
        .with_time_limit(time_limit)
        .with_node_limit(node_limit)
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

pub struct ValidationResult {
    pub stop_reason: StopReason,
    pub stop_nodes: usize,
    pub stop_classes: usize,
    pub stop_time: f64,
    pub last_nodes: usize,
    pub last_classes: usize,
    pub last_time: f64,
    pub mem: usize,
    pub egraph_bytes: usize,
}

pub fn valididty_hook<L: Label + Language + Display, N: Analysis<L> + Default, T: ToEgg<L>>(
    tree: &T,
    max_iters: usize,
    max_nodes: usize,
    max_time: f64,
    rules: &[Rewrite<L, N>],
) -> Option<ValidationResult> {
    let expr = tree.to_rec_expr();
    // egg's Runner can panic on certain malformed inside its merge check.
    // Fixing this would require only constructing correct terms and that is too complicated
    // We use catch_unwind to treat such cases as "not passing the check" rather than crashing the process.
    // An example would be the expression
    // '(cos (* (sqrt (* x (sqrt (i (/ 0 x) x)))) (sin (+ (pow 1 (/ 1 2)) (cos 2)))))'
    // The issue is that the binder check does not catch (i (/ 0 x) x) although (/ 0 x)
    // trivially simplifies to 0
    let runner = Runner::default()
        .with_expr(&expr)
        .with_iter_limit(max_iters)
        .with_node_limit(max_nodes)
        .with_time_limit(Duration::from_secs_f64(max_time))
        .with_scheduler(SimpleScheduler);

    // Setting and unsetting the panic hook so we dont get debug spam. it is fine to ignore the output
    // Afterwards we reinstall the old default panic hook
    let start = Instant::now();
    let Ok(r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules))) else {
        println!("panic caught in iter_check_hook for expr: {expr}");
        println!("It is safe to ignore the output of egg here");
        return None;
    };
    let stop_time = start.elapsed().as_secs_f64();

    let stop_reason = r.stop_reason.clone()?;
    let egraph_bytes = estimate_egraph_bytes(&r.egraph);
    let stop_nodes = r.egraph.nodes().len();
    let stop_classes = r.egraph.classes().len();

    let last_nodes = r.iterations.last()?.egraph_nodes;
    let last_classes = r.iterations.last()?.egraph_classes;
    let last_time = r.iterations.last()?.total_time;

    let before_drop = memory_stats()?;
    drop(r);
    let after_drop = memory_stats()?;
    if matches!(
        stop_reason,
        StopReason::IterationLimit(_) | StopReason::NodeLimit(_) | StopReason::TimeLimit(_)
    ) {
        return Some(ValidationResult {
            stop_reason,
            stop_nodes,
            stop_classes,
            stop_time,
            last_nodes,
            last_classes,
            last_time,
            mem: before_drop.physical_mem - after_drop.physical_mem,
            egraph_bytes,
        });
    }
    None
}

/// Structural estimate of the heap bytes held by `g`.
///
/// Walks only the public API ([`EGraph::nodes`], [`EGraph::classes`],
/// [`EGraph::total_size`], [`EGraph::total_number_of_nodes`],
/// [`EGraph::number_of_classes`]) and assumes:
/// - `L` and `N::Data` are `'static` and own no heap allocations themselves
///   (true for our `Math`/`Lambda` languages).
/// - Explanations are disabled, so `explain: Option<Explain<L>>` contributes
///   only its discriminant.
/// - The egraph has just been rebuilt, so `pending` and `analysis_pending`
///   are effectively empty.
///
/// Approximations:
/// - `memo` and `classes` are sized assuming hashbrown's load factor (≤ 7/8)
///   and a power-of-two bucket count, plus one metadata byte per slot.
/// - `unionfind` is treated as a `Vec<Id>` with one slot per id ever added
///   (i.e. `nodes().len()`); the real `UnionFind` has identical asymptotic
///   storage but may carry a small constant of bookkeeping.
/// - `classes_by_op` is sized by walking `g.nodes()` to collect the distinct
///   discriminants actually present, then querying [`EGraph::classes_for_op`]
///   for each to learn the per-op `HashSet<Id>` length. Each set is sized
///   individually with the same hashbrown formula. This is exact in the
///   payload (one `Id` per canonical enode) and tight on the per-op bucket
///   overhead, but still ignores any unused-but-allocated capacity in those
///   sets.
///
/// Things deliberately not counted:
/// - `explain` contents (assumed disabled).
/// - Transient queues (`pending`, `analysis_pending`).
/// - Allocator slack and per-allocation headers.
/// - Any heap data hanging off `L` or `N::Data` (caller's responsibility).
pub fn estimate_egraph_bytes<L, N>(g: &EGraph<L, N>) -> usize
where
    L: Language,
    N: Analysis<L>,
{
    let n_nodes_total = g.nodes().len();

    let mut bytes = mem::size_of::<EGraph<L, N>>();

    // `nodes: Vec<L>` with one slot per id ever added (non-canonical included).
    bytes += mem::size_of_val(g.nodes());

    // `unionfind`: one parent Id per id ever added.
    bytes += n_nodes_total * mem::size_of::<Id>();

    // `memo: HashMap<L, Id>`
    bytes += hashbrown_bytes::<L, Id>(g.total_size());

    // Per-class storage.
    for class in g.classes() {
        bytes += mem::size_of::<EClass<L, N::Data>>();
        bytes += class.nodes.len() * mem::size_of::<L>();
        bytes += class.parents().len() * mem::size_of::<Id>();
    }

    // `classes_by_op`: discriminants present, then per-op HashSet<Id> sizes.
    let mut discriminants = HashSet::new();
    for node in g.nodes() {
        discriminants.insert(node.discriminant());
    }
    bytes += hashbrown_bytes::<L::Discriminant, ()>(discriminants.len());
    for disc in &discriminants {
        let len = g.classes_for_op(disc).map_or(0, |it| it.len());
        bytes += hashbrown_bytes::<Id, ()>(len);
    }

    // `classes: HashMap<Id, EClass<..>>` shell (EClass bodies counted above).
    bytes += hashbrown_bytes::<Id, ()>(g.number_of_classes());

    bytes
}

fn hashbrown_bytes<K, V>(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let cap = (len * 8).div_ceil(7).next_power_of_two();
    cap * (mem::size_of::<(K, V)>() + 1)
}

#[must_use]
pub fn id0() -> Id {
    Id::from(0)
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
