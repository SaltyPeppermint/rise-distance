use std::{cell::RefCell, rc::Rc, time::Duration};

use egg::{
    Analysis, AstSize, BackoffScheduler, EGraph, Id, Iteration, IterationData, Language, RecExpr,
    Rewrite, Runner, SimpleScheduler, StopReason,
};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use strum::Display;
use thiserror::Error;

use crate::langs::{MyAnalysis, MyLanguage};
use crate::origin::{OriginLang, lower};
use crate::sketch::{self, Sketch};
use crate::utils::{HeapDelta, live_heap_bytes};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EqsatMetadata {
    pub nodes: usize,
    pub classes: usize,
    pub time: f64,
    pub iters: usize,
}

#[derive(Debug, Error, Display, Serialize, Clone)]
pub enum GuideError {
    Unreached(StopReason),
    PanicWhileAttempt,
}

/// Heap coordinates shared by every memory-aware component of one egg run.
///
/// `baseline` is the absolute process live heap sampled immediately before the
/// [`Runner`] is constructed. `relative_limit` is the public absolute
/// `max_memory` ceiling converted once into run-relative coordinates. Copies of
/// this value retain exactly the same baseline; hooks must receive a copy
/// instead of capturing a new [`HeapDelta`].
#[derive(Debug, Clone, Copy)]
pub struct RunHeap {
    baseline: HeapDelta,
    absolute_limit: Option<u64>,
    relative_limit: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct RunHeapReading {
    absolute: u64,
    relative: u64,
}

impl RunHeap {
    /// Capture the sole pre-run baseline and convert the configured absolute
    /// process-heap ceiling into the run-relative coordinate system.
    #[must_use]
    fn start(max_memory: Option<u64>) -> Self {
        let baseline = HeapDelta::start();
        let relative_limit = max_memory.map(|limit| baseline.relative_to(limit));
        Self {
            baseline,
            absolute_limit: max_memory,
            relative_limit,
        }
    }

    /// Current live allocation relative to the shared pre-run baseline.
    #[must_use]
    pub fn current_relative(self) -> u64 {
        self.current().relative
    }

    /// Rebase an absolute process live-heap reading against the shared
    /// baseline. This is also useful to memory-aware hooks with an existing
    /// heap sample.
    #[must_use]
    pub const fn relative_to(self, absolute_live_heap: u64) -> u64 {
        self.baseline.relative_to(absolute_live_heap)
    }

    /// The shared absolute process live-heap baseline.
    #[must_use]
    pub const fn baseline(self) -> u64 {
        self.baseline.baseline()
    }

    /// The configured absolute ceiling converted to run-relative coordinates.
    #[must_use]
    pub const fn relative_limit(self) -> Option<u64> {
        self.relative_limit
    }

    /// Take one absolute process-heap sample and derive its run-relative value.
    fn current(self) -> RunHeapReading {
        let absolute = live_heap_bytes();
        RunHeapReading {
            absolute,
            relative: self.relative_to(absolute),
        }
    }

    #[cfg(test)]
    fn from_baseline(baseline: u64, max_memory: Option<u64>) -> Self {
        let baseline = HeapDelta::from_baseline(baseline);
        Self {
            baseline,
            absolute_limit: max_memory,
            relative_limit: max_memory.map(|limit| baseline.relative_to(limit)),
        }
    }
}

impl EqsatMetadata {
    /// Summarize a single eqsat run from its per-iteration log. egg records
    /// `egraph_nodes`/`egraph_classes` at the *start* of each iteration, so the
    /// last entry holds the final size. `time` sums every iteration's
    /// `total_time`; `iters` is the index of the last applied iteration
    /// (`len() - 1`), matching [`crate::langs::EqsatResult::iters`].
    ///
    /// # Panics
    ///
    /// Panics if `iterations` is empty (a runner always logs at least one).
    #[must_use]
    pub fn from_iterations(iterations: &[Iteration<()>]) -> Self {
        let last = iterations.last().expect("eqsat run logged no iterations");
        Self {
            nodes: last.egraph_nodes,
            classes: last.egraph_classes,
            time: iterations.iter().map(|i| i.total_time).sum(),
            iters: iterations.len() - 1,
        }
    }
}

/// Eqsat resource limits and scheduler choice. Doubles as the shared clap flag
/// group (`--max-*` / `--backoff-scheduler`) for the `goal` / `sample` /
/// `verify` binaries; the Python drivers read the values out of the
/// `generation_args.json` / `goal_args.json` sidecars and forward them on
/// argv.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize, clap::Args)]
pub struct EqsatConfig {
    /// Maximum eqsat iterations.
    #[arg(long)]
    pub max_iters: usize,

    /// Maximum eqsat egraph nodes.
    #[arg(long)]
    pub max_nodes: usize,

    /// Maximum eqsat wall-clock seconds.
    #[arg(long)]
    pub max_time: f64,

    /// Absolute process live-heap ceiling in bytes (jemalloc
    /// `stats.allocated`), enforced by a per-iteration hook (egg has no native
    /// memory limit). At run setup it is converted to a limit relative to the
    /// shared pre-run baseline. `None` (flag unset) = unbounded.
    #[serde(default)]
    #[arg(long)]
    pub max_memory: Option<u64>,

    /// Use the backoff scheduler instead of the simple one.
    #[arg(long)]
    pub backoff_scheduler: bool,
}

impl EqsatConfig {
    /// Build a [`Runner`] configured with this config's limits (including the
    /// live-heap hook when `max_memory` is set) and scheduler.
    #[must_use]
    pub fn build_runner<L, N, D>(&self, expr: &RecExpr<L>) -> (Runner<L, N, D>, RunHeap)
    where
        L: MyLanguage,
        N: MyAnalysis<L>,
        D: IterationData<L, N>,
    {
        // This must remain immediately before Runner construction: allocations
        // by Runner::new and with_expr belong to this run.
        let heap = RunHeap::start(self.max_memory);
        let runner = Runner::<L, N, D>::new(N::default())
            .with_expr(expr)
            .with_iter_limit(self.max_iters)
            .with_node_limit(self.max_nodes)
            .with_time_limit(Duration::from_secs_f64(self.max_time))
            .with_hook(memory_limit_hook(heap));
        let runner = if self.backoff_scheduler {
            runner.with_scheduler(BackoffScheduler::default())
        } else {
            runner.with_scheduler(SimpleScheduler)
        };
        (runner, heap)
    }
}

/// Result of running eqsat. Holds only the last two egraphs (`prev` and
/// `curr`), plus per-iteration metadata in `iter_data` (timings,
/// `egraph_nodes`, etc.). `root` is the id returned by the
/// initial `add`, so it may not be canonical in later iterations.
/// It also canonicalizes with `egraph.find(root)` before using it as a `HashMap` key.
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
    heap: RunHeap,
}

impl<L, N> EqsatResult<L, N>
where
    L: Language,
    N: Analysis<L> + Clone,
    N::Data: Clone,
{
    /// Test-only constructor so downstream code (e.g. sampling) can be
    /// exercised on hand-built egraph pairs without running eqsat.
    #[cfg(test)]
    pub(crate) fn new_for_tests(prev: EGraph<L, N>, curr: EGraph<L, N>, root: Id) -> Self {
        Self {
            iter_data: Vec::new(),
            prev,
            curr,
            root,
            stop_reason: StopReason::Saturated,
            heap: RunHeap::start(None),
        }
    }

    #[must_use]
    pub const fn root(&self) -> Id {
        self.root
    }

    #[must_use]
    pub const fn curr(&self) -> &EGraph<L, N> {
        &self.curr
    }

    #[must_use]
    pub const fn prev(&self) -> &EGraph<L, N> {
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
    pub const fn iters(&self) -> usize {
        self.iter_data.len() - 1
    }

    #[must_use]
    pub const fn stop_reason(&self) -> &StopReason {
        &self.stop_reason
    }

    /// Current run-relative live allocation using the baseline captured before
    /// this result's Runner was constructed.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        self.heap.current_relative()
    }

    /// The heap coordinate system shared by this run's hooks and measurements.
    #[must_use]
    pub const fn run_heap(&self) -> RunHeap {
        self.heap
    }

    /// Consume the result and return the final egraph together with the root id.
    #[must_use]
    pub fn into_curr(self) -> (EGraph<L, N>, Id) {
        (self.curr, self.root)
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

/// Per-iteration hook enforcing the live-heap ceiling (egg has no native memory
/// limit): returning `Err` stops the run and egg records it as
/// `StopReason::Other`. A no-op when `max_memory` is `None`.
///
/// The public ceiling is an absolute process-live-heap value. [`RunHeap`]
/// converts it once to a limit relative to the shared pre-run baseline; this
/// hook compares the current run-relative allocation to that relative limit.
/// The comparison has the same absolute threshold as before while sharing the
/// coordinate system used by training measurements and future predictive
/// hooks.
fn memory_limit_hook<L, N, D>(
    heap: RunHeap,
) -> impl FnMut(&mut Runner<L, N, D>) -> Result<(), String> + 'static
where
    L: Language,
    N: Analysis<L>,
    D: IterationData<L, N>,
{
    move |_runner| {
        if let Some(relative_limit) = heap.relative_limit() {
            let current = heap.current();
            if current.relative > relative_limit {
                let absolute_limit = heap
                    .absolute_limit
                    .expect("relative memory limit requires absolute limit");
                return Err(format!(
                    "memory limit exceeded ({} > {absolute_limit} bytes)",
                    current.absolute
                ));
            }
        }
        Ok(())
    }
}

/// Per-iteration heap annotation egg stores in each [`Iteration`]'s `data` slot.
/// egg calls [`IterationData::make`] at the end of `Runner::run_one`.
/// `allocated` therefore reflects the heap after that iteration's rewrites,
/// while egg's `egraph_nodes`/`egraph_classes` fields describe its start.
/// When a hook stops an iteration before any rewrites run, this reading is
/// effectively the heap value observed by the hook.
///
/// The reading is the *absolute* [`live_heap_bytes`] value because egg's
/// `IterationData::make` has no user-state parameter. [`Measurement::from_run`]
/// rebases it exactly once with the same [`RunHeap`] supplied to the hooks.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct HeapData {
    pub allocated: u64,
}

impl<L: Language, N: Analysis<L>> IterationData<L, N> for HeapData {
    fn make(_runner: &Runner<L, N, Self>) -> Self {
        Self {
            allocated: live_heap_bytes(),
        }
    }
}

/// Per-iteration eqsat stats plus live-heap use, as produced by running a
/// [`Runner`] with [`HeapData`] in its iteration-data slot (via
/// `EqsatConfig::build_runner::<_, _, HeapData>`). Each iteration's `allocated`
/// is live-heap growth over a pre-eqsat baseline once passed through
/// [`Measurement::from_run`]. `total_allocated` is the *true* post-run live-heap
/// delta: [`live_heap_bytes`] sampled the moment the run returns, minus the same
/// pre-eqsat baseline — the final egraph's footprint, not an iteration-boundary
/// snapshot. `memory_limit` is the configured ceiling rebased to that baseline,
/// so analysis can compare it directly with `allocated`.
#[derive(Debug, Serialize)]
pub struct Measurement {
    pub iterations: Vec<Iteration<HeapData>>,
    pub total_allocated: u64,
    pub memory_limit: Option<u64>,
}

impl Measurement {
    /// Assemble a measurement from a finished runner's `iterations`, rebasing
    /// each iteration's absolute [`HeapData::allocated`] reading to growth over
    /// `heap`'s pre-eqsat baseline (the [`live_heap_bytes`] value captured
    /// inside `build_runner` before Runner construction), saturating at zero,
    /// and recording the true
    /// post-run delta in `total_allocated` by sampling live-heap now.
    ///
    /// Call this immediately after the run returns and before the egraph is
    /// dropped, so `total_allocated` still reflects the final egraph's live
    /// allocations.
    #[must_use]
    pub fn from_run(heap: RunHeap, mut iterations: Vec<Iteration<HeapData>>) -> Self {
        for iter in &mut iterations {
            iter.data.allocated = heap.relative_to(iter.data.allocated);
        }
        Self {
            iterations,
            total_allocated: heap.current_relative(),
            memory_limit: heap.relative_limit(),
        }
    }
}

/// Run equality saturation up to `config` maximums and return the
/// final egraph (`curr`) together with the last meaningfully different
/// earlier egraph (`prev`).
///
/// Returns `None` if fewer than 3 iterations completed or if the
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

    // Capture exactly once, immediately before Runner construction. In
    // particular, with_expr's initial egraph allocations are run-relative.
    let heap = RunHeap::start(config.max_memory);
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
        })
        .with_hook(memory_limit_hook(heap));

    runner = if config.backoff_scheduler {
        runner.with_scheduler(BackoffScheduler::default())
    } else {
        runner.with_scheduler(SimpleScheduler)
    }
    .run(rules);

    let stop_reason = runner.stop_reason.unwrap();

    let root = runner.roots[0];
    // Drop hook closures so the slot's Rc has only our local clone left.
    runner.hooks.clear();
    let iter_data = runner.iterations;
    let mut curr = runner.egraph;

    if iter_data.len() < MIN_ITERS {
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
        eprintln!("Egraph never produced a distinct earlier state");
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
        heap,
    })
}

/// What `verify_reachability` searches for. Either a single concrete program
/// (`Expr`, checked with `lookup_expr`) or a set of sketches that must *all*
/// be satisfied by the (canonical) root e-class (`Sketches`, checked with
/// [`eclass_contains`]). The guide/goal binaries use `Expr`; the `mini_rise`
/// tile searches use `Sketches`.
#[derive(Clone)]
pub enum Goal<L: MyLanguage> {
    Expr(RecExpr<L>),
    Sketches(Sketch<L>),
}

impl<L: MyLanguage> Goal<L> {
    /// True once this goal is reached in `egraph`. `root` must be the canonical
    /// id of the unioned guide root. For `Sketches` the egraph must be clean
    /// (rebuilt); [`eclass_contains`] asserts this.
    fn reached<N: Analysis<L>>(&self, egraph: &EGraph<L, N>, root: Id) -> bool {
        match self {
            Goal::Expr(e) => egraph
                .lookup_expr(e)
                .is_some_and(|e| egraph.find(e) == root),
            Goal::Sketches(sketch) => {
                let root = egraph.find(root);
                sketch::eclass_contains(sketch, egraph, root)
            }
        }
    }

    fn extract<N: Analysis<L>>(&self, egraph: &EGraph<L, N>, root: Id) -> Option<RecExpr<L>> {
        self.reached(egraph, root).then_some({
            match self {
                Goal::Expr(rec_expr) => rec_expr.clone(),
                Goal::Sketches(sketch) => sketch::eclass_extract(sketch, AstSize, egraph, root)?.1,
            }
        })
    }
}

/// A reached run's outputs from [`verify_reachability`]: the per-iteration log,
/// the extracted goal term, and the *true* final egraph size (`nodes`/`classes`
/// read off the rebuilt egraph after the run, not an iteration-boundary
/// snapshot).
pub struct ReachedRun<L: MyLanguage> {
    pub iterations: Vec<egg::Iteration<()>>,
    pub target: RecExpr<L>,
    pub nodes: usize,
    pub classes: usize,
    /// Final live allocation relative to the baseline captured before this
    /// run's Runner was constructed.
    pub allocated: u64,
}

/// Run eqsat from `guides` (all unioned together) and check if `goal` becomes reachable.
/// Returns a [`ReachedRun`] if reached, an error otherwise.
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
    goal: &Goal<L>,
    rules: &[Rewrite<L, N>],
    eqsat: &EqsatConfig,
    full_union: bool,
) -> Result<ReachedRun<L>, GuideError>
where
    L: MyLanguage + 'static,
    N: MyAnalysis<L> + Default,
{
    assert!(!guides.is_empty(), "must have at least one guide");
    let goal_clone = goal.clone();

    // Capture exactly once, immediately before Runner construction. Adding and
    // unioning guide expressions below is therefore included in this run.
    let heap = RunHeap::start(eqsat.max_memory);
    let mut runner = Runner::default()
        .with_time_limit(Duration::try_from_secs_f64(eqsat.max_time).unwrap_or(Duration::MAX))
        .with_node_limit(eqsat.max_nodes)
        .with_iter_limit(eqsat.max_iters)
        .with_hook(move |runner| {
            let root = runner.roots[0];
            if goal_clone.reached(&runner.egraph, root) {
                return Err("goal found".to_owned());
            }
            Ok(())
        })
        .with_hook(memory_limit_hook(heap));

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

    let Ok(mut r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules)))
    else {
        eprintln!("Panic caught verify_reachability for guides: {guides:?}");
        return Err(GuideError::PanicWhileAttempt);
    };
    r.egraph.rebuild();
    let root = r.roots[0];
    if let Some(target) = goal.extract(&r.egraph, root) {
        Ok(ReachedRun {
            iterations: r.iterations,
            target,
            nodes: r.egraph.total_number_of_nodes(),
            classes: r.egraph.classes().len(),
            allocated: heap.current_relative(),
        })
    } else {
        Err(GuideError::Unreached(r.stop_reason.clone().unwrap()))
    }
}

fn add_with_root_union<'a, L, N, D, I>(mut runner: Runner<L, N, D>, guides: I) -> Runner<L, N, D>
where
    L: MyLanguage + 'a,
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
/// effect.
/// The canonical ids in `a` and `b` agree on every class, and the
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Mutex;

    use egg::{RecExpr, StopReason};

    use super::{
        EqsatConfig, Goal, GuideError, HeapData, Measurement, RunHeap, live_heap_bytes,
        verify_reachability,
    };
    use crate::OriginLang;
    use crate::langs::math::{self, ConstantFold, Math};

    // jemalloc stats are process-wide, so keep allocation-sensitive tests from
    // perturbing one another.
    static HEAP_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn verify_reachability_enforces_memory_limit() {
        let guide: RecExpr<OriginLang<Math>> = "x".parse().unwrap();
        let goal = Goal::Expr("definitely_unreachable".parse().unwrap());
        let config = EqsatConfig {
            max_iters: 100,
            max_nodes: usize::MAX,
            max_time: 60.0,
            max_memory: Some(0),
            backoff_scheduler: true,
        };

        let result = verify_reachability::<Math, ConstantFold>(
            &[guide],
            &goal,
            &math::rules(),
            &config,
            false,
        );
        assert!(matches!(
            result,
            Err(GuideError::Unreached(StopReason::Other(message)))
                if message.contains("memory limit exceeded")
        ));
    }

    #[test]
    fn measurement_rebases_memory_limit() {
        let heap = RunHeap::from_baseline(10_000, Some(14_096));
        let measurement = Measurement::from_run(heap, Vec::new());
        assert_eq!(measurement.memory_limit, Some(4096));
    }

    #[test]
    fn pre_baseline_allocations_are_excluded_and_post_baseline_allocations_are_included() {
        const BYTES: usize = 32 * 1024 * 1024;
        let _guard = HEAP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let mut before_baseline = vec![0_u8; BYTES];
        for byte in before_baseline.iter_mut().step_by(4096) {
            *byte = 1;
        }
        let heap = RunHeap::start(None);
        let before_post_allocation = heap.current_relative();

        let mut after_baseline = vec![0_u8; BYTES];
        for byte in after_baseline.iter_mut().step_by(4096) {
            *byte = 1;
        }
        std::hint::black_box((&before_baseline, &after_baseline));
        let after_post_allocation = heap.current_relative();

        assert!(
            before_post_allocation < BYTES as u64 / 2,
            "allocation made before baseline leaked into relative reading: \
             {before_post_allocation}"
        );
        assert!(
            after_post_allocation >= BYTES as u64 / 2,
            "allocation made after baseline was not included: {after_post_allocation}"
        );
    }

    #[test]
    fn hard_limit_has_the_same_absolute_threshold_after_rebasing() {
        let heap = RunHeap::from_baseline(1_000, Some(1_500));
        for absolute in [999, 1_000, 1_499, 1_500, 1_501, u64::MAX] {
            assert_eq!(
                heap.relative_to(absolute) > heap.relative_limit().unwrap(),
                absolute > 1_500,
                "threshold changed at absolute live heap {absolute}"
            );
        }
    }

    #[test]
    fn measurement_and_two_hooks_share_the_runner_baseline() {
        let _guard = HEAP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let expr: RecExpr<Math> = "(+ x 0)".parse().unwrap();
        let config = EqsatConfig {
            max_iters: 1,
            max_nodes: usize::MAX,
            max_time: 60.0,
            max_memory: None,
            backoff_scheduler: false,
        };
        let (runner, heap) = config.build_runner::<_, ConstantFold, HeapData>(&expr);
        // Give both hooks the same absolute sample so the assertion tests their
        // shared coordinate system, independent of allocations on other test
        // threads between hook calls.
        let shared_absolute_sample = live_heap_bytes();
        let first_baseline = Rc::new(Cell::new(None));
        let second_baseline = Rc::new(Cell::new(None));
        let first_relative = Rc::new(Cell::new(None));
        let second_relative = Rc::new(Cell::new(None));

        let runner = {
            let baseline = Rc::clone(&first_baseline);
            let relative = Rc::clone(&first_relative);
            runner.with_hook(move |_| {
                baseline.set(Some(heap.baseline()));
                relative.set(Some(heap.relative_to(shared_absolute_sample)));
                Ok(())
            })
        };
        let runner = {
            let baseline = Rc::clone(&second_baseline);
            let relative = Rc::clone(&second_relative);
            runner.with_hook(move |_| {
                baseline.set(Some(heap.baseline()));
                relative.set(Some(heap.relative_to(shared_absolute_sample)));
                Ok(())
            })
        };

        let runner = runner.run(&[]);
        let absolute_iteration_reading = runner.iterations[0].data.allocated;
        let measurement = Measurement::from_run(heap, runner.iterations);

        assert_eq!(first_baseline.get(), Some(heap.baseline()));
        assert_eq!(second_baseline.get(), Some(heap.baseline()));
        assert_eq!(first_relative.get(), second_relative.get());
        assert_eq!(
            measurement.iterations[0].data.allocated,
            heap.relative_to(absolute_iteration_reading)
        );
        assert_eq!(measurement.memory_limit, heap.relative_limit());
    }

    #[test]
    fn runner_and_with_expr_allocations_are_after_the_baseline() {
        let _guard = HEAP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut expr = RecExpr::<Math>::default();
        let mut root = expr.add(Math::Symbol("x".into()));
        for _ in 0..50_000 {
            root = expr.add(Math::Sin(root));
        }
        std::hint::black_box(root);

        let config = EqsatConfig {
            max_iters: 1,
            max_nodes: usize::MAX,
            max_time: 60.0,
            max_memory: None,
            backoff_scheduler: false,
        };
        let (runner, heap) = config.build_runner::<_, ConstantFold, ()>(&expr);
        std::hint::black_box(&runner);

        assert!(
            heap.current_relative() > 1024 * 1024,
            "Runner construction and with_expr allocation were not measured"
        );
    }

    /// `live_heap_bytes` must track live allocations: a large touched buffer
    /// shows up while alive and is gone once freed. This is the property the
    /// per-term memory ceiling relies on (no `malloc_trim` purge needed). Only
    /// meaningful when jemalloc is the active allocator; the test crate installs
    /// it as its global allocator.
    #[test]
    fn live_heap_tracks_allocations() {
        use super::live_heap_bytes;

        const BYTES: usize = 512 * 1024 * 1024; // 512 MiB
        let _guard = HEAP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let before = live_heap_bytes();

        // Touch every page so the allocation is real, then drop it.
        let mut buf = vec![0u8; BYTES];
        for i in (0..BYTES).step_by(4096) {
            buf[i] = 1;
        }
        std::hint::black_box(&buf);
        let peak = live_heap_bytes();
        drop(buf);

        let after = live_heap_bytes();

        // The buffer clearly showed up in the live-heap reading...
        assert!(
            peak >= before + BYTES as u64 / 2,
            "live heap did not grow with the buffer: before {before}, peak {peak}"
        );
        // ...and dropping it released most of it (allow slack for other allocs).
        assert!(
            after < peak - BYTES as u64 / 2,
            "live heap did not drop after free: peak {peak}, after {after}"
        );
    }
}
