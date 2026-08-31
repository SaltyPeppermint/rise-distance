use std::time::Duration;

use egg::{
    Analysis, AstSize, BackoffScheduler, EGraph, Id, Iteration, IterationData, Language,
    MemorySamplePhase, RecExpr, Rewrite, Runner, SchedulerSnapshot, StopReason,
};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use strum::Display;
use thiserror::Error;

use crate::langs::{MyAnalysis, MyLanguage};
use crate::origin::{OriginLang, lower};
use crate::previous::PrevIndex;
use crate::sketch::{self, Sketch};
use crate::utils::live_heap_bytes;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EqsatMetadata {
    pub nodes: usize,
    pub classes: usize,
    pub time: f64,
    pub iters: usize,
}

#[derive(Debug, Error, Display, Serialize, Clone)]
pub enum GuideError {
    Unreached {
        stop_reason: StopReason,
        final_allocated: u64,
        peak_allocated: u64,
    },
    PanicWhileAttempt,
}

impl EqsatMetadata {
    /// Summarize an iteration log. Sizes come from its final entry and `iters`
    /// is the last applied iteration index (`len() - 1`).
    ///
    /// # Panics
    ///
    /// Panics if `iterations` is empty (a runner always logs at least one).
    #[must_use]
    pub fn from_iterations<D>(iterations: &[Iteration<D>]) -> Self {
        let last = iterations.last().expect("eqsat run logged no iterations");
        Self {
            nodes: last.egraph_nodes,
            classes: last.egraph_classes,
            time: iterations.iter().map(|i| i.total_time).sum(),
            iters: iterations.len() - 1,
        }
    }
}

/// Eqsat limits and shared `--max-*` CLI arguments.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, clap::Args)]
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

    /// Process live-heap ceiling in bytes, or unbounded when unset.
    #[serde(default)]
    #[arg(long)]
    pub max_memory: Option<u64>,
}

impl EqsatConfig {
    /// Build a [`Runner`] with these limits, memory tracking, and scheduler.
    #[must_use]
    pub fn build_runner<L, N, D>(&self, expr: &RecExpr<L>) -> Runner<L, N, D>
    where
        L: MyLanguage,
        N: MyAnalysis<L>,
        D: IterationData<L, N>,
    {
        Runner::<L, N, D>::new_with_memory_tracker(N::default(), live_heap_bytes, self.max_memory)
            .with_expr(expr)
            .with_iter_limit(self.max_iters)
            .with_node_limit(self.max_nodes)
            .with_time_limit(Duration::from_secs_f64(self.max_time))
            .with_scheduler(BackoffScheduler::default())
    }
}

/// An eqsat run's final e-graph, previous-boundary marker, and metadata.
/// The stored root may become noncanonical after unions.
pub struct EqsatResult<L, N>
where
    L: Language,
    N: Analysis<L>,
{
    iter_data: Vec<Iteration<Boundary>>,
    prev_boundary: Boundary,
    curr: EGraph<L, N>,
    root: Id,
    stop_reason: StopReason,
    allocated: u64,
    peak_allocated: u64,
}

impl<L, N> EqsatResult<L, N>
where
    L: Language,
    N: Analysis<L>,
{
    /// Construct a test result with a known previous-boundary marker.
    #[cfg(test)]
    pub(crate) fn new_for_tests(
        curr: EGraph<L, N>,
        root: Id,
        prev_raw_node_count: usize,
        prev_union_event_count: usize,
    ) -> Self {
        Self {
            iter_data: Vec::new(),
            prev_boundary: Boundary {
                raw_node_count: prev_raw_node_count,
                union_event_count: prev_union_event_count,
            },
            curr,
            root,
            stop_reason: StopReason::Saturated,
            allocated: 0,
            peak_allocated: 0,
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

    /// Reconstruct the previous-boundary index without mutating the final graph.
    pub(crate) fn prev_index(&self) -> PrevIndex<L> {
        PrevIndex::from_union_history(
            self.curr.nodes(),
            self.prev_boundary.raw_node_count,
            self.prev_boundary.union_event_count,
            self.curr.union_events(),
        )
    }

    /// Per-iteration metadata, starting with the initial e-graph iteration.
    #[must_use]
    pub fn data(&self) -> &[Iteration<Boundary>] {
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

    /// Final process live heap while the final e-graph was alive.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    /// Peak process live heap during the run.
    #[must_use]
    pub fn peak_allocated(&self) -> u64 {
        self.peak_allocated
    }

    /// Consume the result and return the final egraph together with the root id.
    #[must_use]
    pub fn into_curr(self) -> (EGraph<L, N>, Id) {
        (self.curr, self.root)
    }

    /// Summarize the run using the rebuilt final e-graph's size.
    #[must_use]
    pub fn metadata(&self) -> EqsatMetadata {
        EqsatMetadata {
            nodes: self.curr.total_number_of_nodes(),
            classes: self.curr.classes().len(),
            time: self.iter_data.iter().map(|i| i.total_time).sum(),
            iters: self.iters(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Boundary {
    raw_node_count: usize,
    union_event_count: usize,
}

impl Boundary {
    fn of<L: Language, N: Analysis<L>>(egraph: &EGraph<L, N>) -> Self {
        Self {
            raw_node_count: egraph.nodes().len(),
            union_event_count: egraph.union_event_count(),
        }
    }
}

/// Recorded at the end of each iteration, after apply and rebuild, so every
/// logged boundary describes a clean e-graph.
impl<L: Language, N: Analysis<L>> IterationData<L, N> for Boundary {
    fn make(runner: &Runner<L, N, Self>) -> Self {
        debug_assert!(runner.egraph.clean, "iteration boundary must be clean");
        Self::of(&runner.egraph)
    }
}

/// Latest two distinct topology boundaries, stored as history cursors.
#[derive(Debug, Default)]
struct FrontierHistory {
    latest_distinct: Option<Boundary>,
    previous_distinct: Option<Boundary>,
}

impl FrontierHistory {
    fn observe(&mut self, boundary: Boundary) {
        if self.latest_distinct != Some(boundary) {
            self.previous_distinct = self.latest_distinct;
            self.latest_distinct = Some(boundary);
        }
    }
}

/// Minimum iterations needed for a meaningful guide/goal split.
const MIN_ITERS: usize = 3;

/// Per-iteration scheduler and live-heap telemetry.
///
/// Start fields describe the pre-search boundary; `allocated` is the final
/// reading and `iteration_peak_allocated` includes transient peaks.
#[derive(Debug, Clone, Serialize)]
pub struct HeapData {
    /// Final live heap after rebuild.
    pub allocated: u64,
    pub iteration_start_allocated: u64,
    pub iteration_peak_allocated: u64,
    pub iteration_peak_phase: MemorySamplePhase,
    pub iteration_peak_rule: Option<egg::Symbol>,
    /// Scheduler state before hooks and search.
    pub scheduler: SchedulerSnapshot,
}

impl<L: Language, N: Analysis<L>> IterationData<L, N> for HeapData {
    fn make(runner: &Runner<L, N, Self>) -> Self {
        let peak = runner
            .iteration_memory_peak()
            .expect("configured measurement runner has memory tracking");
        Self {
            allocated: runner
                .memory_reading()
                .expect("configured measurement runner has memory tracking"),
            iteration_start_allocated: peak.iteration_start_allocated,
            iteration_peak_allocated: peak.iteration_peak_allocated,
            iteration_peak_phase: peak.peak_phase,
            iteration_peak_rule: peak.peak_rule,
            scheduler: runner.scheduler_snapshot.clone(),
        }
    }
}

/// Run eqsat and retain the last distinct boundary before the final e-graph.
/// Returns `None` when too few iterations or distinct states exist.
///
/// # Panics
///
/// Panics if the runner omits its required stop reason.
pub fn run_eqsat<'a, L, N, R>(
    start: &RecExpr<L>,
    rules: R,
    config: &EqsatConfig,
) -> Option<EqsatResult<L, N>>
where
    L: MyLanguage + 'static,
    N: MyAnalysis<L> + Default + 'static,
    R: IntoIterator<Item = &'a Rewrite<L, N>>,
{
    // Analysis hooks may union while the initial expression is inserted, so
    // recording must precede `with_expr`.
    let mut runner = Runner::<L, N, Boundary>::new_with_memory_tracker(
        N::default(),
        live_heap_bytes,
        config.max_memory,
    )
    .with_time_limit(Duration::try_from_secs_f64(config.max_time).unwrap_or(Duration::MAX))
    .with_node_limit(config.max_nodes)
    .with_iter_limit(config.max_iters)
    .with_union_event_recording()
    .with_expr(start)
    .with_scheduler(BackoffScheduler::default());

    // The pre-search state is never an iteration, so record it by hand. `run`
    // rebuilds before its first iteration; do it here so this boundary is
    // clean and its congruence unions are already in the log.
    runner.egraph.rebuild();
    let mut history = FrontierHistory::default();
    history.observe(Boundary::of(&runner.egraph));

    runner = runner.run(rules);

    let memory = runner
        .final_memory_report()
        .expect("configured eqsat runner has final memory report");
    let stop_reason = runner.stop_reason.unwrap();

    let root = runner.roots[0];
    let iter_data = runner.iterations;
    let curr = runner.egraph;

    if iter_data.len() < MIN_ITERS {
        return None;
    }

    for iteration in &iter_data {
        history.observe(iteration.data);
    }

    // Every iteration rebuilds before recording its data, so the last logged
    // boundary already describes the final e-graph.
    debug_assert!(curr.clean, "a finished run leaves a clean e-graph");
    let final_boundary = history
        .latest_distinct
        .expect("history was seeded with the initial boundary");
    debug_assert_eq!(
        final_boundary,
        Boundary::of(&curr),
        "last logged boundary must match the final e-graph"
    );

    let Some(prev_boundary) = history.previous_distinct else {
        eprintln!("Egraph never produced a distinct earlier state");
        return None;
    };

    assert_ne!(
        prev_boundary, final_boundary,
        "previous and final boundary must be distinct"
    );
    Some(EqsatResult {
        iter_data,
        prev_boundary,
        curr,
        root,
        stop_reason,
        allocated: memory.final_reading,
        peak_allocated: memory.peak_reading,
    })
}

/// A concrete expression or a set of sketches to reach at the root.
#[derive(Clone)]
pub enum Goal<L: MyLanguage> {
    Expr(RecExpr<L>),
    Sketches(Sketch<L>),
}

impl<L: MyLanguage> Goal<L> {
    /// Whether the canonical root reaches this goal.
    /// Sketch lookup requires a rebuilt e-graph.
    fn reached<N: Analysis<L>>(&self, egraph: &EGraph<L, N>, root: Id) -> bool {
        let root = egraph.find(root);
        match self {
            Goal::Expr(e) => egraph
                .lookup_expr(e)
                .is_some_and(|e| egraph.find(e) == root),
            Goal::Sketches(sketch) => sketch::eclass_contains(sketch, egraph, root),
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

/// Outputs from a reached run, including the rebuilt final e-graph size.
#[derive(Serialize)]
pub struct ReachedRun<L: MyLanguage> {
    pub iterations: Vec<egg::Iteration<()>>,
    pub target: RecExpr<L>,
    pub nodes: usize,
    pub classes: usize,
    /// Final process live heap.
    pub allocated: u64,
    /// Peak process live heap.
    pub peak_allocated: u64,
}

/// Run eqsat from unioned `guides` and check whether `goal` is reached.
/// Returns a [`ReachedRun`] if reached, an error otherwise.
///
/// # Errors
///
/// Returns [`GuideError`] if the goal is unreached or the run panics.
///
/// # Panics
///
/// Panics if `guides` is empty.
pub fn guided_eqsat<L, N>(
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

    let runner = Runner::new_with_memory_tracker(N::default(), live_heap_bytes, eqsat.max_memory)
        .with_time_limit(Duration::try_from_secs_f64(eqsat.max_time).unwrap_or(Duration::MAX))
        .with_node_limit(eqsat.max_nodes)
        .with_iter_limit(eqsat.max_iters)
        .with_scheduler(BackoffScheduler::default());

    let mut runner = if full_union {
        add_with_full_union(runner, guides)
    } else {
        add_with_root_union(runner, guides)
    };

    runner.egraph.rebuild();

    run_until_goal(runner, goal, rules, format_args!("guides: {guides:?}"))
}

/// Run single-seed eqsat until `goal` is reached or a limit stops the run.
#[expect(clippy::missing_errors_doc)]
pub fn unguided_eqsat<L, N>(
    start: &RecExpr<L>,
    goal: &Goal<L>,
    rules: &[Rewrite<L, N>],
    eqsat: &EqsatConfig,
) -> Result<ReachedRun<L>, GuideError>
where
    L: MyLanguage + 'static,
    N: MyAnalysis<L> + Default,
{
    let runner = eqsat.build_runner::<L, N, ()>(start);
    run_until_goal(runner, goal, rules, format_args!("start term: {start:?}"))
}

/// Run `rules` on a prepared `runner` until `goal` is reached or a limit stops
/// the run, then report the final e-graph.
fn run_until_goal<L, N>(
    mut runner: Runner<L, N, ()>,
    goal: &Goal<L>,
    rules: &[Rewrite<L, N>],
    context: std::fmt::Arguments,
) -> Result<ReachedRun<L>, GuideError>
where
    L: MyLanguage + 'static,
    N: MyAnalysis<L>,
{
    let goal_clone = goal.clone();
    runner.hooks.insert(
        0,
        Box::new(move |r: &mut Runner<L, N, ()>| {
            let root = r.roots[0];
            if goal_clone.reached(&r.egraph, root) {
                return Err("goal found".to_owned());
            }
            Ok(())
        }),
    );

    let Ok(mut runner) =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules)))
    else {
        eprintln!("Panic caught during verification for {context}");
        return Err(GuideError::PanicWhileAttempt);
    };
    let memory = runner
        .final_memory_report()
        .expect("configured runner has final memory report");
    runner.egraph.rebuild();
    let root = runner.roots[0];
    if let Some(target) = goal.extract(&runner.egraph, root) {
        Ok(ReachedRun {
            iterations: runner.iterations,
            target,
            nodes: runner.egraph.total_number_of_nodes(),
            classes: runner.egraph.classes().len(),
            allocated: memory.final_reading,
            peak_allocated: memory.peak_reading,
        })
    } else {
        Err(GuideError::Unreached {
            stop_reason: runner.stop_reason.clone().unwrap(),
            final_allocated: memory.final_reading,
            peak_allocated: memory.peak_reading,
        })
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use clap::Parser;
    use egg::{RecExpr, StopReason};

    use super::{EqsatConfig, Goal, GuideError, guided_eqsat, run_eqsat, unguided_eqsat};
    use crate::OriginLang;
    use crate::langs::math::{self, ConstantFold, Math};
    use crate::previous::PreviousLookup;
    use crate::utils::live_heap_bytes;
    use crate::utils::sym;

    // jemalloc stats are process-wide, so keep allocation-sensitive tests from
    // perturbing one another.
    static HEAP_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        eqsat: EqsatConfig,
    }

    #[test]
    fn run_eqsat_reconstructs_the_boundary_before_trailing_noop() {
        let start: RecExpr<Math> = "a".parse().unwrap();
        let rules = [
            egg::rewrite!("a-to-b"; "a" => "b"),
            egg::rewrite!("b-to-c"; "b" => "c"),
        ];
        let config = EqsatConfig {
            max_iters: 10,
            max_nodes: 100,
            max_time: 60.0,
            max_memory: None,
        };

        let result = run_eqsat::<Math, (), _>(&start, rules.iter(), &config)
            .expect("three boundary states should produce a previous index");
        let union_event_count = result.curr().union_event_count();
        let prev = result.prev_index();
        assert_eq!(
            result.curr().union_event_count(),
            union_event_count,
            "reconstructing the previous index must not drain the final e-graph"
        );

        let a = prev.lookup(sym("a")).expect("a existed at the boundary");
        let b = prev.lookup(sym("b")).expect("b existed at the boundary");
        assert_eq!(a, b, "a and b were already unioned at the boundary");
        assert!(
            prev.lookup(sym("c")).is_none(),
            "c was added only in the final distinct state"
        );

        let prev_again = result.prev_index();
        assert!(prev_again.lookup(sym("a")).is_some());
        assert!(prev_again.lookup(sym("c")).is_none());
    }

    #[test]
    fn verify_reachability_enforces_memory_limit() {
        let guide: RecExpr<OriginLang<Math>> = "x".parse().unwrap();
        let goal = Goal::Expr("definitely_unreachable".parse().unwrap());
        let config = EqsatConfig {
            max_iters: 100,
            max_nodes: usize::MAX,
            max_time: 60.0,
            max_memory: Some(0),
        };

        let result =
            guided_eqsat::<Math, ConstantFold>(&[guide], &goal, &math::rules(), &config, false);
        assert!(matches!(
            result,
            Err(GuideError::Unreached {
                stop_reason: StopReason::MemoryLimit(observed),
                ..
            }) if observed > 0
        ));
    }

    #[test]
    fn unguided_reachability_canonicalizes_a_stale_root() {
        let seed = "(+ x 0)".parse().unwrap();
        let goal = Goal::Expr("x".parse().unwrap());
        let config = EqsatConfig {
            max_iters: 10,
            max_nodes: 10_000,
            max_time: 60.0,
            max_memory: None,
        };

        let result = unguided_eqsat::<Math, ConstantFold>(&seed, &goal, &math::rules(), &config);
        assert!(
            result.is_ok(),
            "(+ x 0) and x share the root e-class even after x becomes its representative"
        );
    }

    #[test]
    fn memory_limit_stop_reason_has_dedicated_json_variant() {
        assert_eq!(
            serde_json::to_value(StopReason::MemoryLimit(123)).unwrap(),
            serde_json::json!({"MemoryLimit": 123})
        );
    }

    /// Readings include heap allocated before the runner.
    #[test]
    fn readings_include_heap_held_before_the_runner_existed() {
        const BYTES: usize = 32 * 1024 * 1024;
        let _guard = HEAP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let mut held = vec![0_u8; BYTES];
        for byte in held.iter_mut().step_by(4096) {
            *byte = 1;
        }
        let runner = egg::Runner::<Math, ConstantFold>::new_with_memory_tracker(
            ConstantFold,
            live_heap_bytes,
            None,
        );
        std::hint::black_box(&held);
        let reading = runner.sample_memory().unwrap();

        assert!(
            reading >= BYTES as u64,
            "pre-existing heap was excluded from the absolute reading: {reading}"
        );
    }

    #[test]
    fn runner_and_with_expr_allocations_are_measured() {
        let _guard = HEAP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut expr = RecExpr::<Math>::default();
        let mut root = expr.add(Math::Symbol("x".into()));
        for _ in 0..50_000 {
            root = expr.add(Math::Sin(root));
        }
        std::hint::black_box(root);

        let before = live_heap_bytes();
        let config = EqsatConfig {
            max_iters: 1,
            max_nodes: usize::MAX,
            max_time: 60.0,
            max_memory: None,
        };
        let runner = config.build_runner::<_, ConstantFold, ()>(&expr);
        std::hint::black_box(&runner);

        let reading = runner.sample_memory().unwrap();
        assert!(
            reading.saturating_sub(before) > 1024 * 1024,
            "Runner construction and with_expr allocation were not measured: \
             {before} -> {reading}"
        );
    }

    /// `live_heap_bytes` tracks allocation and release under jemalloc.
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
