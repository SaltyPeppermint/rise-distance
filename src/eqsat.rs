use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use egg::{
    Analysis, AstSize, BackoffScheduler, EGraph, Id, Iteration, IterationData, Language,
    MemoryReport, MemorySamplePhase, RecExpr, Rewrite, Runner, SchedulerSnapshot, StopReason,
};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use strum::Display;
use thiserror::Error;

use crate::langs::{MyAnalysis, MyLanguage};
use crate::origin::{OriginLang, lower};
use crate::predictive_memory;
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
    Unreached(StopReason),
    PanicWhileAttempt,
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

/// Eqsat resource limits. Doubles as the shared clap flag group (`--max-*`) for
/// the `goal` / `sample` / `verify` binaries; the Python drivers read the values out of the
/// `generation_args.json` / `goal_args.json` sidecars and forward them on
/// argv.
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

    /// Absolute process live-heap ceiling in bytes (jemalloc
    /// `stats.allocated`), enforced at egg's limit-check boundaries.
    /// `None` (flag unset) = unbounded.
    #[serde(default)]
    #[arg(long)]
    pub max_memory: Option<u64>,

    /// Path to an ONNX model used to predict whether the next eqsat iteration
    /// will cross `max_memory`. The adjacent same-stem JSON manifest supplies
    /// the feature order and safety margin. Disabled when unset and ignored
    /// unless `max_memory` is set.
    #[serde(default)]
    #[arg(long, value_name = "MODEL.onnx")]
    pub predict_next_memory: Option<PathBuf>,
}

impl EqsatConfig {
    /// Build a [`Runner`] configured with this config's limits, process-memory
    /// tracker, and scheduler.
    ///
    /// # Panics
    ///
    /// Panics if the requested predictive ONNX model or its manifest cannot be
    /// loaded, validated, or prewarmed.
    #[must_use]
    pub fn build_runner<L, N, D>(&self, expr: &RecExpr<L>) -> Runner<L, N, D>
    where
        L: MyLanguage,
        N: MyAnalysis<L>,
        D: IterationData<L, N>,
    {
        let predictor = self
            .max_memory
            .and(self.predict_next_memory.as_deref())
            .map(|model_path| {
                predictive_memory::OnnxMemoryGrowthPredictor::load_and_prewarm(model_path)
                    .expect("failed to initialize predictive memory model")
            });
        // Load the model before the Runner exists so the ONNX session's own
        // allocation is not mistaken for eqsat growth mid-run.
        let term_size = expr.as_ref().len();
        let runner = Runner::<L, N, D>::new_with_memory_tracker(
            N::default(),
            live_heap_bytes,
            self.max_memory,
        )
        .with_expr(expr)
        .with_iter_limit(self.max_iters)
        .with_node_limit(self.max_nodes)
        .with_time_limit(Duration::from_secs_f64(self.max_time))
        .with_scheduler(BackoffScheduler::default());
        if let Some(predictor) = predictor {
            runner.with_hook(predictive_memory::hook(term_size, predictor))
        } else {
            runner
        }
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
    allocated: u64,
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
            allocated: 0,
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

    /// Final absolute process live heap sampled while the runner and its
    /// final egraph were alive.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        self.allocated
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

/// Per-iteration decision-point, scheduler, and peak-memory telemetry.
///
/// `iteration_start_allocated`, egg's egraph counts, and `scheduler` all
/// describe the instant immediately before this iteration's hooks and search.
/// The previous [`Iteration`]'s timings and applied counts are therefore the
/// work history available to an online predictor at this boundary.
///
/// `allocated` remains the final live reading after rebuild, while
/// `iteration_peak_allocated` retains transient search/application peaks even
/// if their allocations were freed before the iteration ended.
#[derive(Debug, Clone, Serialize)]
pub struct HeapData {
    /// Final live allocation sampled after rebuild/finalization.
    pub allocated: u64,
    pub iteration_start_allocated: u64,
    pub iteration_peak_allocated: u64,
    pub iteration_peak_phase: MemorySamplePhase,
    pub iteration_peak_rule: Option<egg::Symbol>,
    /// Exact upcoming state captured before hooks and rewrite search.
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

/// Per-iteration eqsat stats plus live-heap use, as produced by running a
/// [`Runner`] with [`HeapData`] in its iteration-data slot (via
/// `EqsatConfig::build_runner::<_, _, HeapData>`). Every byte count here is an
/// absolute process live-heap figure, in the same coordinate system egg's
/// limit check uses, so readings are directly comparable to `memory_limit`
/// across runs and workers. `total_allocated` is Runner's final sample while
/// the final egraph is alive.
#[derive(Debug, Serialize)]
pub struct Measurement {
    pub iterations: Vec<Iteration<HeapData>>,
    pub total_allocated: u64,
    pub memory_limit: Option<u64>,
}

impl Measurement {
    /// Assemble a measurement from a completed runner's final report and
    /// per-iteration readings.
    #[must_use]
    pub fn from_run(report: MemoryReport, iterations: Vec<Iteration<HeapData>>) -> Self {
        Self {
            iterations,
            total_allocated: report.final_reading,
            memory_limit: report.absolute_limit,
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

    let predictor = config
        .max_memory
        .and(config.predict_next_memory.as_deref())
        .map(|model_path| {
            predictive_memory::OnnxMemoryGrowthPredictor::load_and_prewarm(model_path)
                .expect("failed to initialize predictive memory model")
        });
    let term_size = start.as_ref().len();
    let mut runner =
        Runner::new_with_memory_tracker(N::default(), live_heap_bytes, config.max_memory)
            .with_time_limit(Duration::try_from_secs_f64(config.max_time).unwrap_or(Duration::MAX))
            .with_node_limit(config.max_nodes)
            .with_iter_limit(config.max_iters)
            .with_expr(start)
            .with_scheduler(BackoffScheduler::default())
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
    if let Some(predictor) = predictor {
        runner = runner.with_hook(predictive_memory::hook(term_size, predictor));
    }

    runner = runner.run(rules);

    let allocated = runner
        .final_memory_report()
        .expect("configured eqsat runner has final memory report")
        .final_reading;
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
        allocated,
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
    /// Final absolute process live heap, in the same coordinate system as the
    /// configured memory ceiling.
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

    // Predictive stopping is deliberately disabled here: a multi-guide run
    // has no training-compatible single `term_size`. The hard limit remains.
    // Adding and unioning guide expressions below is included in this run.
    let mut runner =
        Runner::new_with_memory_tracker(N::default(), live_heap_bytes, eqsat.max_memory)
            .with_time_limit(Duration::try_from_secs_f64(eqsat.max_time).unwrap_or(Duration::MAX))
            .with_node_limit(eqsat.max_nodes)
            .with_iter_limit(eqsat.max_iters)
            .with_scheduler(BackoffScheduler::default())
            .with_hook(move |runner| {
                let root = runner.roots[0];
                if goal_clone.reached(&runner.egraph, root) {
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

    let Ok(mut r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runner.run(rules)))
    else {
        eprintln!("Panic caught verify_reachability for guides: {guides:?}");
        return Err(GuideError::PanicWhileAttempt);
    };
    let allocated = r
        .final_memory_report()
        .expect("configured reachability runner has final memory report")
        .final_reading;
    r.egraph.rebuild();
    let root = r.roots[0];
    if let Some(target) = goal.extract(&r.egraph, root) {
        Ok(ReachedRun {
            iterations: r.iterations,
            target,
            nodes: r.egraph.total_number_of_nodes(),
            classes: r.egraph.classes().len(),
            allocated,
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
    use std::path::PathBuf;
    use std::sync::Mutex;

    use clap::Parser;
    use egg::{MemoryReport, RecExpr, StopReason};

    use super::{EqsatConfig, Goal, GuideError, HeapData, Measurement, verify_reachability};
    use crate::OriginLang;
    use crate::langs::math::{self, ConstantFold, Math};
    use crate::utils::live_heap_bytes;

    // jemalloc stats are process-wide, so keep allocation-sensitive tests from
    // perturbing one another.
    static HEAP_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        eqsat: EqsatConfig,
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
            predict_next_memory: None,
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
            Err(GuideError::Unreached(StopReason::MemoryLimit(observed))) if observed > 0
        ));
    }

    #[test]
    fn memory_limit_stop_reason_has_dedicated_json_variant() {
        assert_eq!(
            serde_json::to_value(StopReason::MemoryLimit(123)).unwrap(),
            serde_json::json!({"MemoryLimit": 123})
        );
    }

    #[test]
    fn legacy_backoff_key_is_ignored_and_not_serialized() {
        let config: EqsatConfig = serde_json::from_value(serde_json::json!({
            "max_iters": 10,
            "max_nodes": 1000,
            "max_time": 1.0,
            "max_memory": 1_000_000,
            "backoff_scheduler": false
        }))
        .unwrap();
        assert!(config.predict_next_memory.is_none());
        assert!(
            serde_json::to_value(config)
                .unwrap()
                .get("backoff_scheduler")
                .is_none()
        );
    }

    #[test]
    fn predictive_model_path_is_read_from_the_cli_flag() {
        let args = TestCli::try_parse_from([
            "test",
            "--max-iters",
            "10",
            "--max-nodes",
            "1000",
            "--max-time",
            "1",
            "--max-memory",
            "1000000",
            "--predict-next-memory",
            "models/custom.onnx",
        ])
        .unwrap();
        assert_eq!(
            args.eqsat.predict_next_memory,
            Some(PathBuf::from("models/custom.onnx"))
        );
    }

    /// Reported figures are absolute: the limit is the one the user
    /// configured, so readings from different runs and workers are directly
    /// comparable against a single ceiling.
    #[test]
    fn measurement_reports_absolute_memory_figures() {
        let report = MemoryReport {
            final_reading: 12_000,
            absolute_limit: Some(14_096),
        };
        let measurement = Measurement::from_run(report, Vec::new());
        assert_eq!(measurement.memory_limit, Some(14_096));
        assert_eq!(measurement.total_allocated, 12_000);
    }

    /// Readings are whole-process live heap with nothing subtracted out, so
    /// heap the process already held is counted the same as the run's own.
    /// This is what makes a reading comparable to the configured ceiling.
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
    fn heap_data_and_measurement_agree_on_absolute_readings() {
        let _guard = HEAP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let expr: RecExpr<Math> = "(+ x 0)".parse().unwrap();
        let config = EqsatConfig {
            max_iters: 1,
            max_nodes: usize::MAX,
            max_time: 60.0,
            max_memory: None,
            predict_next_memory: None,
        };
        let runner = config.build_runner::<_, ConstantFold, HeapData>(&expr);
        let runner = runner.run(&[]);
        let report = runner.final_memory_report().unwrap();
        let iteration_allocated = runner.iterations[0].data.allocated;
        let measurement = Measurement::from_run(report, runner.iterations);

        assert_eq!(
            measurement.iterations[0].data.allocated,
            iteration_allocated
        );
        assert_eq!(measurement.total_allocated, report.final_reading);
        assert_eq!(measurement.memory_limit, report.absolute_limit);
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
            predict_next_memory: None,
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
