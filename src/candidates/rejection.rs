//! Count-free, exact-novelty candidate construction.
//!
//! Proposal engines extract exact-size terms from the current root. A shared
//! rejection layer removes expressions present at the previous e-graph
//! boundary and batch-local duplicates. This path constructs no term counts,
//! rooted matches, or suffix-convolution tables.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use egg::{EGraph, Id, RecExpr};
use hashbrown::{HashMap, HashSet};
use rand::Rng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;

use crate::candidates::SizeAllocation;
use crate::candidates::exact::count::{RootBudgets, root_budgets};
use crate::eqsat::EqsatResult;
use crate::previous::PrevIndex;
use crate::utils::{combined_rng, live_heap_bytes};
use crate::{MyAnalysis, MyLanguage, OriginLang, stack_children};

/// One low-memory proposal source. `None` is a bounded construction failure,
/// not evidence that the requested size is unreachable.
pub trait ProposalEngine<L: MyLanguage> {
    fn propose(&self, size: usize, rng: &mut ChaCha12Rng) -> Option<RecExpr<OriginLang<L>>>;

    /// Sorted sizes worth scheduling. A walk returns an attempt domain;
    /// feasibility returns exact plain-reachable root sizes.
    fn candidate_sizes(&self) -> Vec<usize>;
}

/// Operational bounds shared by both rejection-backed pools.
#[derive(Clone, Copy, Debug)]
pub struct RejectionLimits {
    /// Recursive state/partition visits allowed to one random-walk proposal.
    pub walk_backtrack: usize,
    /// Proposal attempts allowed for one target size.
    pub attempts_per_size: usize,
    /// Proposal attempts allowed for the whole candidate pool.
    pub global_attempts: usize,
    /// Wall-clock proposal limit.
    pub max_time: Duration,
    /// Absolute live-heap ceiling. `None` is unbounded.
    pub max_memory: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SizeRejectionStats {
    pub proposal_attempts: usize,
    pub construction_failures: usize,
    pub non_novel_rejections: usize,
    pub duplicate_rejections: usize,
    pub accepted_unique_terms: usize,
}

/// Rejection telemetry. Budget exhaustion never means an exact empty frontier.
#[derive(Clone, Debug)]
pub struct RejectionStats {
    pub per_size: BTreeMap<usize, SizeRejectionStats>,
    pub elapsed: Duration,
    pub peak_live_heap: u64,
    pub stop_reason: &'static str,
}

impl RejectionStats {
    #[must_use]
    pub fn proposal_attempts(&self) -> usize {
        self.per_size.values().map(|s| s.proposal_attempts).sum()
    }

    #[expect(clippy::cast_precision_loss)]
    pub fn log(&self, pool: &str) {
        eprintln!(
            "rejection_stats pool={pool} attempts={} construction_failures={} \
             non_novel={} duplicates={} accepted={} elapsed_seconds={:.6} \
             peak_live_heap={} stop_reason={}",
            self.proposal_attempts(),
            self.per_size
                .values()
                .map(|s| s.construction_failures)
                .sum::<usize>(),
            self.per_size
                .values()
                .map(|s| s.non_novel_rejections)
                .sum::<usize>(),
            self.per_size
                .values()
                .map(|s| s.duplicate_rejections)
                .sum::<usize>(),
            self.per_size
                .values()
                .map(|s| s.accepted_unique_terms)
                .sum::<usize>(),
            self.elapsed.as_secs_f64(),
            self.peak_live_heap,
            self.stop_reason,
        );
        for (size, stats) in &self.per_size {
            let acceptance = if stats.proposal_attempts == 0 {
                0.0
            } else {
                stats.accepted_unique_terms as f64 / stats.proposal_attempts as f64
            };
            eprintln!(
                "rejection_size pool={pool} size={size} attempts={} \
                 construction_failures={} non_novel={} duplicates={} accepted={} \
                 acceptance_rate={acceptance:.6}",
                stats.proposal_attempts,
                stats.construction_failures,
                stats.non_novel_rejections,
                stats.duplicate_rejections,
                stats.accepted_unique_terms,
            );
        }
    }
}

pub struct RejectionBatch<L: MyLanguage> {
    pub candidates: Vec<RecExpr<OriginLang<L>>>,
    pub stats: RejectionStats,
}

/// Previous-boundary index and count-independent extraction data borrowing the
/// final replay graph. Dropping this package releases the previous index while
/// leaving the replay result available to optional exact candidate counting.
pub struct RejectionCandidatePackage<'a, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    egraph: &'a EGraph<L, N>,
    previous: PrevIndex<L>,
    root: Id,
    domain: RootBudgets,
}

impl<'a, L, N> RejectionCandidatePackage<'a, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    /// Build the previous lookup and extraction domain with memory checks at
    /// each material allocation boundary.
    pub fn new(result: &'a EqsatResult<L, N>, max_size: usize) -> Self {
        // check_memory(max_memory, "before previous-index construction")?;
        let previous = result.prev_index();
        // check_memory(max_memory, "after previous-index construction")?;
        let egraph = result.curr();
        let root = egraph.find(result.root());
        let domain = root_budgets(egraph, root, max_size);
        // check_memory(max_memory, "after extraction-domain construction")?;
        Self {
            egraph,
            previous,
            root,
            domain,
        }
    }

    #[must_use]
    pub fn random_walk(
        &self,
        start_size: usize,
        limits: RejectionLimits,
    ) -> RandomWalkEngine<'_, L, N> {
        RandomWalkEngine {
            egraph: self.egraph,
            domain: &self.domain,
            root: self.root,
            start_size,
            max_size: self.domain.limit(),
            backtrack: limits.walk_backtrack,
        }
    }

    #[must_use]
    pub fn feasibility(
        &self,
        // max_memory: Option<u64>,
    ) -> FeasibilityEngine<'_, L, N> {
        FeasibilityEngine::build(self.egraph, &self.domain, self.root)
        // check_memory(max_memory, "after feasibility-bitset construction")?;
    }

    /// # Panics
    ///
    /// Panics if `candidate_count` or `novel_size_goal` is zero, or for
    /// count-proportional selection, whose semantics are unavailable without
    /// exact counts.
    #[expect(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn collect<E: ProposalEngine<L>>(
        &self,
        engine: &E,
        candidate_count: usize,
        novel_size_goal: usize,
        allocation: SizeAllocation,
        candidate_seed: u64,
        pool_salt: u64,
        limits: RejectionLimits,
    ) -> RejectionBatch<L> {
        assert!(candidate_count > 0);
        assert!(novel_size_goal > 0);
        let mut sizes = engine.candidate_sizes();
        sizes.sort_unstable();
        sizes.dedup();
        let mut state = RejectionState::new(&sizes, candidate_seed, pool_salt);

        // Probe in rounds so one impossible or old-only size cannot monopolize the
        // discovery budget. Observed means at least one exact-novel proposal, not
        // proof that an earlier unobserved size has an empty frontier.
        let mut active = sizes.clone();
        while state.accepted.len() < novel_size_goal && !active.is_empty() {
            active.retain(|&size| {
                if state.accepted.len() >= novel_size_goal {
                    return false;
                }
                state.attempt(size, engine, &self.previous, limits)
                    && state.accepted.get(&size).map_or(0, Vec::len) < candidate_count
                    && state.stats[&size].proposal_attempts < limits.attempts_per_size
            });
            if matches!(
                state.stop_reason,
                "memory_limit" | "time_limit" | "global_attempt_limit"
            ) {
                break;
            }
        }

        let selected = state
            .accepted
            .keys()
            .copied()
            .take(novel_size_goal)
            .collect::<Vec<_>>();
        let mut candidates = Vec::with_capacity(candidate_count);
        match allocation {
            SizeAllocation::Greedy => {
                for &size in &selected {
                    while state.unique.len() < candidate_count {
                        if !state.attempt(size, engine, &self.previous, limits) {
                            break;
                        }
                    }
                }
                candidates.extend(
                    selected
                        .into_iter()
                        .flat_map(|size| state.accepted.remove(&size).unwrap_or_default())
                        .take(candidate_count),
                );
            }
            SizeAllocation::Uniform => {
                let base = if selected.is_empty() {
                    0
                } else {
                    candidate_count / selected.len()
                };
                let remainder = if selected.is_empty() {
                    0
                } else {
                    candidate_count % selected.len()
                };
                for (index, &size) in selected.iter().enumerate() {
                    let quota = base + usize::from(index < remainder);
                    while state.accepted.get(&size).map_or(0, Vec::len) < quota {
                        if !state.attempt(size, engine, &self.previous, limits) {
                            break;
                        }
                    }
                }
                let mut bins = selected
                    .into_iter()
                    .map(|size| state.accepted.remove(&size).unwrap_or_default().into_iter())
                    .collect::<Vec<_>>();
                while candidates.len() < candidate_count {
                    let before = candidates.len();
                    for bin in &mut bins {
                        if let Some(term) = bin.next() {
                            candidates.push(term);
                            if candidates.len() == candidate_count {
                                break;
                            }
                        }
                    }
                    if candidates.len() == before {
                        break;
                    }
                }
            }
            SizeAllocation::Proportional(_) => unreachable!(
                "proportional size allocation requires exact term counts and is unsupported for \
                 rejection-backed candidate pools"
            ),
        }
        if candidates.len() == candidate_count {
            state.stop_reason = "quota_filled";
        } else if candidates.is_empty() && state.stop_reason == "quota_or_size_budget" {
            state.stop_reason = "no_novel_candidate_observed";
        }
        let live = live_heap_bytes();
        state.peak_live_heap = state.peak_live_heap.max(live);
        RejectionBatch {
            candidates,
            stats: RejectionStats {
                per_size: state.stats,
                elapsed: state.start.elapsed(),
                peak_live_heap: state.peak_live_heap,
                stop_reason: state.stop_reason,
            },
        }
    }
}

/// Exact-size random walk with bounded recursive backtracking.
pub struct RandomWalkEngine<'a, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    egraph: &'a EGraph<L, N>,
    domain: &'a RootBudgets,
    root: Id,
    start_size: usize,
    max_size: usize,
    backtrack: usize,
}

impl<L, N> RandomWalkEngine<'_, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn construct(
        &self,
        id: Id,
        size: usize,
        rng: &mut ChaCha12Rng,
        fuel: &mut usize,
    ) -> Option<RecExpr<OriginLang<L>>> {
        if size == 0 || *fuel == 0 {
            return None;
        }
        *fuel -= 1;
        let id = self.egraph.find(id);
        if self.domain.budget(id).is_none_or(|budget| size > budget) {
            return None;
        }

        let mut eligible = self.egraph[id]
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                let minimum = node.children().iter().try_fold(1usize, |total, &child| {
                    total.checked_add(self.domain.min_size(self.egraph.find(child)))
                })?;
                (minimum <= size).then_some(index)
            })
            .collect::<Vec<_>>();
        eligible.shuffle(rng);

        for node_index in eligible {
            let node = &self.egraph[id].nodes[node_index];
            if node.children().is_empty() {
                if size == 1 {
                    return Some(stack_children(&[], OriginLang::new(node.clone(), id)));
                }
                continue;
            }
            while *fuel > 0 {
                *fuel -= 1;
                let Some(sizes) = self.random_partition(node.children(), size - 1, rng) else {
                    break;
                };
                let mut children = Vec::with_capacity(sizes.len());
                let mut failed = false;
                for (&child, child_size) in node.children().iter().zip(sizes) {
                    let Some(expr) = self.construct(child, child_size, rng, fuel) else {
                        failed = true;
                        break;
                    };
                    children.push(expr);
                }
                if !failed {
                    return Some(stack_children(&children, OriginLang::new(node.clone(), id)));
                }
            }
        }
        None
    }

    fn random_partition(
        &self,
        children: &[Id],
        total: usize,
        rng: &mut ChaCha12Rng,
    ) -> Option<Vec<usize>> {
        let mut sizes = children
            .iter()
            .map(|&child| self.domain.min_size(self.egraph.find(child)))
            .collect::<Vec<_>>();
        let minimum: usize = sizes.iter().sum();
        let mut extra = total.checked_sub(minimum)?;
        match sizes.len() {
            0 => return (extra == 0).then_some(sizes),
            1 => sizes[0] += extra,
            2 => {
                let left = rng.gen_range(0..=extra);
                sizes[0] += left;
                sizes[1] += extra - left;
            }
            _ => {
                // A bounded randomized composition without materializing all
                // stars-and-bars partitions. Every composition has positive
                // probability; common unary/binary arities are specialized.
                for size in sizes.iter_mut().take(children.len() - 1) {
                    let take = rng.gen_range(0..=extra);
                    *size += take;
                    extra -= take;
                }
                *sizes.last_mut().unwrap() += extra;
            }
        }
        Some(sizes)
    }
}

impl<L, N> ProposalEngine<L> for RandomWalkEngine<'_, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn propose(&self, size: usize, rng: &mut ChaCha12Rng) -> Option<RecExpr<OriginLang<L>>> {
        let mut fuel = self.backtrack;
        self.construct(self.root, size, rng, &mut fuel)
    }

    fn candidate_sizes(&self) -> Vec<usize> {
        (self.start_size..=self.max_size).collect()
    }
}

/// Flat exact reachability bits followed by local-uniform construction.
pub struct FeasibilityEngine<'a, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    egraph: &'a EGraph<L, N>,
    domain: &'a RootBudgets,
    root: Id,
    classes: Vec<Id>,
    dense: HashMap<Id, usize>,
    words_per_class: usize,
    bits: Vec<u64>,
}

impl<'a, L, N> FeasibilityEngine<'a, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn build(egraph: &'a EGraph<L, N>, domain: &'a RootBudgets, root: Id) -> Self {
        let mut classes = domain.budgets().keys().copied().collect::<Vec<_>>();
        classes.sort_unstable_by_key(|id| usize::from(*id));
        let dense = classes
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index))
            .collect::<HashMap<_, _>>();
        let words_per_class = (domain.limit() + 64) / 64;
        let mut engine = Self {
            egraph,
            domain,
            root: egraph.find(root),
            bits: vec![0; classes.len() * words_per_class],
            classes,
            dense,
            words_per_class,
        };
        for size in 1..=domain.limit() {
            for dense_index in 0..engine.classes.len() {
                let id = engine.classes[dense_index];
                if size > domain.budget(id).unwrap() {
                    continue;
                }
                if engine.egraph[id]
                    .nodes
                    .iter()
                    .any(|node| engine.node_feasible(node, size))
                {
                    engine.set(dense_index, size);
                }
            }
        }
        engine
    }

    fn set(&mut self, dense_index: usize, size: usize) {
        let index = dense_index * self.words_per_class + size / 64;
        self.bits[index] |= 1u64 << (size % 64);
    }

    fn reachable(&self, id: Id, size: usize) -> bool {
        if size == 0 || size > self.domain.limit() {
            return false;
        }
        let id = self.egraph.find(id);
        let Some(&dense_index) = self.dense.get(&id) else {
            return false;
        };
        let index = dense_index * self.words_per_class + size / 64;
        self.bits[index] & (1u64 << (size % 64)) != 0
    }

    fn node_feasible(&self, node: &L, size: usize) -> bool {
        let Some(total) = size.checked_sub(1) else {
            return false;
        };
        match node.children() {
            [] => total == 0,
            [child] => self.reachable(*child, total),
            [left, right] => (1..total).any(|left_size| {
                self.reachable(*left, left_size) && self.reachable(*right, total - left_size)
            }),
            children => self.partition_exists(children, 0, total),
        }
    }

    fn partition_exists(&self, children: &[Id], index: usize, remaining: usize) -> bool {
        if index == children.len() {
            return remaining == 0;
        }
        let rest_minimum = children[index + 1..]
            .iter()
            .map(|&child| self.domain.min_size(self.egraph.find(child)))
            .sum::<usize>();
        let minimum = self.domain.min_size(self.egraph.find(children[index]));
        let Some(maximum) = remaining.checked_sub(rest_minimum) else {
            return false;
        };
        (minimum..=maximum).any(|size| {
            self.reachable(children[index], size)
                && self.partition_exists(children, index + 1, remaining - size)
        })
    }

    fn choose_partition(
        &self,
        children: &[Id],
        total: usize,
        rng: &mut ChaCha12Rng,
    ) -> Option<Vec<usize>> {
        let mut current = Vec::with_capacity(children.len());
        let mut chosen = None;
        let mut seen = 0usize;
        self.visit_partitions(
            children,
            0,
            total,
            &mut current,
            rng,
            &mut seen,
            &mut chosen,
        );
        chosen
    }

    #[expect(clippy::too_many_arguments)]
    fn visit_partitions(
        &self,
        children: &[Id],
        index: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        rng: &mut ChaCha12Rng,
        seen: &mut usize,
        chosen: &mut Option<Vec<usize>>,
    ) {
        if index == children.len() {
            if remaining == 0 {
                *seen += 1;
                if rng.gen_range(0..*seen) == 0 {
                    *chosen = Some(current.clone());
                }
            }
            return;
        }
        let rest_minimum = children[index + 1..]
            .iter()
            .map(|&child| self.domain.min_size(self.egraph.find(child)))
            .sum::<usize>();
        let minimum = self.domain.min_size(self.egraph.find(children[index]));
        let Some(maximum) = remaining.checked_sub(rest_minimum) else {
            return;
        };
        for size in minimum..=maximum {
            if self.reachable(children[index], size) {
                current.push(size);
                self.visit_partitions(
                    children,
                    index + 1,
                    remaining - size,
                    current,
                    rng,
                    seen,
                    chosen,
                );
                current.pop();
            }
        }
    }

    fn construct(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        let id = self.egraph.find(id);
        assert!(self.reachable(id, size));
        let mut seen = 0usize;
        let mut selected = None;
        for (index, node) in self.egraph[id].nodes.iter().enumerate() {
            if self.node_feasible(node, size) {
                seen += 1;
                if rng.gen_range(0..seen) == 0 {
                    selected = Some(index);
                }
            }
        }
        let node = &self.egraph[id].nodes[selected.expect("reachable state has a feasible node")];
        let partition = self
            .choose_partition(node.children(), size - 1, rng)
            .expect("feasible node has a feasible child-size partition");
        let children = node
            .children()
            .iter()
            .zip(partition)
            .map(|(&child, child_size)| self.construct(child, child_size, rng))
            .collect::<Vec<_>>();
        stack_children(&children, OriginLang::new(node.clone(), id))
    }
}

impl<L, N> ProposalEngine<L> for FeasibilityEngine<'_, L, N>
where
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn propose(&self, size: usize, rng: &mut ChaCha12Rng) -> Option<RecExpr<OriginLang<L>>> {
        self.reachable(self.root, size)
            .then(|| self.construct(self.root, size, rng))
    }

    fn candidate_sizes(&self) -> Vec<usize> {
        (1..=self.domain.limit())
            .filter(|&size| self.reachable(self.root, size))
            .collect()
    }
}

struct RejectionState<L: MyLanguage> {
    accepted: BTreeMap<usize, Vec<RecExpr<OriginLang<L>>>>,
    unique: HashSet<RecExpr<OriginLang<L>>>,
    stats: BTreeMap<usize, SizeRejectionStats>,
    rngs: HashMap<usize, ChaCha12Rng>,
    total_attempts: usize,
    start: Instant,
    peak_live_heap: u64,
    stop_reason: &'static str,
}

impl<L: MyLanguage> RejectionState<L> {
    fn new(sizes: &[usize], candidate_seed: u64, pool_salt: u64) -> Self {
        Self {
            accepted: BTreeMap::new(),
            unique: HashSet::new(),
            stats: sizes
                .iter()
                .map(|&size| (size, SizeRejectionStats::default()))
                .collect(),
            rngs: sizes
                .iter()
                .map(|&size| (size, combined_rng([candidate_seed, pool_salt, size as u64])))
                .collect(),
            total_attempts: 0,
            start: Instant::now(),
            peak_live_heap: live_heap_bytes(),
            stop_reason: "quota_or_size_budget",
        }
    }

    fn can_attempt(&mut self, size: usize, limits: RejectionLimits) -> bool {
        // Advancing jemalloc's stats epoch is materially more expensive than
        // a proposal on small graphs, so measure the heap periodically.
        if self.total_attempts.is_multiple_of(64) {
            let live = live_heap_bytes();
            self.peak_live_heap = self.peak_live_heap.max(live);
            if limits.max_memory.is_some_and(|limit| live > limit) {
                self.stop_reason = "memory_limit";
                return false;
            }
        }
        if self.start.elapsed() >= limits.max_time {
            self.stop_reason = "time_limit";
            return false;
        }
        if self.total_attempts >= limits.global_attempts {
            self.stop_reason = "global_attempt_limit";
            return false;
        }
        self.stats[&size].proposal_attempts < limits.attempts_per_size
    }

    fn attempt<E: ProposalEngine<L>>(
        &mut self,
        size: usize,
        engine: &E,
        previous: &PrevIndex<L>,
        limits: RejectionLimits,
    ) -> bool {
        if !self.can_attempt(size, limits) {
            return false;
        }
        self.total_attempts += 1;
        self.stats.get_mut(&size).unwrap().proposal_attempts += 1;
        let rng = self.rngs.get_mut(&size).unwrap();
        let Some(term) = engine.propose(size, rng) else {
            self.stats.get_mut(&size).unwrap().construction_failures += 1;
            return true;
        };
        debug_assert_eq!(term.as_ref().len(), size);
        if previous.contains_origin_expr(&term) {
            self.stats.get_mut(&size).unwrap().non_novel_rejections += 1;
        } else if !self.unique.insert(term.clone()) {
            self.stats.get_mut(&size).unwrap().duplicate_rejections += 1;
        } else {
            self.stats.get_mut(&size).unwrap().accepted_unique_terms += 1;
            self.accepted.entry(size).or_default().push(term);
        }
        true
    }
}

// fn check_memory(max_memory: Option<u64>, boundary: &str) -> Result<(), String> {
//     let live = live_heap_bytes();
//     if max_memory.is_some_and(|limit| live > limit) {
//         Err(format!(
//             "candidate-construction memory limit exceeded {boundary}: live_heap={live} bytes, limit={} bytes",
//             max_memory.unwrap()
//         ))
//     } else {
//         eprintln!("candidate_memory boundary={boundary:?} live_heap={live}");
//         Ok(())
//     }
// }

#[cfg(test)]
mod tests {
    use num::BigUint;

    use super::*;
    use crate::candidates::exact::count::count_terms_rooted;
    use crate::eqsat::EqsatResult;
    use crate::langs::math::Math;
    use crate::utils::sym;

    fn cyclic_result() -> EqsatResult<Math, ()> {
        let mut graph = EGraph::<Math, ()>::new(());
        graph.enable_union_event_recording();
        let a = graph.add(sym("a"));
        let b = graph.add(sym("b"));
        let add = graph.add(Math::Add([a, b]));
        graph.rebuild();
        let raw = graph.nodes().len();
        let unions = graph.union_event_count();
        graph.union(a, add);
        graph.rebuild();
        EqsatResult::new_for_tests(graph, add, raw, unions)
    }

    #[test]
    fn walk_is_exact_size_deterministic_and_terminates_on_cycle() {
        let result = cyclic_result();
        let package = RejectionCandidatePackage::new(&result, 9);
        let limits = RejectionLimits {
            walk_backtrack: 512,
            attempts_per_size: 10,
            global_attempts: 100,
            max_time: Duration::from_secs(1),
            max_memory: None,
        };
        let walk = package.random_walk(1, limits);
        for size in [1, 3, 5, 7, 9] {
            let mut left = combined_rng([42, size as u64]);
            let mut right = combined_rng([42, size as u64]);
            let first = walk.propose(size, &mut left).unwrap();
            let second = walk.propose(size, &mut right).unwrap();
            assert_eq!(first, second);
            assert_eq!(first.as_ref().len(), size);
        }
        let mut rng = combined_rng([9]);
        assert!(walk.propose(2, &mut rng).is_none());
    }

    #[test]
    fn feasibility_matches_nonzero_plain_counts_and_constructs_every_size() {
        let result = cyclic_result();
        let package = RejectionCandidatePackage::new(&result, 9);
        let feasible = package.feasibility();
        let counts = count_terms_rooted::<BigUint, _, _>(result.curr(), &package.domain);
        let mut exact_sizes = counts.data[&package.root]
            .keys()
            .copied()
            .collect::<Vec<_>>();
        exact_sizes.sort_unstable();
        assert_eq!(feasible.candidate_sizes(), exact_sizes);
        for size in exact_sizes {
            let mut rng = combined_rng([77, size as u64]);
            let term = feasible.propose(size, &mut rng).unwrap();
            assert_eq!(term.as_ref().len(), size);
        }
    }

    #[test]
    fn shared_rejection_accepts_only_exact_novel_unique_terms() {
        let result = cyclic_result();
        let package = RejectionCandidatePackage::new(&result, 9);
        let limits = RejectionLimits {
            walk_backtrack: 512,
            attempts_per_size: 64,
            global_attempts: 256,
            max_time: Duration::from_secs(1),
            max_memory: None,
        };
        let feasible = package.feasibility();
        let batch = package.collect(&feasible, 3, 3, SizeAllocation::Greedy, 5, 11, limits);
        assert!(!batch.candidates.is_empty());
        let mut unique = HashSet::new();
        for term in &batch.candidates {
            assert!(!package.previous.contains_origin_expr(term));
            assert!(unique.insert(term));
        }
        assert!(
            batch
                .stats
                .per_size
                .values()
                .any(|s| s.non_novel_rejections > 0)
        );
    }
}
