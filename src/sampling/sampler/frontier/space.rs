//! Frontier-constrained derivation space shared by frontier sampling policies.
//!
//! This module owns the correctness-critical part of frontier sampling: the
//! state threaded through the derivation and the enumeration of feasible
//! productions. Sampling policies only choose between productions returned by
//! [`FrontierSpace`], so they cannot accidentally construct a term that
//! violates the requested frontier state.

use egg::{EGraph, Id, RecExpr};
use hashbrown::HashMap;
use rand_chacha::ChaCha12Rng;

use crate::Counter;
use crate::sampling::count::{NodeMatch, NovelTermCount, convolve, suffix_convolutions};
use crate::{MyAnalysis, MyLanguage, OriginLang, stack_children};

/// Whether a current-graph extraction is outside the previous graph or agrees
/// with one particular previous e-class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FrontierState {
    OutsidePrev,
    InsidePrev(Id),
}

/// One feasible root-production/profile choice at a `(class, size, state)`.
pub(crate) struct FrontierBranch<'a, C> {
    pub node_idx: usize,
    pub child_states: Vec<FrontierState>,
    pub count: C,
    child_hists: Vec<&'a HashMap<usize, C>>,
}

/// One feasible size for the current child, together with the counts needed
/// by count-proportional policies.
pub(crate) struct ChildSizeChoice<C> {
    pub size: usize,
    pub child_count: C,
    pub rest_count: C,
}

#[derive(Clone, Copy)]
pub(crate) struct BranchSite {
    pub curr: Id,
    pub state: FrontierState,
}

pub(crate) struct ChildSizeSite<'a> {
    pub branch: BranchSite,
    pub node_idx: usize,
    pub child_states: &'a [FrontierState],
    pub child_index: usize,
}

/// A policy for choosing among derivations which are already known to satisfy
/// the frontier constraint.
pub(crate) trait FrontierPolicy<C: Counter> {
    fn pick_branch(
        &mut self,
        site: BranchSite,
        choices: &[FrontierBranch<'_, C>],
        rng: &mut ChaCha12Rng,
    ) -> usize;

    fn pick_child_size(
        &mut self,
        site: ChildSizeSite<'_>,
        choices: &[ChildSizeChoice<C>],
        rng: &mut ChaCha12Rng,
    ) -> usize;
}

/// The constrained derivation space common to all frontier samplers.
///
/// `OutsidePrev` corresponds to the old `Novel` recursion mode and
/// `InsidePrev(pc)` to the old `AgreeWith(pc)` mode. Counts and match data come
/// from [`NovelTermCount`]; this type turns them into feasible productions and
/// performs policy-directed recursive construction.
pub(crate) struct FrontierSpace<'a, 'g, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    counts: &'a NovelTermCount<'g, C, L, N>,
}

impl<'a, 'g, C, L, N> FrontierSpace<'a, 'g, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    pub const fn new(counts: &'a NovelTermCount<'g, C, L, N>) -> Self {
        Self { counts }
    }

    pub const fn graph(&self) -> &'g EGraph<L, N> {
        self.counts.curr()
    }

    pub const fn counts(&self) -> &'a NovelTermCount<'g, C, L, N> {
        self.counts
    }

    /// Construct a term in `state` using only feasible frontier productions.
    pub fn construct<P>(
        &self,
        id: Id,
        size: usize,
        state: FrontierState,
        policy: &mut P,
        rng: &mut ChaCha12Rng,
    ) -> RecExpr<OriginLang<L>>
    where
        P: FrontierPolicy<C>,
    {
        let curr = self.graph().find(id);
        let branch_site = BranchSite { curr, state };
        let branches = self.branches(curr, size, state);
        assert!(
            !branches.is_empty(),
            "frontier state has at least one feasible production"
        );

        let branch_idx = policy.pick_branch(branch_site, &branches, rng);
        let branch = &branches[branch_idx];
        let node = &self.graph()[curr].nodes[branch.node_idx];
        let child_budget = size - 1;
        let suffix = suffix_convolutions(&branch.child_hists, child_budget);

        let mut remaining = child_budget;
        let mut children = Vec::with_capacity(node.children().len());
        for (child_index, &child_id) in node.children().iter().enumerate() {
            let choices = branch.child_hists[child_index]
                .iter()
                .filter_map(|(&child_size, child_count)| {
                    let rest_size = remaining.checked_sub(child_size)?;
                    let rest_count = suffix[child_index + 1].get(&rest_size)?;
                    (*rest_count != C::zero()).then(|| ChildSizeChoice {
                        size: child_size,
                        child_count: child_count.clone(),
                        rest_count: rest_count.clone(),
                    })
                })
                .collect::<Vec<_>>();

            assert!(
                !choices.is_empty(),
                "chosen frontier production has a feasible child-size split"
            );
            let size_idx = policy.pick_child_size(
                ChildSizeSite {
                    branch: branch_site,
                    node_idx: branch.node_idx,
                    child_states: &branch.child_states,
                    child_index,
                },
                &choices,
                rng,
            );
            let child_size = choices[size_idx].size;
            remaining -= child_size;
            children.push(self.construct(
                child_id,
                child_size,
                branch.child_states[child_index],
                policy,
                rng,
            ));
        }

        stack_children(&children, OriginLang::new(node.clone(), curr))
    }

    fn branches(&self, curr: Id, size: usize, state: FrontierState) -> Vec<FrontierBranch<'_, C>> {
        match state {
            FrontierState::OutsidePrev => self.outside_branches(curr, size),
            FrontierState::InsidePrev(prev) => self.inside_branches(curr, size, prev),
        }
    }

    fn inside_branches(&self, curr: Id, size: usize, prev: Id) -> Vec<FrontierBranch<'_, C>> {
        let eclass = &self.graph()[curr];
        let child_budget = size - 1;

        eclass
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(node_idx, node)| {
                self.counts
                    .matches_of(curr, node_idx)
                    .iter()
                    .filter(move |m| m.prev_class == prev)
                    .filter_map(move |m| {
                        let child_hists = node
                            .children()
                            .iter()
                            .zip(m.prev_children.iter())
                            .map(|(child, &pc)| self.counts.joint_histogram(*child, pc))
                            .collect::<Option<Vec<_>>>()?;
                        let count = convolve_at::<C>(&child_hists, child_budget)?;
                        Some(FrontierBranch {
                            node_idx,
                            child_states: m
                                .prev_children
                                .iter()
                                .copied()
                                .map(FrontierState::InsidePrev)
                                .collect(),
                            count,
                            child_hists,
                        })
                    })
            })
            .collect()
    }

    fn outside_branches(&self, curr: Id, size: usize) -> Vec<FrontierBranch<'_, C>> {
        let eclass = &self.graph()[curr];
        let child_budget = size - 1;

        eclass
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(node_idx, node)| {
                let matches = self.counts.matches_of(curr, node_idx);
                let children = node.children();
                let slot_options = children
                    .iter()
                    .map(|child| {
                        let mut options = vec![FrontierState::OutsidePrev];
                        options.extend(
                            self.counts
                                .cover_of(*child)
                                .iter()
                                .copied()
                                .map(FrontierState::InsidePrev),
                        );
                        options
                    })
                    .collect::<Vec<_>>();

                enumerate_profiles(&slot_options)
                    .into_iter()
                    .filter(|profile| !completes_some_match(profile, matches))
                    .filter_map(move |child_states| {
                        let child_hists = children
                            .iter()
                            .zip(child_states.iter())
                            .map(|(child, state)| match state {
                                FrontierState::OutsidePrev => self.counts.novel_histogram(*child),
                                FrontierState::InsidePrev(pc) => {
                                    self.counts.joint_histogram(*child, *pc)
                                }
                            })
                            .collect::<Option<Vec<_>>>()?;
                        let count = convolve_at::<C>(&child_hists, child_budget)?;
                        Some(FrontierBranch {
                            node_idx,
                            child_states,
                            count,
                            child_hists,
                        })
                    })
            })
            .collect()
    }
}

fn enumerate_profiles<T: Clone>(slot_options: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut profiles = vec![Vec::new()];
    for slot in slot_options {
        let mut next = Vec::with_capacity(profiles.len() * slot.len());
        for prefix in &profiles {
            for option in slot {
                let mut profile = prefix.clone();
                profile.push(option.clone());
                next.push(profile);
            }
        }
        profiles = next;
    }
    profiles
}

fn completes_some_match(profile: &[FrontierState], matches: &[NodeMatch]) -> bool {
    matches.iter().any(|m| {
        profile.len() == m.prev_children.len()
            && profile
                .iter()
                .zip(m.prev_children.iter())
                .all(|(state, &pc)| *state == FrontierState::InsidePrev(pc))
    })
}

fn convolve_at<C: Counter>(histograms: &[&HashMap<usize, C>], budget: usize) -> Option<C> {
    if histograms.iter().any(|h| h.is_empty()) {
        return None;
    }
    convolve(histograms, budget).get(&budget).cloned()
}
