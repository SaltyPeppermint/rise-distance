//! Coverage-balanced frontier sampling.
//!
//! Unlike [`super::IndependentFrontierSampler`], this sampler coordinates the
//! derivation choices made for all terms in one size bucket. It preferentially
//! selects under-used e-nodes, frontier-state profiles, and child sizes. Every
//! choice still comes from [`FrontierSpace`], so balancing cannot weaken the
//! frontier constraint.

use egg::{Id, RecExpr};
use hashbrown::{HashMap, HashSet};
use rand::Rng;
use rand_chacha::ChaCha12Rng;

use super::space::{
    BranchSite, ChildSizeChoice, ChildSizeSite, FrontierBranch, FrontierPolicy, FrontierSpace,
    FrontierState,
};
use crate::Counter;
use crate::sampling::count::NovelTermCount;
use crate::sampling::sampler::{MAX_OVERSAMPLE, Sampler};
use crate::{MyAnalysis, MyLanguage, OriginLang, lower, utils};

/// Relative penalties used when balancing local derivation choices.
///
/// Larger values make the sampler more reluctant to repeat that feature. A
/// zero value disables the corresponding coverage signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BalanceConfig {
    pub node_penalty: u64,
    pub profile_penalty: u64,
    pub child_size_penalty: u64,
}

impl Default for BalanceConfig {
    fn default() -> Self {
        Self {
            node_penalty: 2,
            profile_penalty: 1,
            child_size_penalty: 1,
        }
    }
}

/// Constructs frontier terms while balancing the derivation decisions made
/// across each requested size bucket.
pub struct BalancedFrontierSampler<'a, 'g, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    space: FrontierSpace<'a, 'g, C, L, N>,
    root: Id,
    config: BalanceConfig,
}

impl<'a, 'g, C, L, N> BalancedFrontierSampler<'a, 'g, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    #[must_use]
    pub const fn new(counts: &'a NovelTermCount<'g, C, L, N>, root: Id) -> Self {
        Self::with_config(
            counts,
            root,
            BalanceConfig {
                node_penalty: 2,
                profile_penalty: 1,
                child_size_penalty: 1,
            },
        )
    }

    #[must_use]
    pub const fn with_config(
        counts: &'a NovelTermCount<'g, C, L, N>,
        root: Id,
        config: BalanceConfig,
    ) -> Self {
        Self {
            space: FrontierSpace::new(counts),
            root,
            config,
        }
    }

    fn construct(
        &self,
        id: Id,
        size: usize,
        coverage: &mut CoveragePolicy,
        rng: &mut ChaCha12Rng,
    ) -> RecExpr<OriginLang<L>> {
        let sample = self
            .space
            .construct(id, size, FrontierState::OutsidePrev, coverage, rng);
        debug_assert!(
            self.space
                .counts()
                .prev()
                .lookup_expr(&lower(sample.clone()))
                .is_none(),
            "balanced frontier sampler produced a term extractable from the previous graph"
        );
        sample
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NodeKey {
    curr: Id,
    state: FrontierState,
    node_idx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProfileKey {
    node: NodeKey,
    child_states: Vec<FrontierState>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ChildSizeKey {
    profile: ProfileKey,
    child_index: usize,
    child_size: usize,
}

struct CoveragePolicy {
    config: BalanceConfig,
    node_usage: HashMap<NodeKey, u64>,
    profile_usage: HashMap<ProfileKey, u64>,
    child_size_usage: HashMap<ChildSizeKey, u64>,
}

impl CoveragePolicy {
    fn new(config: BalanceConfig) -> Self {
        Self {
            config,
            node_usage: HashMap::new(),
            profile_usage: HashMap::new(),
            child_size_usage: HashMap::new(),
        }
    }

    fn node_key(site: BranchSite, node_idx: usize) -> NodeKey {
        NodeKey {
            curr: site.curr,
            state: site.state,
            node_idx,
        }
    }

    fn profile_key(
        site: BranchSite,
        node_idx: usize,
        child_states: &[FrontierState],
    ) -> ProfileKey {
        ProfileKey {
            node: Self::node_key(site, node_idx),
            child_states: child_states.to_vec(),
        }
    }

    fn branch_score(&self, site: BranchSite, branch: &FrontierBranch<'_, impl Counter>) -> u64 {
        let node = Self::node_key(site, branch.node_idx);
        let profile = Self::profile_key(site, branch.node_idx, &branch.child_states);
        self.config
            .node_penalty
            .saturating_mul(*self.node_usage.get(&node).unwrap_or(&0))
            .saturating_add(
                self.config
                    .profile_penalty
                    .saturating_mul(*self.profile_usage.get(&profile).unwrap_or(&0)),
            )
    }

    fn choose_lowest_score(scores: impl Iterator<Item = u64>, rng: &mut ChaCha12Rng) -> usize {
        let scores = scores.collect::<Vec<_>>();
        let best = *scores.iter().min().expect("at least one coverage choice");
        let tied = scores
            .iter()
            .enumerate()
            .filter_map(|(idx, &score)| (score == best).then_some(idx))
            .collect::<Vec<_>>();
        tied[rng.gen_range(0..tied.len())]
    }
}

impl<C: Counter> FrontierPolicy<C> for CoveragePolicy {
    fn pick_branch(
        &mut self,
        site: BranchSite,
        choices: &[FrontierBranch<'_, C>],
        rng: &mut ChaCha12Rng,
    ) -> usize {
        let selected =
            Self::choose_lowest_score(choices.iter().map(|b| self.branch_score(site, b)), rng);
        let branch = &choices[selected];
        let node = Self::node_key(site, branch.node_idx);
        let profile = Self::profile_key(site, branch.node_idx, &branch.child_states);
        let node_usage = self.node_usage.entry(node).or_default();
        *node_usage = node_usage.saturating_add(1);
        let profile_usage = self.profile_usage.entry(profile).or_default();
        *profile_usage = profile_usage.saturating_add(1);
        selected
    }

    fn pick_child_size(
        &mut self,
        site: ChildSizeSite<'_>,
        choices: &[ChildSizeChoice<C>],
        rng: &mut ChaCha12Rng,
    ) -> usize {
        let profile = Self::profile_key(site.branch, site.node_idx, site.child_states);
        let selected = Self::choose_lowest_score(
            choices.iter().map(|choice| {
                let key = ChildSizeKey {
                    profile: profile.clone(),
                    child_index: site.child_index,
                    child_size: choice.size,
                };
                self.config
                    .child_size_penalty
                    .saturating_mul(*self.child_size_usage.get(&key).unwrap_or(&0))
            }),
            rng,
        );
        let key = ChildSizeKey {
            profile,
            child_index: site.child_index,
            child_size: choices[selected].size,
        };
        let usage = self.child_size_usage.entry(key).or_default();
        *usage = usage.saturating_add(1);
        selected
    }
}

impl<C, L, N> Sampler<C, L, N> for BalancedFrontierSampler<'_, '_, C, L, N>
where
    C: Counter,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    fn root(&self) -> Id {
        self.root
    }

    fn find(&self, id: Id) -> Id {
        self.space.graph().find(id)
    }

    fn size_histogram(&self, id: Id) -> Option<&HashMap<usize, C>> {
        self.space.counts().data().get(&self.find(id))
    }

    fn sample(&self, id: Id, size: usize, rng: &mut ChaCha12Rng) -> RecExpr<OriginLang<L>> {
        self.construct(id, size, &mut CoveragePolicy::new(self.config), rng)
    }

    fn sample_size(
        &self,
        id: Id,
        size: usize,
        samples: u64,
        seed: [u64; 2],
    ) -> Option<Vec<RecExpr<OriginLang<L>>>> {
        let requested = usize::try_from(samples).unwrap();
        let count = self.size_histogram(self.find(id))?.get(&size)?;
        let target = requested.min(count.to_usize().unwrap_or(requested));
        if target == 0 {
            return None;
        }

        let mut rng = utils::combined_rng([size as u64, seed[0], seed[1]]);
        let mut coverage = CoveragePolicy::new(self.config);
        let mut terms = HashSet::with_capacity(target);

        // Keep the coverage state across refill draws so duplicates steer
        // subsequent constructions toward under-used choices. As in the
        // default sampler, cap the work to keep duplicate-heavy frontiers from
        // turning this into an unbounded rejection loop.
        let mut budget = samples * MAX_OVERSAMPLE;
        while terms.len() < target && budget > 0 {
            terms.insert(self.construct(id, size, &mut coverage, &mut rng));
            budget -= 1;
        }

        let mut terms = terms.into_iter().collect::<Vec<_>>();
        terms.sort_unstable();
        (!terms.is_empty()).then_some(terms)
    }
}

#[cfg(test)]
mod tests {
    use egg::EGraph;
    use num::BigUint;

    use super::*;
    use crate::langs::math::Math;
    use crate::sampling::count::PlainTermCount;
    use crate::test_utils::sym;

    #[test]
    fn balanced_sampler_covers_frontier_profiles_without_rejection() {
        // prev contains (+ a b). After merging a and b, the other three
        // combinations are frontier terms represented by three distinct
        // child-state profiles under the same root e-node.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let root = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.rebuild();

        let plain = PlainTermCount::<BigUint>::new(5, &curr);
        let counts = NovelTermCount::new(5, &curr, &prev, plain);
        let sampler = BalancedFrontierSampler::new(&counts, root);
        let terms = sampler
            .sample_size(root, 3, 3, [17, 23])
            .expect("three frontier terms");
        let lowered = terms
            .into_iter()
            .map(|term| lower(term).to_string())
            .collect::<HashSet<_>>();

        assert_eq!(
            lowered,
            HashSet::from([
                "(+ a a)".to_owned(),
                "(+ b a)".to_owned(),
                "(+ b b)".to_owned(),
            ])
        );
    }

    #[test]
    fn balanced_sampler_never_returns_previous_term() {
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let root = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();
        curr.union(a, b);
        curr.rebuild();

        let plain = PlainTermCount::<BigUint>::new(5, &curr);
        let counts = NovelTermCount::new(5, &curr, &prev, plain);
        let sampler = BalancedFrontierSampler::new(&counts, root);
        let terms = sampler
            .sample_size(root, 3, 3, [0, 0])
            .expect("non-empty frontier");

        for term in terms {
            assert!(prev.lookup_expr(&lower(term)).is_none());
        }
    }

    #[test]
    fn balanced_sampler_refills_duplicates_up_to_target() {
        // The previous graph contains only (+ a b). After merging a, b, and c,
        // the root has eight novel size-3 combinations. With coverage penalties
        // disabled, this seed repeats at least one combination in the first
        // eight draws, so reaching all eight requires a refill draw.
        let mut curr = EGraph::<Math, ()>::new(());
        let a = curr.add(sym("a"));
        let b = curr.add(sym("b"));
        let c = curr.add(sym("c"));
        let root = curr.add(Math::Add([a, b]));
        curr.rebuild();
        let prev = curr.clone();

        curr.union(a, b);
        curr.union(a, c);
        curr.rebuild();

        let plain = PlainTermCount::<BigUint>::new(3, &curr);
        let counts = NovelTermCount::new(3, &curr, &prev, plain);
        let config = BalanceConfig {
            node_penalty: 0,
            profile_penalty: 0,
            child_size_penalty: 0,
        };
        let sampler = BalancedFrontierSampler::with_config(&counts, root, config);
        let seed = [17, 23];

        let mut rng = utils::combined_rng([3, seed[0], seed[1]]);
        let mut coverage = CoveragePolicy::new(config);
        let first_batch = (0..8)
            .map(|_| sampler.construct(root, 3, &mut coverage, &mut rng))
            .collect::<HashSet<_>>();
        assert!(
            first_batch.len() < 8,
            "fixture must contain an initial duplicate"
        );

        let terms = sampler
            .sample_size(root, 3, 8, seed)
            .expect("eight frontier terms");
        assert_eq!(terms.len(), 8);
    }
}
