//! Boltzmann Sampling for E-Graphs
//!
//! Provides methods for sampling terms from an e-graph with control over
//! term size distribution (via Boltzmann parameter λ).
//!
//! # Boltzmann Sampling
//!
//! Each term is weighted by `λ^size` where `λ ∈ (0, 1]`. Smaller λ values
//! bias toward smaller terms. The "critical" λ gives a target expected size.
//!
//! Iterative fixed-point computation handles cycles
//! correctly by converging to stable weights.

use hashbrown::HashMap;
use ordered_float::OrderedFloat;
use rand::prelude::*;

use super::graph::EGraph;
use super::ids::{DataChildId, DataId, EClassId, ExprChildId, NatId};
use super::nodes::{ENode, Label};
use super::tree::TreeNode;

/// Sampler using fixed-point iteration for weight computation.
///
/// This sampler handles cyclic e-graphs correctly by computing weights
/// via iterative convergence rather than recursion. Use this when your
/// e-graph may contain cycles.
///
/// Internally uses log-space arithmetic for numerical stability when
/// λ is small or trees are deep.
pub struct FixpointSampler<'a, L: Label, R: Rng + SeedableRng> {
    graph: &'a EGraph<L>,
    log_lambda: OrderedFloat<f64>,
    /// Log-weights for each e-class: log(W[id])
    log_weights: HashMap<EClassId, OrderedFloat<f64>>,
    rng: R,
    max_depth: usize,
}

/// Configuration for the fixed-point sampler.
#[derive(Debug, Clone, bon::Builder)]
#[builder(derive(Clone, Debug))]
pub struct FixpointSamplerConfig {
    /// Convergence threshold for weight computation.
    #[builder(default = 1e-3)]
    pub epsilon: f64,
    /// Maximum iterations for weight convergence.
    #[builder(default = 1000)]
    pub max_iterations: usize,
    /// Maximum depth during sampling (prevents infinite loops on cycles).
    #[builder(default = 1000)]
    pub max_depth: usize,
}

impl FixpointSamplerConfig {
    /// Config for graphs with cycles - uses smaller lambda for faster convergence.
    #[must_use]
    pub fn for_cyclic() -> Self {
        Self::builder().max_depth(100).build()
    }
}

/// Find the lambda value that produces trees of approximately the target size.
///
/// Uses binary search over lambda, estimating average tree size at each point
/// by sampling. Larger lambda values produce larger trees (monotonic relationship).
///
/// # Arguments
/// * `graph` - The e-graph to sample from
/// * `target_size` - The desired average tree size (number of nodes)
/// * `config` - Configuration for the search
/// * `rng` - Random number generator (will be used to create seedable sub-rngs)
///
/// # Returns
/// * `Ok((lambda, actual_avg_size))` - The found lambda and the actual average size achieved
/// * `Err(FindLambdaError)` - Error
///
/// # Errors
/// Errors if the search fails (e.g., target unreachable)
pub fn find_lambda_for_target_size<L: Label, R: Rng + SeedableRng>(
    graph: &EGraph<L>,
    target_size: usize,
    fixpoint_config: &FixpointSamplerConfig,
    rng: &mut R,
) -> Result<(f64, f64), FindLambdaError> {
    let mut lo = 0.001;

    // For cyclic graphs, we need to find the maximum lambda that converges.
    // Binary search to find the critical lambda (highest that still converges).
    let mut hi = find_critical_lambda(graph, fixpoint_config, rng)?;
    eprintln!("Critical lambda found: {hi}");

    // First, check bounds to ensure target is achievable
    let size_at_min =
        estimate_avg_size(graph, fixpoint_config, lo, R::seed_from_u64(rng.next_u64()))?;
    let size_at_max =
        estimate_avg_size(graph, fixpoint_config, hi, R::seed_from_u64(rng.next_u64()))?;
    eprintln!("Achievable size range: {size_at_min:.1} - {size_at_max:.1} (target: {target_size})");

    #[expect(clippy::cast_precision_loss)]
    let target = target_size as f64;

    if target < size_at_min {
        return Err(FindLambdaError::TargetTooSmall {
            target: target_size,
            min_achievable: size_at_min,
        });
    }
    if target > size_at_max {
        return Err(FindLambdaError::TargetTooLarge {
            target: target_size,
            max_achievable: size_at_max,
        });
    }

    // Binary search
    let mut best_lambda = f64::midpoint(lo, hi);
    let mut best_size = 0.0;

    while hi - lo > fixpoint_config.epsilon {
        let mid = f64::midpoint(lo, hi);
        let avg_size = estimate_avg_size(
            graph,
            fixpoint_config,
            mid,
            R::seed_from_u64(rng.next_u64()),
        )?;

        best_lambda = mid;
        best_size = avg_size;

        if avg_size < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok((best_lambda, best_size))
}

/// Find the critical lambda - the highest value where the sampler still converges.
///
/// For cyclic e-graphs, there's a threshold lambda above which the fixed-point
/// iteration diverges. This function uses binary search to find that threshold.
fn find_critical_lambda<L: Label, R: Rng + SeedableRng>(
    graph: &EGraph<L>,
    config: &FixpointSamplerConfig,
    rng: &mut R,
) -> Result<f64, FindLambdaError> {
    let mut lo = 0.001; // Known to converge (very small lambda)
    let mut hi = 1.0; // Upper bound to search

    // First verify that lo converges
    if FixpointSampler::new(graph, lo, config, R::seed_from_u64(rng.next_u64())).is_none() {
        return Err(FindLambdaError::SamplerDidNotConverge { lambda: lo });
    }

    // Binary search for the critical lambda
    while hi - lo > 0.001 {
        let mid = f64::midpoint(lo, hi);
        if FixpointSampler::new(graph, mid, config, R::seed_from_u64(rng.next_u64())).is_some() {
            lo = mid; // mid converges, try higher
        } else {
            hi = mid; // mid diverges, try lower
        }
    }

    // Return slightly below the critical point for safety margin
    Ok(lo * 0.99)
}

/// Estimate the average tree size for a given lambda by sampling.
fn estimate_avg_size<L: Label, R: Rng + SeedableRng>(
    graph: &EGraph<L>,
    config: &FixpointSamplerConfig,
    lambda: f64,
    rng: R,
) -> Result<f64, FindLambdaError> {
    let samples = FixpointSampler::new(graph, lambda, config, rng)
        .ok_or(FindLambdaError::SamplerDidNotConverge { lambda })?
        .take(1000)
        .collect::<Vec<_>>();

    if samples.is_empty() {
        return Err(FindLambdaError::SamplerDidNotConverge { lambda });
    }

    #[expect(clippy::cast_precision_loss)]
    let avg = samples.iter().map(|t| t.size()).sum::<usize>() as f64 / samples.len() as f64;

    Ok(avg)
}

/// Errors that can occur when finding lambda for a target size.
#[derive(Debug, Clone, thiserror::Error)]
pub enum FindLambdaError {
    /// The target size is smaller than achievable with the minimum lambda.
    #[error("target size {target} is too small; minimum achievable is {min_achievable:.1}")]
    TargetTooSmall { target: usize, min_achievable: f64 },
    /// The target size is larger than achievable with the maximum lambda.
    #[error("target size {target} is too large; maximum achievable is {max_achievable:.1}")]
    TargetTooLarge { target: usize, max_achievable: f64 },
    /// The sampler failed to converge for a given lambda.
    #[error("sampler did not converge for lambda={lambda}")]
    SamplerDidNotConverge { lambda: f64 },
}

impl<'a, L: Label, R: Rng + SeedableRng> FixpointSampler<'a, L, R> {
    /// Create a new fixed-point sampler.
    ///
    /// Computes weights via fixed-point iteration until convergence.
    /// Uses log-space arithmetic internally for numerical stability.
    ///
    /// # Returns
    /// `None` if weight computation does not converge, otherwise the sampler.
    ///
    /// # Panics
    /// Panics if `lambda` is not in the range (0, 1].
    pub fn new(
        graph: &'a EGraph<L>,
        lambda: f64,
        config: &FixpointSamplerConfig,
        rng: R,
    ) -> Option<Self> {
        assert!(lambda > 0.0 && lambda <= 1.0, "λ must be in (0, 1]");

        let log_lambda = OrderedFloat(lambda.ln());

        // Collect canonical class IDs and initialize log-weights to log(λ)
        let mut log_weights = graph
            .class_ids()
            .map(|id| (graph.canonicalize(id), log_lambda))
            .collect::<HashMap<_, _>>();
        let class_ids = log_weights.keys().copied().collect::<Vec<_>>();

        let mut prev_max_delta = f64::INFINITY;
        let mut divergence_count = 0;

        for _ in 0..config.max_iterations {
            let mut max_delta = 0.0_f64;

            for &id in &class_ids {
                // new_log_weight = log(sum over nodes of (λ × product of child weights))
                //                = log_sum_exp(log(λ) + sum of log(child weights))
                let node_log_weights = graph.class(id).nodes().iter().map(|node| {
                    node.children()
                        .iter()
                        .map(|child| match child {
                            ExprChildId::EClass(eid) => log_weights[&graph.canonicalize(*eid)],
                            ExprChildId::Nat(nat_id) => {
                                Self::nat_log_weight(graph, *nat_id, log_lambda)
                            }
                            ExprChildId::Data(dt_id) => {
                                Self::dt_log_weight(graph, *dt_id, log_lambda)
                            }
                        })
                        .sum::<OrderedFloat<f64>>()
                        + log_lambda
                });

                let max = node_log_weights
                    .clone()
                    .max()
                    .expect("log_sum_exp requires non-empty input");
                let new_log_weight =
                    max + node_log_weights.map(|v| (v - max).exp()).sum::<f64>().ln();

                // Convergence check: |Δ log w| < ε means relative change |w_new/w_old - 1| ≈ ε
                let delta = (new_log_weight - log_weights[&id]).abs();
                max_delta = max_delta.max(delta);
                log_weights.insert(id, new_log_weight);
            }
            if max_delta < config.epsilon {
                return Some(Self {
                    graph,
                    log_lambda,
                    log_weights,
                    rng,
                    max_depth: config.max_depth,
                });
            }

            // Detect divergence: if max_delta is consistently increasing, we're diverging
            if max_delta > prev_max_delta * 1.5 {
                divergence_count += 1;
                if divergence_count >= 3 {
                    // Weights are diverging - lambda is too large for this cyclic graph
                    eprintln!("Divergence detected: lambda={lambda} is too large");
                    return None;
                }
            } else {
                divergence_count = 0;
            }
            prev_max_delta = max_delta;
        }
        None
    }

    fn nat_log_weight(
        graph: &EGraph<L>,
        id: NatId,
        log_lambda: OrderedFloat<f64>,
    ) -> OrderedFloat<f64> {
        graph
            .nat(id)
            .children()
            .iter()
            .map(|child_id| Self::nat_log_weight(graph, *child_id, log_lambda))
            .sum::<OrderedFloat<f64>>()
            + log_lambda
    }

    fn dt_log_weight(
        graph: &EGraph<L>,
        id: DataId,
        log_lambda: OrderedFloat<f64>,
    ) -> OrderedFloat<f64> {
        graph
            .data_ty(id)
            .children()
            .iter()
            .map(|child_id| match child_id {
                DataChildId::Nat(nat_id) => Self::nat_log_weight(graph, *nat_id, log_lambda),
                DataChildId::DataType(data_id) => Self::dt_log_weight(graph, *data_id, log_lambda),
            })
            .sum::<OrderedFloat<f64>>()
            + log_lambda
    }

    /// Compute log-weight for a node (sum of children log-weights + log(λ)).
    fn node_log_weight(&self, node: &ENode<L>) -> OrderedFloat<f64> {
        node.children()
            .iter()
            .map(|child| match child {
                ExprChildId::EClass(eid) => {
                    let c = self.graph.canonicalize(*eid);
                    self.log_weights[&c]
                }
                ExprChildId::Nat(nat_id) => {
                    Self::nat_log_weight(self.graph, *nat_id, self.log_lambda)
                }
                ExprChildId::Data(dt_id) => {
                    Self::dt_log_weight(self.graph, *dt_id, self.log_lambda)
                }
            })
            .sum::<OrderedFloat<f64>>()
            + self.log_lambda
    }

    /// Sample a term rooted at the given e-class.
    fn sample_from(&mut self, id: EClassId, depth: usize) -> Option<TreeNode<L>> {
        if depth >= self.max_depth {
            return None;
        }

        let nodes = self.graph.class(id).nodes();

        // Compute log-weights for each node
        let log_weights = nodes
            .iter()
            .map(|node| (node, self.node_log_weight(node)))
            .collect::<HashMap<_, _>>();

        // Convert to probabilities using softmax (numerically stable)
        let max_log = log_weights.values().max().expect("e-class has no nodes");

        let chosen_node = nodes
            // Compute exp(log_w - max) for numerical stability
            .choose_weighted(&mut self.rng, |node| (log_weights[node] - max_log).exp())
            .expect("weights are always positive");

        Some(TreeNode::new_typed(
            chosen_node.label().clone(),
            chosen_node
                .children()
                .iter()
                .map(|c| self.sample_child(c, depth + 1))
                .collect::<Option<_>>()?,
            Some(TreeNode::from_eclass(self.graph, id)),
        ))
    }

    /// Sample a child, dispatching on child type.
    fn sample_child(&mut self, child: &ExprChildId, depth: usize) -> Option<TreeNode<L>> {
        match child {
            ExprChildId::EClass(eid) => self.sample_from(*eid, depth),
            ExprChildId::Nat(nid) => Some(TreeNode::from_nat(self.graph, *nid)),
            ExprChildId::Data(did) => Some(TreeNode::from_data(self.graph, *did)),
        }
    }

    /// Create a sampler tuned to produce trees of approximately `target_size` nodes.
    ///
    /// Finds the appropriate lambda via binary search, then constructs the sampler.
    ///
    /// # Returns
    /// The sampler and the `(lambda, expected_size)` that were found.
    ///
    /// # Errors
    /// Errors when it can't find a lambda
    pub fn for_target_size(
        graph: &'a EGraph<L>,
        target_size: usize,
        config: &FixpointSamplerConfig,
        rng: &mut R,
    ) -> Result<(Self, f64, f64), FindLambdaError> {
        let (lambda, expected_size) = find_lambda_for_target_size(graph, target_size, config, rng)?;
        let sampler = Self::new(graph, lambda, config, R::seed_from_u64(rng.next_u64()))
            .ok_or(FindLambdaError::SamplerDidNotConverge { lambda })?;
        Ok((sampler, lambda, expected_size))
    }
}

impl<L: Label, R: Rng + SeedableRng> Iterator for FixpointSampler<'_, L, R> {
    type Item = TreeNode<L>;

    fn next(&mut self) -> Option<Self::Item> {
        self.sample_from(self.graph.root(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EClass;
    use crate::ids::NatId;
    use crate::ids::TypeChildId;
    use crate::nodes::{ENode, NatNode};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn dummy_ty() -> TypeChildId {
        TypeChildId::Nat(NatId::new(0))
    }

    fn dummy_nat_nodes() -> HashMap<NatId, NatNode<String>> {
        let mut map = HashMap::new();
        map.insert(NatId::new(0), NatNode::leaf("0".to_owned()));
        map
    }

    fn eid(i: usize) -> ExprChildId {
        ExprChildId::EClass(EClassId::new(i))
    }

    fn cfv(classes: Vec<EClass<String>>) -> HashMap<EClassId, EClass<String>> {
        classes
            .into_iter()
            .enumerate()
            .map(|(i, c)| (EClassId::new(i), c))
            .collect()
    }

    fn cyclic_graph() -> EGraph<String> {
        // Class 0: "f" with child Class 0 (cycle!), or leaf "x"
        // This represents: x, f(x), f(f(x)), f(f(f(x))), ...
        EGraph::new(
            cfv(vec![EClass::new(
                vec![
                    ENode::new("f".to_owned(), vec![eid(0)]), // cycle back to self
                    ENode::leaf("x".to_owned()),
                ],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        )
    }

    #[test]
    fn fixpoint_handles_cycles() {
        let graph = cyclic_graph();
        let config = FixpointSamplerConfig::for_cyclic();
        let rng = StdRng::seed_from_u64(42);

        let mut sampler = FixpointSampler::new(&graph, 0.5, &config, rng)
            .expect("Should converge with λ < 1 on cyclic graph");

        // Should be able to sample without infinite loop
        let terms = sampler.by_ref().take(100).collect::<Vec<_>>();
        assert!(!terms.is_empty(), "Should produce some terms");

        // With small lambda, most terms should be small
        let small_terms = terms.iter().filter(|t| t.label() == "x").count();
        assert!(
            small_terms > 30,
            "With λ=0.5, should prefer leaf 'x', got {small_terms}/100"
        );
    }

    #[test]
    fn fixpoint_weights_converge() {
        let graph = cyclic_graph();

        // With λ = 0.5, the weight equation for the cyclic class is:
        // W = λ × W + λ  (from "f" with child + "x" leaf)
        // W = 0.5W + 0.5
        // 0.5W = 0.5
        // W = 1.0
        //
        // This means P(x) = 0.5/1.0 = 50%, P(f(...)) = 50%
        let config = FixpointSamplerConfig::builder().max_depth(100).build();
        let rng = StdRng::seed_from_u64(42);
        let mut sampler = FixpointSampler::new(&graph, 0.5, &config, rng).unwrap();

        // Verify the probability distribution matches theory
        let leaf_count = sampler
            .by_ref()
            .take(1000)
            .filter(|t| t.label() == "x")
            .count();

        // Should be close to 50% (allowing for variance)
        assert!(
            (400..600).contains(&leaf_count),
            "Expected ~50% leaves, got {leaf_count}/1000"
        );
    }

    #[test]
    fn sampling_yields_samples() {
        let graph = cyclic_graph();
        let config = FixpointSamplerConfig::for_cyclic();
        let rng = StdRng::seed_from_u64(42);

        let mut fp_sampler = FixpointSampler::new(&graph, 0.5, &config, rng)
            .expect("Should converge with λ < 1 on cyclic graph");

        let samples = fp_sampler.by_ref().take(50).collect::<Vec<_>>();

        assert_eq!(samples.len(), 50, "Should yield exactly 50 samples");

        // All samples should have valid root labels
        for sample in &samples {
            assert!(
                sample.label() == "f" || sample.label() == "x",
                "Unexpected root label: {}",
                sample.label()
            );
        }
    }

    #[test]
    fn find_lambda_for_target_size_works() {
        let graph = cyclic_graph();

        let mut rng = StdRng::seed_from_u64(42);
        let config = FixpointSamplerConfig::builder().build();

        // Target size of 2 (e.g., f(x) has 2 nodes)
        let result = find_lambda_for_target_size(&graph, 2, &config, &mut rng);
        assert!(
            result.is_ok(),
            "Should find a lambda for target size 2: {:?}",
            result.err()
        );

        let (lambda, avg_size) = result.unwrap();
        assert!(lambda > 0.0 && lambda < 1.0, "Lambda should be in (0, 1)");
        // Allow some variance in the achieved size
        assert!(
            (1.5..4.0).contains(&avg_size),
            "Average size {avg_size} should be roughly near target 2"
        );
    }
}
