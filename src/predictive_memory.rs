//! Opt-in next-iteration memory prediction.
//!
//! The supplied model must be calibrated for the workload. Callers explicitly
//! opt in with a model path; this is not a general-purpose egg memory model.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use egg::{Analysis, Iteration, IterationData, Language, Runner, SchedulerSnapshot};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;

const BASE_FEATURE_NAMES: [&str; 16] = [
    "egraph_nodes",
    "egraph_classes",
    "nodes_per_class",
    "allocated",
    "bytes_per_node",
    "prev_growth",
    "prev_node_growth",
    "total_applied",
    "hook_time",
    "search_time",
    "apply_time",
    "rebuild_time",
    "total_time",
    "n_rebuilds",
    "iter_index",
    "term_size",
];
const SCHEDULER_FEATURE_NAMES: [&str; 8] = [
    "n_active",
    "n_banned",
    "n_newly_unbanned",
    "min_ban_remaining",
    "total_times_banned",
    "max_active_log2_match_limit",
    "log2_active_match_limit_sum",
    "max_active_times_banned",
];
const RULE_FEATURE_SUFFIXES: [&str; 5] = [
    "will_search",
    "newly_unbanned",
    "times_banned",
    "ban_remaining",
    "log2_match_limit",
];

#[derive(Debug, Deserialize)]
struct ModelManifest {
    schema_version: u32,
    features: Vec<String>,
    scheduler: String,
    rules: Vec<String>,
    safety_margin: f64,
    rust_parity_sample: ParitySample,
}

#[derive(Debug, Deserialize)]
struct ParitySample {
    features: Vec<f32>,
    sklearn_prediction: f64,
    onnx_prediction: f64,
    absolute_tolerance: f64,
}

pub(crate) struct OnnxMemoryGrowthPredictor {
    session: Session,
    safety_margin: f64,
    features: Vec<String>,
    scheduler: String,
    rules: Vec<String>,
    parity_sample: ParitySample,
}

fn escape_rule_name(name: &str) -> String {
    let mut escaped = String::new();
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'-' {
            escaped.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            write!(escaped, "%{byte:02X}").expect("writing to String cannot fail");
        }
    }
    escaped
}

fn rule_feature_name(rule: &str, suffix: &str) -> String {
    format!("rule_{}_{}", escape_rule_name(rule), suffix)
}

fn expected_feature_names(rules: &[String]) -> Vec<String> {
    BASE_FEATURE_NAMES
        .into_iter()
        .chain(SCHEDULER_FEATURE_NAMES)
        .map(str::to_owned)
        .chain(rules.iter().flat_map(|rule| {
            RULE_FEATURE_SUFFIXES
                .into_iter()
                .map(move |suffix| rule_feature_name(rule, suffix))
        }))
        .collect()
}

fn validate_manifest(manifest: &ModelManifest, path: &Path) -> Result<(), String> {
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported memory-model schema version {} in {}; expected 1",
            manifest.schema_version,
            path.display()
        ));
    }
    if manifest.scheduler != "backoff" {
        return Err(format!(
            "unsupported memory-model scheduler {:?} in {}; expected \"backoff\"",
            manifest.scheduler,
            path.display()
        ));
    }
    if !manifest.safety_margin.is_finite() {
        return Err(format!(
            "memory-model safety margin in {} is non-finite",
            path.display()
        ));
    }
    let unique_rules: HashSet<_> = manifest.rules.iter().collect();
    if unique_rules.len() != manifest.rules.len() {
        return Err(format!(
            "memory-model rule list in {} contains duplicates",
            path.display()
        ));
    }
    let mut sorted_rules = manifest.rules.clone();
    sorted_rules.sort();
    if sorted_rules != manifest.rules {
        return Err(format!(
            "memory-model rule list in {} is not deterministically sorted",
            path.display()
        ));
    }
    let expected = expected_feature_names(&manifest.rules);
    if manifest.features != expected {
        return Err(format!(
            "memory-model feature schema mismatch in {}: Rust can produce {:?}, manifest lists {:?}",
            path.display(),
            expected,
            manifest.features
        ));
    }
    let unique_features: HashSet<_> = manifest.features.iter().collect();
    if unique_features.len() != manifest.features.len() {
        return Err(format!(
            "memory-model feature schema in {} contains duplicate names",
            path.display()
        ));
    }
    if manifest.rust_parity_sample.features.len() != manifest.features.len() {
        return Err(format!(
            "memory-model parity sample in {} has {} values for {} features",
            path.display(),
            manifest.rust_parity_sample.features.len(),
            manifest.features.len()
        ));
    }
    if !manifest.rust_parity_sample.absolute_tolerance.is_finite()
        || manifest.rust_parity_sample.absolute_tolerance < 0.0
        || !manifest.rust_parity_sample.sklearn_prediction.is_finite()
        || !manifest.rust_parity_sample.onnx_prediction.is_finite()
        || manifest
            .rust_parity_sample
            .features
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(format!(
            "memory-model parity sample in {} contains non-finite or invalid values",
            path.display()
        ));
    }
    Ok(())
}

impl OnnxMemoryGrowthPredictor {
    fn manifest_path(model_path: &Path) -> PathBuf {
        model_path.with_extension("json")
    }

    fn load(model_path: &Path) -> Result<Self, String> {
        let manifest_path = Self::manifest_path(model_path);
        let manifest_json = std::fs::read_to_string(&manifest_path).map_err(|error| {
            format!(
                "failed to read memory-model manifest {}: {error}",
                manifest_path.display()
            )
        })?;
        let manifest: ModelManifest = serde_json::from_str(&manifest_json).map_err(|error| {
            format!(
                "invalid memory-model manifest {}: {error}",
                manifest_path.display()
            )
        })?;
        validate_manifest(&manifest, &manifest_path)?;
        let session = Session::builder()
            .map_err(|error| error.to_string())?
            .with_intra_threads(1)
            .map_err(|error| error.to_string())?
            .commit_from_file(model_path)
            .map_err(|error| {
                format!(
                    "failed to load ONNX memory model {}: {error}",
                    model_path.display()
                )
            })?;
        Ok(Self {
            session,
            safety_margin: manifest.safety_margin,
            features: manifest.features,
            scheduler: manifest.scheduler,
            rules: manifest.rules,
            parity_sample: manifest.rust_parity_sample,
        })
    }

    /// Construct and prewarm one predictor for one eqsat run. Call this before
    /// the Runner exists, so the session's own allocation settles before any
    /// reading is taken against the ceiling.
    pub(crate) fn load_and_prewarm(model_path: &Path) -> Result<Self, String> {
        let mut predictor = Self::load(model_path)?;
        let sample = predictor.parity_sample.features.clone();
        let predicted = predictor.predict_log_growth(&sample)?;
        let expected_onnx = predictor.parity_sample.onnx_prediction;
        let expected_sklearn = predictor.parity_sample.sklearn_prediction;
        let tolerance = predictor.parity_sample.absolute_tolerance;
        if (predicted - expected_onnx).abs() > tolerance
            || (predicted - expected_sklearn).abs() > tolerance
        {
            return Err(format!(
                "Rust/sklearn/ONNX parity prewarm failed for {}: Rust {predicted}, \
                 sklearn {expected_sklearn}, ONNX {expected_onnx} (tolerance {tolerance})",
                model_path.display()
            ));
        }
        Ok(predictor)
    }

    fn predict_log_growth(&mut self, features: &[f32]) -> Result<f64, String> {
        if features.len() != self.features.len() {
            return Err(format!(
                "memory-model input has {} values for {} manifest features",
                features.len(),
                self.features.len()
            ));
        }
        let input = Tensor::from_array(([1_usize, features.len()], features.to_vec()))
            .map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs!["features" => input])
            .map_err(|error| error.to_string())?;
        let (_, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        values
            .first()
            .copied()
            .map(f64::from)
            .ok_or_else(|| "memory model returned an empty output".to_owned())
    }
}

#[derive(Debug, Clone, Copy)]
struct IterationSnapshot {
    egraph_nodes: usize,
    total_applied: usize,
    hook_time: f64,
    search_time: f64,
    apply_time: f64,
    rebuild_time: f64,
    total_time: f64,
    n_rebuilds: usize,
}

fn sum_applied_counts<'a>(counts: impl Iterator<Item = &'a usize>) -> usize {
    counts.copied().fold(0_usize, usize::saturating_add)
}

impl<D> From<&Iteration<D>> for IterationSnapshot {
    fn from(iteration: &Iteration<D>) -> Self {
        Self {
            egraph_nodes: iteration.egraph_nodes,
            total_applied: sum_applied_counts(iteration.applied.values()),
            hook_time: iteration.hook_time,
            search_time: iteration.search_time,
            apply_time: iteration.apply_time,
            rebuild_time: iteration.rebuild_time,
            total_time: iteration.total_time,
            n_rebuilds: iteration.n_rebuilds,
        }
    }
}

fn finite_ratio(numerator: f64, denominator: f64) -> f64 {
    let ratio = numerator / denominator;
    if ratio.is_finite() { ratio } else { 1.0 }
}

fn validate_scheduler_compatibility(
    scheduler: &SchedulerSnapshot,
    expected_scheduler: &str,
    expected_rules: &[String],
) -> Result<(), String> {
    if scheduler.scheduler != expected_scheduler {
        return Err(format!(
            "memory model requires {expected_scheduler:?} scheduler state, runner exposes {:?}",
            scheduler.scheduler
        ));
    }
    let actual_rules: HashSet<_> = scheduler
        .rules
        .iter()
        .map(|rule| rule.name.as_str())
        .collect();
    if actual_rules.len() != scheduler.rules.len() {
        return Err("runner scheduler snapshot contains duplicate rule names".to_owned());
    }
    let expected_rule_set: HashSet<_> = expected_rules.iter().map(String::as_str).collect();
    if actual_rules != expected_rule_set {
        let mut actual: Vec<_> = actual_rules.into_iter().collect();
        actual.sort_unstable();
        return Err(format!(
            "memory-model rule-set compatibility error: manifest requires {:?}, runner exposes {:?}",
            expected_rules, actual
        ));
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the deployed ONNX interface intentionally uses float32 features"
)]
fn build_features(
    previous_iteration: IterationSnapshot,
    current_egraph_nodes: usize,
    current_egraph_classes: usize,
    allocated: u64,
    previous_allocated: Option<u64>,
    iter_index: usize,
    term_size: usize,
    scheduler: &SchedulerSnapshot,
    expected_scheduler: &str,
    expected_rules: &[String],
) -> Result<Vec<f32>, String> {
    validate_scheduler_compatibility(scheduler, expected_scheduler, expected_rules)?;
    let by_name: HashMap<_, _> = scheduler
        .rules
        .iter()
        .map(|rule| (rule.name.as_str(), rule))
        .collect();

    let nodes = current_egraph_nodes as f64;
    let classes = current_egraph_classes as f64;
    let allocated_f64 = allocated as f64;
    let mut values = vec![
        nodes,
        classes,
        finite_ratio(nodes, classes),
        allocated_f64,
        finite_ratio(allocated_f64, nodes),
        previous_allocated.map_or(1.0, |value| finite_ratio(allocated_f64, value as f64)),
        finite_ratio(nodes, previous_iteration.egraph_nodes as f64),
        previous_iteration.total_applied as f64,
        previous_iteration.hook_time,
        previous_iteration.search_time,
        previous_iteration.apply_time,
        previous_iteration.rebuild_time,
        previous_iteration.total_time,
        previous_iteration.n_rebuilds as f64,
        iter_index as f64,
        term_size as f64,
        scheduler.n_active as f64,
        scheduler.n_banned as f64,
        scheduler.n_newly_unbanned as f64,
        scheduler.min_ban_remaining as f64,
        scheduler.total_times_banned as f64,
        scheduler.max_active_log2_match_limit,
        scheduler.log2_active_match_limit_sum,
        scheduler.max_active_times_banned as f64,
    ];
    for rule_name in expected_rules {
        let rule = by_name[rule_name.as_str()];
        values.extend([
            if rule.will_search { 1.0 } else { 0.0 },
            if rule.newly_unbanned { 1.0 } else { 0.0 },
            rule.times_banned as f64,
            rule.ban_remaining as f64,
            rule.log2_match_limit,
        ]);
    }
    Ok(values
        .into_iter()
        .map(|value| {
            let converted = value as f32;
            if converted.is_finite() {
                converted
            } else {
                1.0
            }
        })
        .collect())
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounds and finiteness are checked before the saturating byte conversion"
)]
/// Project the upcoming iteration's absolute peak live heap. `allocated` is
/// already an absolute pre-search reading, and the model predicts peak growth
/// as a ratio against it, so the result is absolute without rebasing.
fn predicted_next_absolute(allocated: u64, predicted_log_growth: f64, safety_margin: f64) -> u64 {
    let upper_log_growth = predicted_log_growth + safety_margin;
    if !upper_log_growth.is_finite() {
        return u64::MAX;
    }
    let predicted = (allocated as f64) * upper_log_growth.exp();
    if !predicted.is_finite() || predicted > u64::MAX as f64 {
        return u64::MAX;
    }
    predicted.ceil() as u64
}

fn predictive_decision(
    predicted_log_growth: f64,
    allocated: u64,
    absolute_limit: u64,
    safety_margin: f64,
) -> Option<u64> {
    let predicted = predicted_next_absolute(allocated, predicted_log_growth, safety_margin);
    (predicted >= absolute_limit).then_some(predicted)
}

/// Hook invoked before upcoming iteration `k`, using iteration `k - 1`'s
/// completed work metadata, the runner's actual current egraph/allocation, and
/// the scheduler snapshot already captured for iteration `k`.
pub(crate) fn hook<L, N, D>(
    term_size: usize,
    mut predictor: OnnxMemoryGrowthPredictor,
) -> impl FnMut(&mut Runner<L, N, D>) -> Result<(), String> + 'static
where
    L: Language,
    N: Analysis<L>,
    D: IterationData<L, N>,
{
    let mut previous_allocated = None;
    move |runner| {
        // This runs even for iteration zero, before the first rewrite search,
        // so incompatible schedulers/rule sets fail before eqsat work begins.
        validate_scheduler_compatibility(
            &runner.scheduler_snapshot,
            &predictor.scheduler,
            &predictor.rules,
        )?;
        let allocated = runner
            .memory_reading()
            .expect("predictive runner has memory tracking");
        let Some(previous_iteration) = runner.iterations.last() else {
            // Iteration zero is not predictable (there is no prior work), but
            // its decision-boundary allocation is the history needed for the
            // first supervised/online row at iteration one.
            previous_allocated = Some(allocated);
            return Ok(());
        };
        let features = build_features(
            previous_iteration.into(),
            runner.egraph.total_size(),
            runner.egraph.number_of_classes(),
            allocated,
            previous_allocated,
            runner.iterations.len(),
            term_size,
            &runner.scheduler_snapshot,
            &predictor.scheduler,
            &predictor.rules,
        )?;
        previous_allocated = Some(allocated);

        let absolute_limit = runner
            .absolute_memory_limit()
            .expect("predictive hook requires an absolute memory limit");
        let safety_margin = predictor.safety_margin;
        let predicted_log_growth = predictor.predict_log_growth(&features)?;
        if let Some(predicted) = predictive_decision(
            predicted_log_growth,
            allocated,
            absolute_limit,
            safety_margin,
        ) {
            return Err(format!(
                "predicted upcoming-iteration peak memory limit crossing \
                 ({predicted} >= {absolute_limit} bytes)"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::SchedulerRuleState;

    fn iteration(nodes: usize, applied: usize) -> IterationSnapshot {
        IterationSnapshot {
            egraph_nodes: nodes,
            total_applied: applied,
            hook_time: 0.1,
            search_time: 0.2,
            apply_time: 0.3,
            rebuild_time: 0.4,
            total_time: 1.0,
            n_rebuilds: 2,
        }
    }

    fn scheduler() -> SchedulerSnapshot {
        SchedulerSnapshot {
            scheduler: "backoff",
            n_active: 1,
            n_banned: 0,
            n_newly_unbanned: 1,
            min_ban_remaining: 0,
            total_times_banned: 2,
            max_active_log2_match_limit: 5.0,
            log2_active_match_limit_sum: 5.044_394,
            max_active_times_banned: 2,
            rules: vec![SchedulerRuleState {
                name: "assoc_add".into(),
                will_search: true,
                newly_unbanned: true,
                times_banned: 2,
                ban_remaining: 0,
                match_limit: 32,
                log2_match_limit: 5.0,
            }],
        }
    }

    fn features(
        previous_nodes: usize,
        current_nodes: usize,
        classes: usize,
        allocated: u64,
        previous_allocated: Option<u64>,
    ) -> Vec<f32> {
        build_features(
            iteration(previous_nodes, 7),
            current_nodes,
            classes,
            allocated,
            previous_allocated,
            3,
            11,
            &scheduler(),
            "backoff",
            &["assoc_add".to_owned()],
        )
        .unwrap()
    }

    fn assert_feature(actual: f32, expected: f32) {
        assert!((actual - expected).abs() <= f32::EPSILON);
    }

    #[test]
    fn deterministic_schema_escapes_arbitrary_rule_names() {
        let names = expected_feature_names(&["assoc-add".to_owned(), "x_y/λ".to_owned()]);
        assert_eq!(names[0], "egraph_nodes");
        assert_eq!(names[15], "term_size");
        assert_eq!(
            &names[16..24],
            &[
                "n_active",
                "n_banned",
                "n_newly_unbanned",
                "min_ban_remaining",
                "total_times_banned",
                "max_active_log2_match_limit",
                "log2_active_match_limit_sum",
                "max_active_times_banned",
            ]
        );
        assert_eq!(names[24], "rule_assoc-add_will_search");
        assert_eq!(names[29], "rule_x%5Fy%2F%CE%BB_will_search");
    }

    #[test]
    fn missing_allocation_history_defaults_but_current_egraph_is_live() {
        let features = features(20, 30, 5, 1_000, None);
        assert_feature(features[5], 1.0);
        assert_feature(features[0], 30.0);
        assert_feature(features[1], 5.0);
        assert_feature(features[6], 1.5);
    }

    #[test]
    fn subsequent_iteration_calculates_both_growth_features() {
        let features = features(20, 30, 5, 1_500, Some(1_000));
        assert_feature(features[5], 1.5);
        assert_feature(features[6], 1.5);
    }

    #[test]
    fn online_vector_matches_equivalent_offline_decision_row() {
        assert_eq!(sum_applied_counts([&2, &5, &10].into_iter()), 17);
        let features = features(20, 30, 5, 1_500, Some(1_000));
        let offline_decision_row = vec![
            // Current pre-search egraph/allocation.
            30.0, 5.0, 6.0, 1_500.0, 50.0, 1.5, 1.5,
            // Previous iteration's completed work.
            7.0, 0.1, 0.2, 0.3, 0.4, 1.0, 2.0,
            // Upcoming identity and scheduler snapshot.
            3.0, 11.0, 1.0, 0.0, 1.0, 0.0, 2.0, 5.0, 5.044_394, 2.0,
            // Upcoming assoc_add state.
            1.0, 1.0, 2.0, 0.0, 5.0,
        ];
        assert_eq!(features, offline_decision_row);
    }

    #[test]
    fn zero_sizes_and_allocation_produce_only_finite_features() {
        let features = features(0, 0, 0, 0, Some(0));
        assert!(features.into_iter().all(f32::is_finite));
    }

    #[test]
    fn incompatible_scheduler_or_rule_set_is_an_error() {
        let snapshot = scheduler();
        let scheduler_error = build_features(
            iteration(1, 0),
            1,
            1,
            1,
            None,
            1,
            1,
            &snapshot,
            "generic",
            &["assoc_add".to_owned()],
        )
        .unwrap_err();
        assert!(scheduler_error.contains("requires"));
        let rule_error = build_features(
            iteration(1, 0),
            1,
            1,
            1,
            None,
            1,
            1,
            &snapshot,
            "backoff",
            &["different".to_owned()],
        )
        .unwrap_err();
        assert!(rule_error.contains("rule-set compatibility"));
    }

    #[test]
    fn prediction_below_ceiling_continues() {
        assert_eq!(predictive_decision(0.0, 300, 301, 0.0), None);
    }

    #[test]
    fn prediction_at_or_above_ceiling_stops() {
        assert_eq!(predictive_decision(0.0, 300, 300, 0.0), Some(300));
    }

    /// Zero predicted growth projects the current reading unchanged: the
    /// prediction scales the reading and adds no offset of its own, which is
    /// what keeps it comparable to the configured ceiling.
    #[test]
    fn zero_growth_projects_the_current_reading() {
        assert_eq!(predicted_next_absolute(300, 0.0, 0.0), 300);
    }

    #[test]
    fn non_finite_and_overflowing_predictions_stop_conservatively() {
        assert_eq!(predicted_next_absolute(200, f64::NAN, 0.0), u64::MAX);
        assert_eq!(predicted_next_absolute(200, f64::MAX, 0.0), u64::MAX);
    }

    #[test]
    fn checked_in_model_loads_prewarms_and_matches_parity_sample() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/memory_growth.onnx");
        OnnxMemoryGrowthPredictor::load_and_prewarm(&model_path)
            .expect("checked-in model and manifest must load and prewarm");
    }
}
