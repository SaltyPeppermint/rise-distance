//! Opt-in next-iteration memory prediction.
//!
//! The supplied model must be calibrated for the workload. Callers explicitly
//! opt in with a model path; this is not a general-purpose egg memory model.

use std::path::{Path, PathBuf};

use egg::{Analysis, Iteration, IterationData, Language, Runner};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;

use crate::eqsat::RunHeap;

pub(crate) const FEATURE_NAMES: [&str; 16] = [
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
const FEATURE_COUNT: usize = FEATURE_NAMES.len();

#[derive(Debug, Deserialize)]
struct ModelManifest {
    features: Vec<String>,
    safety_margin: f64,
    #[cfg(test)]
    rust_parity_sample: ParitySample,
}

#[cfg(test)]
#[derive(Debug, Deserialize)]
struct ParitySample {
    features: [f32; FEATURE_COUNT],
    sklearn_prediction: f64,
    absolute_tolerance: f64,
}

trait MemoryGrowthPredictor {
    fn predict_log_growth(&mut self, features: &[f32; FEATURE_COUNT]) -> Result<f64, String>;
}

pub(crate) struct OnnxMemoryGrowthPredictor {
    session: Session,
    safety_margin: f64,
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
        if manifest.features != FEATURE_NAMES {
            return Err(format!(
                "memory-model feature order mismatch in {}: {:?}",
                manifest_path.display(),
                manifest.features
            ));
        }
        if !manifest.safety_margin.is_finite() {
            return Err(format!(
                "memory-model safety margin in {} is non-finite",
                manifest_path.display()
            ));
        }
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
        })
    }

    /// Construct and prewarm one predictor for one eqsat run. Call this before
    /// capturing the run's heap baseline.
    pub(crate) fn load_and_prewarm(model_path: &Path) -> Result<Self, String> {
        let mut predictor = Self::load(model_path)?;
        predictor.predict_log_growth(&[0.0; FEATURE_COUNT])?;
        Ok(predictor)
    }
}

impl MemoryGrowthPredictor for OnnxMemoryGrowthPredictor {
    fn predict_log_growth(&mut self, features: &[f32; FEATURE_COUNT]) -> Result<f64, String> {
        let input = Tensor::from_array(([1_usize, FEATURE_COUNT], features.to_vec()))
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
    egraph_classes: usize,
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
            egraph_classes: iteration.egraph_classes,
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

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the deployed ONNX interface intentionally uses float32 features"
)]
fn build_features(
    last: IterationSnapshot,
    previous: Option<IterationSnapshot>,
    allocated: u64,
    previous_allocated: Option<u64>,
    iter_index: usize,
    term_size: usize,
) -> [f32; FEATURE_COUNT] {
    let nodes = last.egraph_nodes as f64;
    let classes = last.egraph_classes as f64;
    let allocated_f64 = allocated as f64;
    let values = [
        nodes,
        classes,
        finite_ratio(nodes, classes),
        allocated_f64,
        finite_ratio(allocated_f64, nodes),
        previous_allocated.map_or(1.0, |value| finite_ratio(allocated_f64, value as f64)),
        previous.map_or(1.0, |value| finite_ratio(nodes, value.egraph_nodes as f64)),
        last.total_applied as f64,
        last.hook_time,
        last.search_time,
        last.apply_time,
        last.rebuild_time,
        last.total_time,
        last.n_rebuilds as f64,
        iter_index as f64,
        term_size as f64,
    ];
    values.map(|value| {
        let converted = value as f32;
        if converted.is_finite() {
            converted
        } else {
            1.0
        }
    })
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounds and finiteness are checked before the saturating byte conversion"
)]
fn predicted_next_absolute(
    baseline: u64,
    allocated: u64,
    predicted_log_growth: f64,
    safety_margin: f64,
) -> u64 {
    let upper_log_growth = predicted_log_growth + safety_margin;
    if !upper_log_growth.is_finite() {
        return u64::MAX;
    }
    let predicted_delta = (allocated as f64) * upper_log_growth.exp();
    if !predicted_delta.is_finite() || predicted_delta > u64::MAX as f64 {
        return u64::MAX;
    }
    baseline.saturating_add(predicted_delta.ceil() as u64)
}

fn predictive_decision<P: MemoryGrowthPredictor>(
    predictor: &mut P,
    features: &[f32; FEATURE_COUNT],
    baseline: u64,
    allocated: u64,
    absolute_limit: u64,
    safety_margin: f64,
) -> Result<Option<u64>, String> {
    let prediction = predictor.predict_log_growth(features)?;
    let predicted = predicted_next_absolute(baseline, allocated, prediction, safety_margin);
    Ok((predicted >= absolute_limit).then_some(predicted))
}

/// Hook invoked before iteration `i + 1`, using iteration `i`'s egg metadata.
pub(crate) fn hook<L, N, D>(
    heap: RunHeap,
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
        let Some((last, earlier)) = runner.iterations.split_last() else {
            return Ok(());
        };
        let allocated = heap.current_relative();
        let features = build_features(
            last.into(),
            earlier.last().map(Into::into),
            allocated,
            previous_allocated,
            earlier.len(),
            term_size,
        );
        previous_allocated = Some(allocated);

        let absolute_limit = heap
            .absolute_limit()
            .expect("predictive hook requires an absolute memory limit");
        let safety_margin = predictor.safety_margin;
        if let Some(predicted) = predictive_decision(
            &mut predictor,
            &features,
            heap.baseline(),
            allocated,
            absolute_limit,
            safety_margin,
        )? {
            return Err(format!(
                "predicted next-iteration memory limit crossing \
                 ({predicted} >= {absolute_limit} bytes)"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iteration(nodes: usize, classes: usize, applied: usize) -> IterationSnapshot {
        IterationSnapshot {
            egraph_nodes: nodes,
            egraph_classes: classes,
            total_applied: applied,
            hook_time: 0.1,
            search_time: 0.2,
            apply_time: 0.3,
            rebuild_time: 0.4,
            total_time: 1.0,
            n_rebuilds: 2,
        }
    }

    struct FixedPredictor(f64);

    impl MemoryGrowthPredictor for FixedPredictor {
        fn predict_log_growth(&mut self, _features: &[f32; FEATURE_COUNT]) -> Result<f64, String> {
            Ok(self.0)
        }
    }

    fn assert_feature(actual: f32, expected: f32) {
        assert!((actual - expected).abs() <= f32::EPSILON);
    }

    #[test]
    fn feature_order_and_count_are_stable() {
        assert_eq!(FEATURE_COUNT, 16);
        assert_eq!(FEATURE_NAMES[0], "egraph_nodes");
        assert_eq!(FEATURE_NAMES[15], "term_size");
    }

    #[test]
    fn first_completed_iteration_uses_growth_defaults() {
        let features = build_features(iteration(20, 4, 7), None, 1_000, None, 0, 11);
        assert_feature(features[5], 1.0);
        assert_feature(features[6], 1.0);
    }

    #[test]
    fn subsequent_iteration_calculates_both_growth_features() {
        let features = build_features(
            iteration(30, 5, 7),
            Some(iteration(20, 4, 3)),
            1_500,
            Some(1_000),
            1,
            11,
        );
        assert_feature(features[5], 1.5);
        assert_feature(features[6], 1.5);
    }

    #[test]
    fn applied_counts_are_summed_without_rule_names() {
        assert_eq!(sum_applied_counts([&2, &5, &10].into_iter()), 17);
        let features = build_features(iteration(20, 4, 17), None, 1_000, None, 0, 11);
        assert_feature(features[7], 17.0);
    }

    #[test]
    fn zero_sizes_and_allocation_produce_only_finite_features() {
        let features = build_features(
            iteration(0, 0, 0),
            Some(iteration(0, 0, 0)),
            0,
            Some(0),
            1,
            0,
        );
        assert!(features.into_iter().all(f32::is_finite));
        assert_feature(features[2], 1.0);
        assert_feature(features[4], 1.0);
        assert_feature(features[5], 1.0);
        assert_feature(features[6], 1.0);
    }

    #[test]
    fn prediction_below_ceiling_continues() {
        let result = predictive_decision(&mut FixedPredictor(0.0), &[0.0; 16], 100, 200, 301, 0.0);
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn prediction_at_or_above_ceiling_stops() {
        let result = predictive_decision(&mut FixedPredictor(0.0), &[0.0; 16], 100, 200, 300, 0.0);
        assert_eq!(result.unwrap(), Some(300));
    }

    #[test]
    fn non_finite_and_overflowing_predictions_stop_conservatively() {
        assert_eq!(predicted_next_absolute(100, 200, f64::NAN, 0.0), u64::MAX);
        assert_eq!(predicted_next_absolute(100, 200, f64::MAX, 0.0), u64::MAX);
    }

    #[test]
    fn onnx_file_matches_the_exporters_sklearn_prediction() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/memory_growth.onnx");
        let manifest_json =
            std::fs::read_to_string(OnnxMemoryGrowthPredictor::manifest_path(&model_path)).unwrap();
        let manifest: ModelManifest = serde_json::from_str(&manifest_json).unwrap();
        let mut first = OnnxMemoryGrowthPredictor::load_and_prewarm(&model_path).unwrap();
        let mut second = OnnxMemoryGrowthPredictor::load_and_prewarm(&model_path).unwrap();
        let actual = first
            .predict_log_growth(&manifest.rust_parity_sample.features)
            .unwrap();
        let second_actual = second
            .predict_log_growth(&manifest.rust_parity_sample.features)
            .unwrap();
        assert!(
            (actual - manifest.rust_parity_sample.sklearn_prediction).abs()
                <= manifest.rust_parity_sample.absolute_tolerance
        );
        assert!(
            (second_actual - manifest.rust_parity_sample.sklearn_prediction).abs()
                <= manifest.rust_parity_sample.absolute_tolerance
        );
    }
}
