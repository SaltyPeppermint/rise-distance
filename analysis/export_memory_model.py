"""Train, export, and evaluate the scalar next-iteration memory model.

The generated ONNX file is embedded by the Rust predictive-memory hook. Run:

    uv run python analysis/export_memory_model.py

Pass ``--seed-dir`` to select a trace set instead of the newest seed-term run.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import skl2onnx
import sklearn
import tyro
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import GroupKFold, cross_val_predict

import iteration_data as data
import memory_model

FEATURES = (
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
)
TARGET = "y_log_growth"
MODEL_PARAMETERS = memory_model.BOOSTED_PARAMETERS
SAFETY_QUANTILE = 0.99
FLOAT32_PARITY_ATOL = 1e-5
DEFAULT_CEILINGS = (32 << 20, 64 << 20, 128 << 20)


@dataclass(frozen=True)
class Args:
    seed_dir: Path | None = None
    """Trace directory. Defaults to the newest seed-term run."""

    output_dir: Path = Path("models")
    """Directory for the model, manifest, and evaluation report."""

    n_jobs: int = 5
    """Number of parallel cross-validation jobs."""

    ceilings: tuple[int, ...] = DEFAULT_CEILINGS
    """Memory ceilings in bytes to use during replay evaluation."""


def _model():
    return memory_model.make_models(FEATURES, ())["gradient boosting"]


def _onnx_bytes(model) -> bytes:
    graph = convert_sklearn(
        model,
        name="rise-distance scalar memory growth",
        initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
        target_opset=18,
    )
    return graph.SerializeToString()


def _ort_predict(model_bytes: bytes, X: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    output = session.run(None, {"features": X.astype(np.float32)})[0]
    return np.asarray(output).reshape(-1)


def _validate_conversion(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Compare sklearn and ONNX on a group-held-out fold."""
    train, held_out = next(GroupKFold(n_splits=5).split(X, y, groups))
    model = _model().fit(X[train], y[train])
    sklearn_prediction = model.predict(X[held_out])
    onnx_prediction = _ort_predict(_onnx_bytes(model), X[held_out])
    error = np.abs(sklearn_prediction - onnx_prediction)
    max_error = float(error.max())
    if not np.allclose(sklearn_prediction, onnx_prediction, rtol=0.0, atol=FLOAT32_PARITY_ATOL):
        raise AssertionError(
            f"ONNX parity failed: max error {max_error:g} > {FLOAT32_PARITY_ATOL:g}"
        )
    return {
        "rows": len(held_out),
        "absolute_tolerance": FLOAT32_PARITY_ATOL,
        "max_absolute_error": max_error,
        "mean_absolute_error": float(error.mean()),
    }


def _replay(
    transitions,
    predictions: np.ndarray,
    safety_margin: float,
    ceilings: tuple[int, ...],
) -> list[dict]:
    """Evaluate one-step boundary decisions and distance to later crossings."""
    rows = []
    term = transitions["term"].to_numpy()
    iteration = transitions["iter_index"].to_numpy()
    allocated = transitions["allocated"].to_numpy().astype(np.float64)
    actual_next = transitions["next_allocated"].to_numpy().astype(np.float64)

    for label, margin in (("raw", 0.0), ("conservative", safety_margin)):
        predicted_next = allocated * np.exp(predictions + margin)
        for ceiling in ceilings:
            currently_below = allocated < ceiling
            actual_crossing = currently_below & (actual_next >= ceiling)
            predicted_stop = currently_below & (predicted_next >= ceiling)

            # Simulate the hook: only the first predicted stop in each run is
            # observable. Later decisions cannot happen after a predictive
            # stop or the hard limit's first crossing.
            starts = np.r_[0, np.flatnonzero(term[1:] != term[:-1]) + 1]
            ends = np.r_[starts[1:], len(term)]
            actual_crossings = avoided = missed = false_early = 0
            iterations_early = []
            for start, end in zip(starts, ends, strict=True):
                actual_indices = np.flatnonzero(actual_crossing[start:end])
                stop_indices = np.flatnonzero(predicted_stop[start:end])
                crossing = int(actual_indices[0]) if len(actual_indices) else None
                stop = int(stop_indices[0]) if len(stop_indices) else None
                if crossing is not None:
                    actual_crossings += 1
                    if stop is not None and stop <= crossing:
                        avoided += 1
                        iterations_early.append(
                            int(iteration[start + crossing] - iteration[start + stop])
                        )
                    else:
                        missed += 1
                elif stop is not None:
                    false_early += 1

            rows.append(
                {
                    "boundary": label,
                    "ceiling_bytes": ceiling,
                    "actual_crossings": actual_crossings,
                    "avoided_crossings": avoided,
                    "missed_crossings": missed,
                    "false_early_stops": false_early,
                    "iterations_early_mean": (
                        float(np.mean(iterations_early)) if iterations_early else None
                    ),
                    "iterations_early_max": max(iterations_early, default=None),
                }
            )
    return rows


def _measure_overhead(
    model_bytes: bytes,
    X: np.ndarray,
    predictions: np.ndarray,
    transitions,
    safety_margin: float,
    ceilings: tuple[int, ...],
) -> dict:
    """Time single-row inference and estimate hook_time feedback."""
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    sample = X[: min(len(X), 2_000)].astype(np.float32)
    for row in sample[:10]:
        session.run(None, {"features": row.reshape(1, -1)})
    started = time.perf_counter()
    for row in sample:
        session.run(None, {"features": row.reshape(1, -1)})
    seconds = time.perf_counter() - started
    per_inference = seconds / len(sample)

    perturbed = X.copy()
    perturbed[:, FEATURES.index("hook_time")] += per_inference
    perturbed_predictions = _ort_predict(model_bytes, perturbed)
    changed = {}
    allocated = transitions["allocated"].to_numpy().astype(np.float64)
    before = allocated * np.exp(predictions + safety_margin)
    after = allocated * np.exp(perturbed_predictions + safety_margin)
    for ceiling in ceilings:
        changed[str(ceiling)] = int(((before >= ceiling) != (after >= ceiling)).sum())
    return {
        "sample_rows": len(sample),
        "total_seconds": seconds,
        "mean_seconds_per_inference": per_inference,
        "prediction_mean_absolute_change": float(
            np.mean(np.abs(perturbed_predictions - predictions))
        ),
        "changed_stop_decisions_by_ceiling": changed,
    }


def main() -> None:
    args = tyro.cli(Args, description=__doc__)

    seed_dir = args.seed_dir or data.resolve_seed_dir()
    try:
        trace_name = str(seed_dir.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        trace_name = str(seed_dir.resolve())
    transitions = data.build_transitions(data.load_iterations(seed_dir))
    scalar_features, _ = data.feature_columns(transitions)
    if tuple(scalar_features) != FEATURES:
        raise AssertionError(f"feature mismatch: {scalar_features!r}")

    # Training in the deployed input precision keeps tree thresholds on the
    # same side of borderline values in sklearn and ONNX Runtime.
    X = memory_model.design_matrix(transitions, FEATURES).astype(np.float32)
    y = transitions[TARGET].to_numpy()
    groups = transitions["term"].to_numpy()
    cv = GroupKFold(n_splits=5)
    oof = cross_val_predict(_model(), X, y, cv=cv, groups=groups, n_jobs=args.n_jobs)
    # Positive residual means the model underpredicted actual log growth.
    residuals = y - oof
    safety_margin = float(np.quantile(residuals, SAFETY_QUANTILE))
    observed_miss_rate = float(np.mean(residuals > safety_margin))

    parity = _validate_conversion(X, y, groups)
    fitted = _model().fit(X, y)
    model_bytes = _onnx_bytes(fitted)
    full_onnx_predictions = _ort_predict(model_bytes, X)
    rust_parity_sample = {
        "features": X[0].tolist(),
        "sklearn_prediction": float(fitted.predict(X[:1])[0]),
        "onnx_prediction": float(full_onnx_predictions[0]),
        "absolute_tolerance": FLOAT32_PARITY_ATOL,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "memory_growth.onnx"
    manifest_path = args.output_dir / "memory_growth.json"
    report_path = args.output_dir / "memory_growth_evaluation.json"
    model_path.write_bytes(model_bytes)
    manifest = {
        "features": list(FEATURES),
        "target": TARGET,
        "model": "sklearn.ensemble.HistGradientBoostingRegressor",
        "model_parameters": MODEL_PARAMETERS,
        "safety_margin": safety_margin,
        "safety_quantile": SAFETY_QUANTILE,
        "observed_oof_miss_rate": observed_miss_rate,
        "training_rows": len(transitions),
        "training_groups": int(transitions["term"].n_unique()),
        "training_trace": trace_name,
        "sklearn_version": sklearn.__version__,
        "skl2onnx_version": skl2onnx.__version__,
        "float32_parity": parity,
        "rust_parity_sample": rust_parity_sample,
        "distribution_warning": (
            "Trained only on rise-distance seed-term eqsat traces; enable explicitly "
            "and do not assume calibration transfers to unrelated workloads."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report = {
        "safety_margin": {
            "quantile": SAFETY_QUANTILE,
            "log_growth": safety_margin,
            "growth_multiplier": float(np.exp(safety_margin)),
            "observed_oof_miss_rate": observed_miss_rate,
        },
        "parity": parity,
        "replay": _replay(
            transitions,
            oof,
            safety_margin,
            tuple(args.ceilings),
        ),
        "inference_overhead": _measure_overhead(
            model_bytes,
            X,
            full_onnx_predictions,
            transitions,
            safety_margin,
            tuple(args.ceilings),
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {model_path}, {manifest_path}, and {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
