"""Train/export the scheduler-aware upcoming-iteration peak-memory model.

Run ``uv run python analysis/export_memory_model.py --seed-dir TRACE`` after
regenerating traces with iteration-local peaks and per-rule scheduler state.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
import skl2onnx
import sklearn
import tyro
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

import iteration_data as data
import memory_model

TARGET = "y_log_peak_growth"
MODEL_PARAMETERS = memory_model.BOOSTED_PARAMETERS
SAFETY_QUANTILE = 0.99
FLOAT32_PARITY_ATOL = 1e-5
DEFAULT_CEILINGS = (64 << 20, 128 << 20, 256 << 20, 500 << 20, 1000 << 20, 2000 << 20)


@dataclass(frozen=True)
class Args:
    seed_dir: Path | None = None
    """Scheduler/peak-aware trace directory. Defaults to the newest trace."""

    output_dir: Path = Path("models")
    """Directory for model, manifest, and evaluation report."""

    n_jobs: int = 5
    """Parallelism hint for analysis (fold fitting is deterministic)."""

    ceilings: tuple[int, ...] = DEFAULT_CEILINGS
    """Absolute process-memory ceilings used for first-stop replay."""


def _model(features):
    return memory_model.make_models(features, ())["gradient boosting"]


def _onnx_bytes(model, features) -> bytes:
    graph = convert_sklearn(
        model,
        name="rise-distance scheduler-aware peak memory growth",
        initial_types=[("features", FloatTensorType([None, len(features)]))],
        target_opset=18,
    )
    return graph.SerializeToString()


def _ort_predict(model_bytes: bytes, X: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    output = session.run(None, {"features": X.astype(np.float32)})[0]
    return np.asarray(output).reshape(-1)


def _validate_conversion(X, y, groups, features) -> dict:
    train, held_out = next(GroupKFold(n_splits=5).split(X, y, groups))
    model = _model(features).fit(X[train], y[train])
    sklearn_prediction = model.predict(X[held_out])
    onnx_prediction = _ort_predict(_onnx_bytes(model, features), X[held_out])
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


def _measure_overhead(model_bytes, X, predictions, decisions, features, margin, ceilings):
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
    perturbed[:, features.index("hook_time")] += per_inference
    changed_predictions = _ort_predict(model_bytes, perturbed)
    allocated = decisions.filter(pl.col("target_trainable"))["allocated"].to_numpy()
    before = allocated * np.exp(predictions + margin)
    after = allocated * np.exp(changed_predictions + margin)
    return {
        "sample_rows": len(sample),
        "total_seconds": seconds,
        "mean_seconds_per_inference": per_inference,
        "prediction_mean_absolute_change": float(
            np.mean(np.abs(changed_predictions - predictions))
        ),
        "changed_stop_decisions_by_ceiling": {
            str(ceiling): int(((before >= ceiling) != (after >= ceiling)).sum())
            for ceiling in ceilings
        },
    }


def _classifier_comparison(decisions, features, ceilings, regression_replay):
    """Test a separate crossing classifier and retain it only on clear wins."""
    X = memory_model.design_matrix(decisions, features).astype(np.float32)
    groups = decisions["term"].to_numpy()
    peaks = decisions["iteration_peak_allocated"].to_numpy()
    allocated = decisions["allocated"].to_numpy().astype(np.float64)
    rows = []
    for ceiling in ceilings:
        eligible = allocated < ceiling
        labels = peaks >= ceiling
        if int((eligible & labels).sum()) < 5:
            rows.append(
                {
                    "ceiling_mib": ceiling / 2**20,
                    "evaluated": False,
                    "reason": "fewer than five crossing decision rows",
                }
            )
            continue
        probabilities = np.zeros(len(decisions))
        valid = True
        for train, test in GroupKFold(n_splits=5).split(X, labels, groups):
            fit = train[eligible[train]]
            if np.unique(labels[fit]).size < 2:
                valid = False
                break
            classifier = HistGradientBoostingClassifier(
                **{
                    key: value
                    for key, value in MODEL_PARAMETERS.items()
                    if key != "l2_regularization"
                },
                l2_regularization=MODEL_PARAMETERS["l2_regularization"],
            ).fit(X[fit], labels[fit])
            probabilities[test] = classifier.predict_proba(X[test])[:, 1]
        if not valid:
            rows.append(
                {
                    "ceiling_mib": ceiling / 2**20,
                    "evaluated": False,
                    "reason": "a grouped training fold had no crossing examples",
                }
            )
            continue
        stop = probabilities >= 0.5
        artificial_log_growth = np.where(
            stop,
            np.log(np.maximum(ceiling / allocated, 1.0)),
            -100.0,
        )
        score = memory_model.ceiling_decisions(
            decisions,
            artificial_log_growth,
            (ceiling,),
        ).to_dicts()[0]
        reg = next(
            row
            for row in regression_replay
            if row["boundary"] == "conservative" and row["ceiling_mib"] == ceiling / 2**20
        )
        improved = score["caught"] > reg["caught"] and score["false_stops"] <= reg["false_stops"]
        rows.append(
            {
                "ceiling_mib": ceiling / 2**20,
                "evaluated": True,
                "threshold": 0.5,
                "classifier_caught": score["caught"],
                "classifier_missed": score["missed"],
                "classifier_false_stops": score["false_stops"],
                "regressor_caught": reg["caught"],
                "regressor_missed": reg["missed"],
                "regressor_false_stops": reg["false_stops"],
                "materially_improved": improved,
            }
        )
    retained = bool(rows) and all(row.get("materially_improved", False) for row in rows)
    return {
        "retained": retained,
        "reason": "retained only on consistent first-stop wins",
        "rows": rows,
    }


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    seed_dir = args.seed_dir or data.resolve_seed_dir()
    try:
        trace_name = str(seed_dir.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        trace_name = str(seed_dir.resolve())

    iterations = data.load_iterations(seed_dir)
    decisions = data.build_decision_rows(iterations)
    rules = data.rules_from_frame(decisions)
    features = data.feature_schema(rules)
    scalar_features, per_rule_features = data.feature_columns(decisions)
    if (*scalar_features, *per_rule_features) != features:
        raise AssertionError("Python feature blocks disagree with deterministic schema")
    if decisions["scheduler_kind"].unique().to_list() != ["backoff"]:
        raise AssertionError("deployed model requires BackoffScheduler traces")

    trainable = decisions.filter(pl.col("target_trainable"))
    X = memory_model.design_matrix(trainable, features).astype(np.float32)
    y = trainable[TARGET].to_numpy()
    groups = trainable["term"].to_numpy()

    oof_all, trainable_mask = memory_model.grouped_oof_predictions(decisions, features)
    actual_all = decisions[TARGET].to_numpy()
    residuals = actual_all[trainable_mask] - oof_all[trainable_mask]
    safety_margin = float(np.quantile(residuals, SAFETY_QUANTILE))
    observed_miss_rate = float(np.mean(residuals > safety_margin))

    parity = _validate_conversion(X, y, groups, features)
    fitted = _model(features).fit(X, y)
    model_bytes = _onnx_bytes(fitted, features)
    full_onnx_predictions = _ort_predict(model_bytes, X)
    rust_parity_sample = {
        "features": X[0].tolist(),
        "sklearn_prediction": float(fitted.predict(X[:1])[0]),
        "onnx_prediction": float(full_onnx_predictions[0]),
        "absolute_tolerance": FLOAT32_PARITY_ATOL,
    }

    replay = memory_model.ceiling_decisions(
        decisions,
        oof_all,
        args.ceilings,
        margins=(("raw", 0.0), ("conservative", safety_margin)),
    ).to_dicts()
    rule_replay = memory_model.crossing_by_responsible_rule(
        decisions, oof_all, args.ceilings, safety_margin
    ).to_dicts()
    ablation_regression, ablation_replay = memory_model.scheduler_feature_ablation(
        decisions,
        rules,
        args.ceilings,
        safety_quantile=SAFETY_QUANTILE,
        n_jobs=args.n_jobs,
    )
    classifier = _classifier_comparison(decisions, features, args.ceilings, replay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "memory_growth.onnx"
    manifest_path = args.output_dir / "memory_growth.json"
    report_path = args.output_dir / "memory_growth_evaluation.json"
    model_path.write_bytes(model_bytes)
    manifest = {
        "schema_version": 1,
        "features": list(features),
        "scheduler": "backoff",
        "rules": rules,
        "target": TARGET,
        "model": "sklearn.ensemble.HistGradientBoostingRegressor",
        "model_parameters": MODEL_PARAMETERS,
        "safety_margin": safety_margin,
        "safety_quantile": SAFETY_QUANTILE,
        "observed_oof_miss_rate": observed_miss_rate,
        "training_rows": len(trainable),
        "decision_rows": len(decisions),
        "training_groups": int(trainable["term"].n_unique()),
        "training_trace": trace_name,
        "sklearn_version": sklearn.__version__,
        "skl2onnx_version": skl2onnx.__version__,
        "onnx_input_dtype": "float32",
        "float32_parity": parity,
        "rust_parity_sample": rust_parity_sample,
        "scope": "single-term generation/guide replay; verification remains disabled",
        "distribution_warning": (
            "Trained only on rise-distance seed-term eqsat traces; a different or "
            "incomplete scheduler rule set is a compatibility error."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    crossing_rows = {
        str(ceiling): int(
            (
                (decisions["allocated"] < ceiling)
                & (decisions["iteration_peak_allocated"] >= ceiling)
            ).sum()
        )
        for ceiling in args.ceilings
    }
    report = {
        "training_trace": trace_name,
        "decision_rows": len(decisions),
        "training_rows": len(trainable),
        "crossing_rows_by_ceiling": crossing_rows,
        "safety_margin": {
            "quantile": SAFETY_QUANTILE,
            "log_peak_growth": safety_margin,
            "growth_multiplier": float(np.exp(safety_margin)),
            "observed_oof_miss_rate": observed_miss_rate,
        },
        "parity": parity,
        "replay": replay,
        "responsible_rule_replay": rule_replay,
        "remaining_missed_catastrophic_rules": sorted(
            {
                row["responsible_rule"]
                for row in rule_replay
                if row["missed_runs"] > 0 and row["responsible_rule"] is not None
            }
        ),
        "scheduler_feature_ablation": {
            "regression": ablation_regression.to_dicts(),
            "replay": ablation_replay.to_dicts(),
        },
        "crossing_classifier_comparison": classifier,
        "inference_overhead": _measure_overhead(
            model_bytes,
            X,
            full_onnx_predictions,
            trainable,
            list(features),
            safety_margin,
            args.ceilings,
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {model_path}, {manifest_path}, and {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
