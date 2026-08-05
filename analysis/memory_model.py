"""Models for next-iteration eqsat memory prediction."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Features transformed with log1p for the linear model.
LOG_SCALARS = frozenset(
    {
        "egraph_nodes",
        "egraph_classes",
        "allocated",
        "bytes_per_node",
        "n_rebuilds",
        "total_applied",
        "hook_time",
        "search_time",
        "apply_time",
        "rebuild_time",
        "total_time",
    }
)

BOOSTED_PARAMETERS = {
    "max_iter": 300,
    "learning_rate": 0.08,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 0,
}


def _base_column(col: str) -> str:
    """Strip a `_lag<k>` suffix."""
    if "_lag" in col:
        return col.rsplit("_lag", 1)[0]
    return col


@dataclass(frozen=True)
class Target:
    """Prediction target and baseline."""

    key: str
    label: str
    description: str

    def naive(self, df: pl.DataFrame) -> np.ndarray:
        """Carry-forward baseline in the target's own units."""
        if self.key == "y_log_next":
            return np.log(df["allocated"].to_numpy().astype(np.float64))
        return np.zeros(len(df), dtype=np.float64)


TARGETS = (
    Target(
        "y_log_next",
        "log next-iteration memory",
        "Log memory in the next iteration.",
    ),
    Target(
        "y_log_growth",
        "log memory growth ratio",
        "Log ratio of next-iteration to current memory.",
    ),
)


def design_matrix(df: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Return a dense float matrix in feature order."""
    return df.select(list(features)).to_numpy().astype(np.float64)


def make_models(features: Sequence[str], rules: Sequence[str]) -> dict[str, Pipeline]:
    """Create ridge and gradient-boosting models."""
    rule_set = set(rules)
    log_cols = [
        i
        for i, col in enumerate(features)
        if _base_column(col) in LOG_SCALARS or _base_column(col) in rule_set
    ]
    linear_cols = [i for i in range(len(features)) if i not in set(log_cols)]

    ridge = make_pipeline(
        ColumnTransformer(
            [
                ("log", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), log_cols),
                ("linear", StandardScaler(), linear_cols),
            ]
        ),
        Ridge(alpha=1.0),
    )
    boosted = make_pipeline(HistGradientBoostingRegressor(**BOOSTED_PARAMETERS))
    return {"ridge": ridge, "gradient boosting": boosted}


def evaluate(
    df: pl.DataFrame,
    features: Sequence[str],
    rules: Sequence[str],
    *,
    n_splits: int = 5,
    n_jobs: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cross-validate each model and target, grouped by seed term."""
    X = design_matrix(df, features)
    groups = df["term"].to_numpy()
    cv = GroupKFold(n_splits=n_splits)

    metric_rows, prediction_frames = [], []
    for target in TARGETS:
        y = df[target.key].to_numpy()
        scored = {"naive (carry forward)": target.naive(df)}
        for name, model in make_models(features, rules).items():
            scored[name] = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=n_jobs)

        for name, pred in scored.items():
            metric_rows.append(
                {
                    "target": target.label,
                    "model": name,
                    "MAE (log)": mean_absolute_error(y, pred),
                    "RMSE (log)": float(np.sqrt(np.mean((y - pred) ** 2))),
                    "R2": r2_score(y, pred),
                    "median error x": float(np.exp(np.median(np.abs(y - pred)))),
                }
            )
            prediction_frames.append(
                pl.DataFrame(
                    {
                        "target": [target.label] * len(y),
                        "model": [name] * len(y),
                        "actual": y,
                        "predicted": pred,
                        "residual": pred - y,
                        "egraph_nodes": df["egraph_nodes"],
                        "iter_index": df["iter_index"],
                        "term_size": df["term_size"],
                    }
                )
            )

    metrics = pl.DataFrame(metric_rows)
    return metrics, pl.concat(prediction_frames, how="vertical")


def crossing_labels(df: pl.DataFrame, ceiling: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows sitting below `ceiling` and, of those, which ones cross it next.

    A run can only be stopped from below, so rows already at or above the
    ceiling are outside the decision problem entirely and are excluded rather
    than counted as easy negatives.
    """
    allocated = df["allocated"].to_numpy().astype(np.float64)
    next_allocated = df["next_allocated"].to_numpy().astype(np.float64)
    below = allocated < ceiling
    return below, below & (next_allocated >= ceiling)


def ceiling_decisions(
    df: pl.DataFrame,
    predictions: np.ndarray,
    ceilings: Sequence[float],
    *,
    margins: Sequence[tuple[str, float]] = (("raw", 0.0),),
) -> pl.DataFrame:
    """Score log-growth predictions as ceiling-crossing stop decisions.

    Replays the Rust hook's rule -- stop when
    `allocated * exp(prediction + margin) >= ceiling` -- over each run in
    iteration order. Only the *first* predicted stop in a run is observable,
    because the hook halts the run there; later rows describe a future that
    would never have happened. Crossings are counted per run, not per row, for
    the same reason.

    `predictions` must be held-out (grouped-CV) log-growth predictions aligned
    with `df`'s row order. Returns one row per (margin, ceiling) with the
    quantities that matter operationally: how many runs were caught before
    crossing, how many were missed, how many were stopped that never would have
    crossed, and how much warning the catches gave.
    """
    if len(predictions) != len(df):
        raise ValueError(f"predictions/rows mismatch: {len(predictions)} vs {len(df)}")

    ordered = df.with_columns(pl.Series("_prediction", predictions)).sort("term", "iter_index")
    term = ordered["term"].to_numpy()
    iteration = ordered["iter_index"].to_numpy()
    allocated = ordered["allocated"].to_numpy().astype(np.float64)
    prediction = ordered["_prediction"].to_numpy()

    # Run boundaries in the sorted frame.
    starts = np.r_[0, np.flatnonzero(term[1:] != term[:-1]) + 1]
    ends = np.r_[starts[1:], len(term)]

    rows = []
    for label, margin in margins:
        predicted_next = allocated * np.exp(prediction + margin)
        for ceiling in ceilings:
            below, actual_crossing = crossing_labels(ordered, ceiling)
            predicted_stop = below & (predicted_next >= ceiling)

            crossings = caught = missed = false_stops = 0
            warning = []
            for start, end in zip(starts, ends, strict=True):
                crossing_at = np.flatnonzero(actual_crossing[start:end])
                stop_at = np.flatnonzero(predicted_stop[start:end])
                crossing = int(crossing_at[0]) if len(crossing_at) else None
                stop = int(stop_at[0]) if len(stop_at) else None
                if crossing is not None:
                    crossings += 1
                    if stop is not None and stop <= crossing:
                        caught += 1
                        warning.append(int(iteration[start + crossing] - iteration[start + stop]))
                    else:
                        missed += 1
                elif stop is not None:
                    false_stops += 1

            runs_at_risk = crossings + false_stops
            rows.append(
                {
                    "boundary": label,
                    "ceiling_mib": ceiling / 2**20,
                    "runs": len(starts),
                    "crossing_runs": crossings,
                    "caught": caught,
                    "missed": missed,
                    "false_stops": false_stops,
                    "recall": caught / crossings if crossings else None,
                    # Of every run we stopped, the share that really would have
                    # crossed. This is the cost side: a false stop throws away
                    # work that would have completed fine.
                    "precision": caught / runs_at_risk if runs_at_risk else None,
                    "iters_warning_mean": float(np.mean(warning)) if warning else None,
                    "iters_warning_max": max(warning, default=None),
                }
            )
    return pl.DataFrame(rows)


def ceiling_sweep(
    df: pl.DataFrame,
    features: Sequence[str],
    rules: Sequence[str],
    ceilings: Sequence[float],
    *,
    safety_quantile: float = 0.99,
    n_splits: int = 5,
    n_jobs: int = 5,
) -> tuple[pl.DataFrame, float]:
    """Cross-validate the boosted model, then score its ceiling decisions.

    Fits on rows with a trustworthy growth target (`next_is_stop_iter` false,
    when that column is present) so post-run heap totals never enter training,
    but *scores* decisions on every row including the crossings themselves.

    Returns the per-ceiling decision table and the safety margin, the
    `safety_quantile` of held-out residuals. The margin is what buys recall:
    the model underpredicts growth on some runs, and shifting predictions up by
    this much converts most of those near-misses into catches.
    """
    trainable = (
        df.filter(pl.col("next_is_stop_iter").not_()) if "next_is_stop_iter" in df.columns else df
    )
    X_train = design_matrix(trainable, features)
    y_train = trainable["y_log_growth"].to_numpy()
    cv = GroupKFold(n_splits=n_splits)
    oof = cross_val_predict(
        make_models(features, rules)["gradient boosting"],
        X_train,
        y_train,
        cv=cv,
        groups=trainable["term"].to_numpy(),
        n_jobs=n_jobs,
    )
    # Positive residual means the model underpredicted actual log growth.
    safety_margin = float(np.quantile(y_train - oof, safety_quantile))

    # Score every row, including the stop transitions held out of training.
    fitted = make_models(features, rules)["gradient boosting"].fit(X_train, y_train)
    scored = fitted.predict(design_matrix(df, features))

    decisions = ceiling_decisions(
        df,
        scored,
        ceilings,
        margins=(("raw", 0.0), ("conservative", safety_margin)),
    )
    return decisions, safety_margin


def rule_feature_ablation(
    df: pl.DataFrame,
    scalar_features: Sequence[str],
    rules: Sequence[str],
    *,
    n_splits: int = 5,
    n_jobs: int = 5,
) -> pl.DataFrame:
    """Compare boosted models with and without per-rule application counts.

    Both variants use the same grouped folds and hyperparameters. This isolates
    the value of the rule-count block from model and train/test-split effects.
    """
    groups = df["term"].to_numpy()
    cv = GroupKFold(n_splits=n_splits)
    variants = {
        "scalars only": list(scalar_features),
        "scalars + rules": [*scalar_features, *rules],
    }

    rows = []
    for target in TARGETS:
        y = df[target.key].to_numpy()
        for feature_set, features in variants.items():
            X = design_matrix(df, features)
            model = make_models(features, rules if feature_set == "scalars + rules" else ())[
                "gradient boosting"
            ]
            pred = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=n_jobs)
            rows.append(
                {
                    "target": target.label,
                    "feature set": feature_set,
                    "n_features": len(features),
                    "MAE (log)": mean_absolute_error(y, pred),
                    "RMSE (log)": float(np.sqrt(np.mean((y - pred) ** 2))),
                    "R2": r2_score(y, pred),
                    "median error x": float(np.exp(np.median(np.abs(y - pred)))),
                }
            )

    return pl.DataFrame(rows)


def importances(
    df: pl.DataFrame,
    features: Sequence[str],
    rules: Sequence[str],
    target: str = "y_log_growth",
    *,
    n_repeats: int = 5,
    top: int = 20,
) -> pl.DataFrame:
    """Compute permutation importance on one held-out group fold."""
    X = design_matrix(df, features)
    y = df[target].to_numpy()
    groups = df["term"].to_numpy()

    train_idx, test_idx = next(GroupKFold(n_splits=5).split(X, y, groups))
    model = make_models(features, rules)["gradient boosting"]
    model.fit(X[train_idx], y[train_idx])

    result = permutation_importance(
        model, X[test_idx], y[test_idx], n_repeats=n_repeats, random_state=0, n_jobs=5
    )
    return (
        pl.DataFrame(
            {
                "feature": list(features),
                "importance": result.importances_mean,
                "std": result.importances_std,
                "block": ["rule" if col in set(rules) else "scalar" for col in features],
            }
        )
        .sort("importance", descending=True)
        .head(top)
    )


def window_sweep(
    iterations: pl.DataFrame,
    windows: Sequence[int] = (1, 2, 3, 4, 6, 8),
    *,
    n_splits: int = 5,
    n_jobs: int = 5,
) -> pl.DataFrame:
    """Score the models at several history depths."""
    import iteration_data as data

    rows = []
    for window in windows:
        transitions = data.build_transitions(iterations, window=window)
        scalars, rules = data.feature_columns(transitions)
        features = scalars + data.window_columns(transitions) + rules

        metrics, _ = evaluate(transitions, features, rules, n_splits=n_splits, n_jobs=n_jobs)
        rows.append(
            metrics.with_columns(
                pl.lit(window).alias("window"),
                pl.lit(len(features)).alias("n_features"),
                pl.lit(len(transitions)).alias("n_rows"),
            )
        )
    return pl.concat(rows, how="vertical")


def size_extrapolation(
    df: pl.DataFrame,
    features: Sequence[str],
    rules: Sequence[str],
    *,
    split_size: int = 36,
) -> pl.DataFrame:
    """Train on terms below `split_size` and test on the remaining terms."""
    X = design_matrix(df, features)
    is_large = (df["term_size"] >= split_size).to_numpy()
    if not is_large.any() or is_large.all():
        lo, hi = df["term_size"].min(), df["term_size"].max()
        raise ValueError(
            f"split_size={split_size} must divide the observed term sizes [{lo}, {hi}]"
        )

    rows = []
    for target in TARGETS:
        y = df[target.key].to_numpy()
        naive = target.naive(df)
        rows.append(
            {
                "target": target.label,
                "model": "naive (carry forward)",
                "MAE (log)": mean_absolute_error(y[is_large], naive[is_large]),
                "R2": r2_score(y[is_large], naive[is_large]),
            }
        )
        for name, model in make_models(features, rules).items():
            model.fit(X[~is_large], y[~is_large])
            pred = model.predict(X[is_large])
            rows.append(
                {
                    "target": target.label,
                    "model": name,
                    "MAE (log)": mean_absolute_error(y[is_large], pred),
                    "R2": r2_score(y[is_large], pred),
                }
            )
    return pl.DataFrame(rows)
