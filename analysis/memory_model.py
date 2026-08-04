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
    boosted = make_pipeline(
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=0,
        )
    )
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
