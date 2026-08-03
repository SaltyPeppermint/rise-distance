"""Classical-ML baselines for next-iteration eqsat memory prediction.

Fits on the transition frame from `iteration_data`: features are one eqsat
iteration's state, targets are the next iteration's live-heap use. Everything
is evaluated with a term-grouped CV split, since iterations inside one run are
strongly autocorrelated and a random row split would leak neighbours across
the fold boundary.
"""

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

# Heavy-tailed features that a linear model should see in log space. Rule
# counts get the same treatment; `log1p` keeps their many zeros finite.
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
    """Strip a `_lag<k>` suffix, so a lagged column scales like its original.

    Log-space delta columns have no meaningful base and are left alone; they
    are already centred near zero and belong in the linear block.
    """
    if "_lag" in col:
        return col.rsplit("_lag", 1)[0]
    return col


@dataclass(frozen=True)
class Target:
    """One prediction task and the naive baseline it must beat."""

    key: str
    label: str
    description: str

    def naive(self, df: pl.DataFrame) -> np.ndarray:
        """Carry-forward baseline in the target's own units."""
        if self.key == "y_log_next":
            # Assume memory does not change: log(next) = log(current).
            return np.log(df["allocated"].to_numpy().astype(np.float64))
        # Assume no growth: log(next/current) = 0.
        return np.zeros(len(df), dtype=np.float64)


TARGETS = (
    Target(
        "y_log_next",
        "log next-iteration memory",
        "Deployable target. Memory is strongly autocorrelated, so the "
        "carry-forward baseline is already close; treat its score as the bar.",
    ),
    Target(
        "y_log_growth",
        "log memory growth ratio",
        "Signal test. The carry-forward baseline scores ~0 by construction, "
        "so any gain here is genuine predictive content.",
    ),
)


def design_matrix(df: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Dense float matrix in `features` order (sklearn takes numpy directly)."""
    return df.select(list(features)).to_numpy().astype(np.float64)


def make_models(features: Sequence[str], rules: Sequence[str]) -> dict[str, Pipeline]:
    """Ridge on log-scaled features, plus a gradient-boosted tree ensemble.

    Trees are scale- and monotone-invariant, so the boosted model takes the
    raw columns; only the linear model needs the log/standardise treatment.
    """
    rule_set = set(rules)
    # A lagged copy is the same quantity one step back, so it needs the same
    # log treatment as its base column. Deltas are already in log space.
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
    """Cross-validate every model on every target, grouped by seed term.

    Returns a tidy metrics frame and a long frame of out-of-fold predictions
    for the plots.
    """
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
                    # Back out of log space: typical multiplicative error.
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


def importances(
    df: pl.DataFrame,
    features: Sequence[str],
    rules: Sequence[str],
    target: str = "y_log_growth",
    *,
    n_repeats: int = 5,
    top: int = 20,
) -> pl.DataFrame:
    """Permutation importance for the boosted model on held-out terms.

    Fit on four folds and permute on the fifth, so importance reflects
    generalisation to unseen terms rather than in-sample fit.
    """
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
    """Score the models at several history depths.

    Takes the raw iteration frame rather than a transition frame, since each
    window size needs its lags rebuilt before the row filters. Backfilled
    warm-up rows keep the row set identical across `windows`, so the scores are
    directly comparable and the naive baseline is a fixed reference line.
    """
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
    """Train on small seed terms and test on large ones.

    A harder question than grouped CV: does a model fitted on cheap runs still
    predict the expensive ones you actually care about?
    """
    X = design_matrix(df, features)
    is_large = (df["term_size"] >= split_size).to_numpy()

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
