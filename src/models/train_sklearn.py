"""
train_sklearn.py: training entrypoint for scikit-learn models.

Invoked by `make train` or directly:

    python -m src.models.train_sklearn --model all --split all
    python -m src.models.train_sklearn --model rf --split cross_z_ini

For each (model, split) combination:
  1. Load features and labels from gold tables.
  2. Run GridSearchCV with PredefinedSplit (explicit val set, in-distribution).
  3. Refit the best pipeline on the combined train+val set.
  4. Evaluate on the test set.
  5. Log parameters, metrics, and the fitted pipeline to MLflow.
  6. Save the fitted pipeline to models/.

If all three base models (lr, rf, gbm) are trained in the same invocation,
a soft-voting ensemble is also trained and evaluated for each split.

GridSearchCV scoring: average_precision (PR-AUC). Chosen over accuracy and
F1 because it summarises the precision-recall tradeoff across all thresholds,
which is most informative for imbalanced classification (~81% spurious).
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import cast

import duckdb
import joblib
import mlflow
import numpy as np
from mlflow.sklearn import log_model as mlflow_sklearn_log_model
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline

from src.config import AppConfig, configure_logging, load_config
from src.db import get_connection
from src.models.data import SplitArrays, load_split_data
from src.models.evaluate import compute_metrics
from src.models.sklearn_models import MODEL_REGISTRY, ModelSpec, voting_ensemble

configure_logging()
log = logging.getLogger(__name__)

# Suppress sklearn parallel warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

_MODELS_DIR = Path("models")
_ALL_SPLITS = ["within_sim", "cross_softening", "cross_z_ini"]
_BASE_MODEL_NAMES = ["lr", "rf", "gbm"]


class SplitData(SplitArrays):
    """SplitArrays extended with the PredefinedSplit fold array for GridSearchCV."""

    test_fold: np.ndarray


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def train_sklearn(
    cfg: AppConfig,
    model_names: list[str],
    split_names: list[str],
) -> None:
    """
    Train and evaluate scikit-learn models for the given models and splits.

    Parameters
    ----------
    cfg:
        Loaded application configuration.
    model_names:
        List of model names to train (e.g. ['lr', 'rf', 'gbm']).
    split_names:
        List of split names to train on (e.g. ['within_sim', 'cross_z_ini']).
    """
    _MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    _ = mlflow.set_experiment(cfg.mlflow.experiment)

    conn = get_connection(cfg.database_path)
    try:
        for split_name in split_names:
            log.info("=== Split: %s ===", split_name)
            data = _load_split_data(conn, split_name)
            log.info(
                "  train=%d  val=%d  test=%d",
                len(data["y_train"]),
                len(data["y_val"]),
                len(data["y_test"]),
            )

            # Train base models, collecting best pipelines for the ensemble.
            trained_base: dict[str, tuple[Pipeline, dict[str, object]]] = {}
            for model_name in model_names:
                spec = MODEL_REGISTRY[model_name]()
                pipeline, best_params = _train_single_model(spec, data, split_name)
                trained_base[model_name] = (pipeline, best_params)

            # Build ensemble if all three base models were trained this run.
            if all(name in trained_base for name in _BASE_MODEL_NAMES):
                _train_ensemble(trained_base, data, split_name)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data loading (private)
# ---------------------------------------------------------------------------


def _load_split_data(conn: duckdb.DuckDBPyConnection, split_name: str) -> SplitData:
    """Load split arrays and add the PredefinedSplit fold array for GridSearchCV."""
    base = load_split_data(conn, split_name)
    # PredefinedSplit fold array: -1 = train, 0 = val.
    test_fold = np.concatenate(
        [
            np.full(len(base["y_train"]), -1),
            np.zeros(len(base["y_val"]), dtype=int),
        ]
    )
    return {**base, "test_fold": test_fold}


# ---------------------------------------------------------------------------
# Training helpers (private)
# ---------------------------------------------------------------------------


def _worker_init() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn.utils.parallel",
    )


def _train_single_model(
    spec: ModelSpec,
    data: SplitData,
    split_name: str,
) -> tuple[Pipeline, dict[str, object]]:
    """Run GridSearchCV, refit on train+val, evaluate on test, log to MLflow."""
    run_name = f"{spec.name}_{split_name}"
    log.info("Training %s ...", run_name)

    ps = PredefinedSplit(data["test_fold"])
    grid_search = GridSearchCV(
        estimator=spec.pipeline,
        param_grid=spec.param_grid,
        scoring="average_precision",
        cv=ps,
        refit=False,  # we refit manually on train+val below
        verbose=1,
    )
    with joblib.parallel_config(backend="loky", initializer=_worker_init):
        _ = grid_search.fit(data["X_trainval"], data["y_trainval"])

    best_params: dict[str, object] = grid_search.best_params_
    log.info("  Best params: %s", best_params)
    log.info("  Best val average_precision: %.4f", grid_search.best_score_)

    # Refit on full train+val with best hyperparameters.
    best_pipeline = cast(Pipeline, clone(spec.pipeline)).set_params(**best_params)
    _ = best_pipeline.fit(data["X_trainval"], data["y_trainval"])

    y_pred_proba = best_pipeline.predict_proba(data["X_test"])[:, 1]
    metrics = compute_metrics(data["y_test"], y_pred_proba)

    _log_mlflow_run(
        run_name=run_name,
        params={**best_params, "split_name": split_name, "model": spec.name},
        metrics=metrics,
        pipeline=best_pipeline,
    )

    artefact_path = _MODELS_DIR / f"{run_name}.joblib"
    _ = joblib.dump(best_pipeline, artefact_path)
    log.info("  Saved to %s", artefact_path)

    return best_pipeline, best_params


def _train_ensemble(
    trained_base: dict[str, tuple[Pipeline, dict[str, object]]],
    data: SplitData,
    split_name: str,
) -> None:
    """Assemble a soft-voting ensemble from tuned base pipelines and evaluate."""
    run_name = f"ensemble_{split_name}"
    log.info("Training %s ...", run_name)

    # Clone each base pipeline and set its best params so VotingClassifier
    # fits them with tuned hyperparameters (clone discards fit state).
    base_pipelines = [
        (name, cast(Pipeline, clone(pipeline)).set_params(**best_params))
        for name, (pipeline, best_params) in trained_base.items()
    ]

    ensemble_spec = voting_ensemble(base_pipelines)
    _ = ensemble_spec.pipeline.fit(data["X_trainval"], data["y_trainval"])

    y_pred_proba = ensemble_spec.pipeline.predict_proba(data["X_test"])[:, 1]
    metrics = compute_metrics(data["y_test"], y_pred_proba)

    _log_mlflow_run(
        run_name=run_name,
        params={"split_name": split_name, "model": "ensemble", "voting": "soft"},
        metrics=metrics,
        pipeline=ensemble_spec.pipeline,
    )

    artefact_path = _MODELS_DIR / f"{run_name}.joblib"
    _ = joblib.dump(ensemble_spec.pipeline, artefact_path)
    log.info("  Saved to %s", artefact_path)


# ---------------------------------------------------------------------------
# Evaluation and logging
# ---------------------------------------------------------------------------


def _log_mlflow_run(
    run_name: str,
    params: dict[str, object],
    metrics: dict[str, float],
    pipeline: Pipeline,
) -> None:
    with mlflow.start_run(run_name=run_name):
        _ = mlflow.log_params(params)
        _ = mlflow.log_metrics(metrics)
        _ = mlflow_sklearn_log_model(pipeline, name="model")

    log.info(
        "  [%s] avg_precision=%.4f  roc_auc=%.4f  f1=%.4f",
        run_name,
        metrics["test_average_precision"],
        metrics["test_roc_auc"],
        metrics["test_f1"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train scikit-learn models for the spurious halo classifier."
    )
    _ = parser.add_argument(
        "--model",
        choices=[*_BASE_MODEL_NAMES, "all"],
        default="all",
        help="Model to train. 'all' trains lr, rf, and gbm (and the ensemble).",
    )
    _ = parser.add_argument(
        "--split",
        choices=[*_ALL_SPLITS, "all"],
        default="all",
        help="Split to train on. 'all' trains on all three splits.",
    )
    args = parser.parse_args()

    model_arg = cast(str, args.model)
    split_arg = cast(str, args.split)
    model_names = _BASE_MODEL_NAMES if model_arg == "all" else [model_arg]
    split_names = _ALL_SPLITS if split_arg == "all" else [split_arg]

    cfg = load_config()
    log.info("Database: %s", cfg.database_path)
    log.info("Models: %s", model_names)
    log.info("Splits: %s", split_names)

    train_sklearn(cfg, model_names, split_names)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
