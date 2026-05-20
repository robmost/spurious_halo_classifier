"""
sklearn_models.py: model definitions for the spurious halo classifier.

Each factory function returns a ModelSpec containing a configured sklearn
Pipeline and its hyperparameter grid for GridSearchCV. No training logic
lives here.

Three models are defined and registered:

  - lr:    Logistic regression. Interpretable baseline. Requires imputation
           and scaling because protohalo features are nullable and LR is
           sensitive to feature scale.

  - rf:    Random Forest. Fast to tune, provides Gini feature importances.
           Requires imputation; tree-based models are scale-invariant.

  - gbm:    HistGradientBoostingClassifier. Best expected performance on tabular
           data. Handles NaN natively, so no imputer is needed.

A fourth model is available but not in the registry:

  - ensemble:    Soft-voting ensemble of lr, rf, and gbm. Assembled in
                 train_sklearn.py after GridSearchCV tunes the base models.
                 Not in MODEL_REGISTRY because it requires the tuned base
                 pipelines as arguments rather than being self-contained.

Preprocessing is bundled into each Pipeline so that imputation and scaling
are never fit on val or test data — GridSearchCV refits the full pipeline
on the training fold only.

Nullable features (protohalo): sphericity_s, log10_m_hmm, a_hmm.
All other features are guaranteed non-null by the gold pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Random state for all models. Controls internal model randomness (tree
# building, weight initialisation).
# NOTE: this is separate from the data split seed, which lives in config.yaml.
_RANDOM_STATE: int = 55667788


@dataclass
class ModelSpec:
    """
    Container for a model name, its pipeline, and its hyperparameter grid.

    Attributes
    ----------
    name:
        Short identifier used in MLflow run names and saved artefact filenames.
    pipeline:
        Fully configured sklearn Pipeline. Preprocessing (imputation, scaling)
        is included where required by the estimator.
    param_grid:
        Hyperparameter grid for GridSearchCV. Keys must match pipeline step
        names using the double-underscore convention (e.g. 'classifier__C').
    """

    name: str
    pipeline: Pipeline
    param_grid: Mapping[str, Sequence[object]]


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------


def logistic_regression() -> ModelSpec:
    """
    Logistic regression pipeline with imputation and standard scaling.

    Uses the saga solver, which supports the full range of l1_ratio values.
    l1_ratio=0.0 gives pure L2 regularisation; l1_ratio=1.0 gives pure L1;
    values between give elastic net, combining sparsity and shrinkage.

    Imputation uses the column mean, which is appropriate for protohalo features
    whose missingness is not random (halos with no protohalo record are
    structurally different), so mean imputation is a deliberate simplification
    that keeps the model comparable to RF and GBM. SHAP analysis will reveal
    whether imputed features drive predictions.

    Returns
    -------
    ModelSpec
        name='lr', pipeline with SimpleImputer -> StandardScaler -> LogisticRegression.
    """
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="saga",
                    max_iter=2000,
                    random_state=_RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__l1_ratio": [0.0, 0.1, 0.5, 0.9, 1.0],
        "classifier__class_weight": [None, "balanced"],
    }

    return ModelSpec(name="lr", pipeline=pipeline, param_grid=param_grid)


def random_forest() -> ModelSpec:
    """
    Random Forest pipeline with mean imputation.

    No scaling needed, as tree-based models are scale-invariant. Imputation
    is required because RandomForestClassifier does not handle NaN natively.
    Cost-complexity pruning (ccp_alpha) is included in the grid to prevent
    individual trees from overfitting, particularly on the minority class.

    Returns
    -------
    ModelSpec
        name='rf', pipeline with SimpleImputer -> RandomForestClassifier.
    """
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "classifier",
                RandomForestClassifier(
                    n_jobs=-1,
                    random_state=_RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [200, 500, 1000],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 5, 20],
        "classifier__max_features": ["sqrt", 0.5],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"],
        "classifier__ccp_alpha": [0.0, 0.001, 0.01],
    }

    return ModelSpec(name="rf", pipeline=pipeline, param_grid=param_grid)


def gradient_boosted_trees() -> ModelSpec:
    """
    HistGradientBoostingClassifier pipeline.

    Handles NaN natively, so no imputer is needed. Faster and more memory-efficient
    than GradientBoostingClassifier at this dataset size.

    Returns
    -------
    ModelSpec
        name='gbm', pipeline with HistGradientBoostingClassifier only.
    """
    pipeline = Pipeline(
        [
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=_RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__max_iter": [100, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__min_samples_leaf": [20, 50],
        "classifier__class_weight": [None, "balanced"],
    }

    return ModelSpec(name="gbm", pipeline=pipeline, param_grid=param_grid)


def voting_ensemble(
    base_pipelines: list[tuple[str, Pipeline]],
) -> ModelSpec:
    """
    Soft-voting ensemble of pre-configured base pipelines.

    Takes pipelines that already have their best hyperparameters set (via
    GridSearchCV in train_sklearn.py) and wraps them in a VotingClassifier.
    The ensemble is then fit on the combined train+val set, at which point
    VotingClassifier clones and fits each sub-pipeline internally.

    Not in MODEL_REGISTRY. To be assembled in train_sklearn.py after base models
    are tuned, so it requires arguments rather than being self-contained.

    Parameters
    ----------
    base_pipelines:
        List of (name, pipeline) tuples where each pipeline has best params
        set but is not yet fitted. Typically [('lr', ...), ('rf', ...), ('gbm', ...)].

    Returns
    -------
    ModelSpec
        name='ensemble', no param_grid (ensemble is not tuned separately).
    """
    vc = VotingClassifier(estimators=base_pipelines, voting="soft", n_jobs=-1)
    pipeline = Pipeline([("classifier", vc)])
    return ModelSpec(name="ensemble", pipeline=pipeline, param_grid={})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps CLI model names to factory functions.
MODEL_REGISTRY: dict[str, Callable[[], ModelSpec]] = {
    "lr": logistic_regression,
    "rf": random_forest,
    "gbm": gradient_boosted_trees,
}


def get_model(name: str) -> ModelSpec:
    """
    Instantiate a ModelSpec by name.

    Parameters
    ----------
    name:
        One of 'lr', 'rf', 'gbm'.

    Returns
    -------
    ModelSpec

    Raises
    ------
    ValueError
        If name is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Valid options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
