"""
evaluate.py: shared evaluation utilities for the spurious halo classifier.

Provides metric computation and plotting functions used by both train_sklearn.py
and train_pytorch.py, and imported directly by notebooks for post-hoc analysis
of saved models.

Plotting functions follow the ax=None pattern: if an Axes is supplied the plot
is drawn onto it (for multi-panel notebook figures); if not, a new figure is
created and returned.

Usage example:
    from src.models.evaluate import compute_metrics, plot_pr_curve, load_model
    from src.utils.plotting import apply_style

    apply_style()
    pipeline = load_model("models/rf_cross_z_ini.joblib")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    plot_pr_curve(y_test, {"Random Forest": y_pred_proba}, ax=ax)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.models.mlp import MLP, build_model, get_mlp
from src.utils.plotting import apply_style

log = logging.getLogger(__name__)

# Class labels used on plot axes and confusion matrix displays.
_CLASS_LABELS: list[str] = ["Genuine", "Spurious"]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute classification metrics at a given probability threshold.

    Reports metrics for both the spurious class (positive label = 1) and the
    genuine class (positive label = 0).

    NOTE: Genuine-class metrics matter most here: with ~81% spurious,
    a lazy classifier can score well on spurious metrics
    while completely ignoring the minority genuine class.

    Parameters
    ----------
    y_true:
        True binary labels (0 = genuine, 1 = spurious).
    y_pred_proba:
        Predicted probabilities for the spurious class (label 1).
    threshold:
        Decision threshold applied to y_pred_proba to obtain binary predictions.

    Returns
    -------
    dict[str, float]
        Metric names and values. Keys:
        test_average_precision, test_roc_auc,
        test_f1, test_precision, test_recall,
        test_f1_genuine, test_precision_genuine, test_recall_genuine.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "test_average_precision": float(average_precision_score(y_true, y_pred_proba)),
        "test_roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "test_f1_genuine": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "test_precision_genuine": float(
            precision_score(y_true, y_pred, pos_label=0, zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
        "test_recall_genuine": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),  # pyright: ignore[reportArgumentType]
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_pr_curve(
    y_true: np.ndarray,
    results: dict[str, np.ndarray],
    ax: Axes | None = None,
) -> Axes:
    """
    Plot precision-recall curves for one or more models on a single axes.

    Parameters
    ----------
    y_true:
        True binary labels (0 = genuine, 1 = spurious).
    results:
        Mapping of model name to predicted probabilities for the spurious class.
        Multiple models are overlaid using the project colour cycle.
    ax:
        Axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    Axes
        The axes containing the plot.
    """
    apply_style()
    if ax is None:
        _, ax = plt.subplots()

    # Baseline: a random classifier achieves precision = class prevalence.
    prevalence = float(y_true.mean())
    _ = ax.axhline(
        prevalence,
        linestyle="--",
        color="grey",
        linewidth=1.0,
        label=f"Random (AP = {prevalence:.2f})",
    )

    for model_name, y_pred_proba in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        _ = ax.plot(recall, precision, label=f"{model_name} (AP = {ap:.3f})")

    _ = ax.set_xlabel("Recall")
    _ = ax.set_ylabel("Precision")
    _ = ax.set_xlim(0.0, 1.0)
    _ = ax.set_ylim(0.0, 1.05)
    _ = ax.legend()

    return ax


def plot_roc_curve(
    y_true: np.ndarray,
    results: dict[str, np.ndarray],
    ax: Axes | None = None,
) -> Axes:
    """
    Plot ROC curves for one or more models on a single axes.

    Parameters
    ----------
    y_true:
        True binary labels (0 = genuine, 1 = spurious).
    results:
        Mapping of model name to predicted probabilities for the spurious class.
        Multiple models are overlaid using the project colour cycle.
    ax:
        Axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    Axes
        The axes containing the plot.
    """
    apply_style()
    if ax is None:
        _, ax = plt.subplots()

    # Diagonal baseline: random classifier.
    _ = ax.plot(
        [0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.0, label="Random (AUC = 0.50)"
    )

    for model_name, y_pred_proba in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        _ = ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

    _ = ax.set_xlabel("False Positive Rate")
    _ = ax.set_ylabel("True Positive Rate")
    _ = ax.set_xlim(0.0, 1.0)
    _ = ax.set_ylim(0.0, 1.05)
    _ = ax.legend()

    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot a normalised confusion matrix for a single model.

    Displays both raw counts and row-normalised percentages so absolute
    numbers and relative rates are visible simultaneously.

    Parameters
    ----------
    y_true:
        True binary labels (0 = genuine, 1 = spurious).
    y_pred:
        Predicted binary labels at the chosen threshold.
    model_name:
        Used as the axes title.
    ax:
        Axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    Axes
        The axes containing the plot.
    """
    apply_style()
    if ax is None:
        _, ax = plt.subplots()

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    # Build annotation strings combining raw count and row-normalised percentage.
    # With text.usetex=True, '%' starts a TeX comment and must be escaped.
    percent_symbol = r"\%" if plt.rcParams.get("text.usetex", False) else "%"
    annotations: np.ndarray = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm_norm[i, j]
            annotations[i, j] = f"{cm[i, j]:,}\n({pct:.0f}{percent_symbol})"

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=_CLASS_LABELS,
    )
    _ = disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="")

    # Overwrite the default cell text with the combined annotation.
    # disp.text_ is populated by disp.plot() above.
    # The None guard satisfies the type checker since
    # text_ is ndarray | None before plot() is called.
    if disp.text_ is not None:
        for text, annotation in zip(disp.text_.ravel(), annotations.ravel()):
            _ = text.set_text(annotation)

    _ = ax.set_title(model_name)

    return ax


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(path: str | Path) -> Pipeline:
    """
    Load a saved sklearn pipeline from disk.

    Parameters
    ----------
    path:
        Path to a joblib file produced by train_sklearn.py.

    Returns
    -------
    Pipeline
        The fitted sklearn pipeline.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: '{path}'")
    pipeline: Pipeline = joblib.load(path)
    log.info("Loaded model from '%s'", path)
    return pipeline


def load_mlp_model(
    name: str,
    path: str | Path,
    device: torch.device | None = None,
) -> MLP:
    """
    Load a saved MLP state dict and return a ready-to-use model.

    Reconstructs the MLP architecture from the registry spec, loads the
    saved state dict, and sets the model to eval mode.

    Parameters
    ----------
    name:
        MLP variant name (e.g. 'mlp_impute', 'mlp_mask'). Must be a key
        in MLP_REGISTRY.
    path:
        Path to the .pt state dict file produced by train_pytorch.py.
    device:
        Device to load the model onto. Defaults to CPU if not specified.

    Returns
    -------
    MLP
        Loaded MLP in eval mode, ready for inference.

    Raises
    ------
    FileNotFoundError
        If the state dict file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MLP state dict not found: '{path}'")

    if device is None:
        device = torch.device("cpu")

    spec = get_mlp(name)
    model = build_model(spec)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    _ = model.load_state_dict(state_dict)
    _ = model.to(device)
    _ = model.eval()
    log.info("Loaded MLP '%s' from '%s'", name, path)
    return model
