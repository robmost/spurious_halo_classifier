"""
train_pytorch.py: training entrypoint for PyTorch MLP models.

Invoked by `make train` or directly:

    python -m src.models.train_pytorch --model all --split all
    python -m src.models.train_pytorch --model mlp_mask --split cross_z_ini

For each (model, split) combination:
  1. Load features and labels from gold tables.
  2. Preprocess to tensors: mean imputation (mlp_impute) or zero-fill plus
     binary missingness indicators (mlp_mask).
  3. Train with BCEWithLogitsLoss, pos_weight for class imbalance, Adam
     optimiser, ReduceLROnPlateau scheduler, and early stopping on val loss.
  4. Evaluate best checkpoint on the test set.
  5. Log parameters, metrics, and the model to MLflow.
  6. Save the model state dict to models/.

Device priority: MPS (Apple Silicon) -> CUDA -> CPU.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TypedDict, cast

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow.pytorch import log_model as mlflow_pytorch_log_model
from torch.utils.data import DataLoader, TensorDataset

from src.config import AppConfig, configure_logging, load_config
from src.db import get_connection
from src.gold.features import FEATURE_COLS
from src.models.data import SplitArrays, load_split_data
from src.models.evaluate import compute_metrics
from src.models.mlp import (
    ALL_MLP_NAMES,
    MLP,
    MLP_REGISTRY,
    NULLABLE_FEATURES,
    MLPSpec,
    build_model,
)

configure_logging(force=True)  # mlflow.pytorch configures the root logger on import
log = logging.getLogger(__name__)

_MODELS_DIR = Path("models")
_ALL_SPLITS = ["within_sim", "cross_softening", "cross_z_ini"]

# Training hyperparameters.
_LEARNING_RATE: float = 1e-3
_BATCH_SIZE: int = 256
_MAX_EPOCHS: int = 400
_PATIENCE: int = 30  # early stopping patience in epochs
_LR_PATIENCE: int = 15  # ReduceLROnPlateau patience in epochs
_LR_FACTOR: float = 0.5  # LR reduction factor on plateau


class TorchSplitData(TypedDict):
    """
    Preprocessed tensors ready for model training.
    """

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_trainval: torch.Tensor
    y_trainval: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    col_means: np.ndarray  # training-set column means used for imputation


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def train_pytorch(
    cfg: AppConfig,
    model_names: list[str],
    split_names: list[str],
) -> None:
    """
    Train and evaluate PyTorch MLP models for the given models and splits.

    Parameters
    ----------
    cfg:
        Loaded application configuration.
    model_names:
        List of model names to train (e.g. ['mlp_impute', 'mlp_mask']).
    split_names:
        List of split names to train on (e.g. ['within_sim', 'cross_z_ini']).
    """
    _MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    _ = mlflow.set_experiment(cfg.mlflow.experiment)

    device = _get_device()
    log.info("Device: %s", device)

    conn = get_connection(cfg.database_path)
    try:
        for split_name in split_names:
            log.info("=== Split: %s ===", split_name)
            numpy_data = load_split_data(conn, split_name)
            log.info(
                "  train=%d  val=%d  test=%d",
                len(numpy_data["y_train"]),
                len(numpy_data["y_val"]),
                len(numpy_data["y_test"]),
            )
            for model_name in model_names:
                spec = MLP_REGISTRY[model_name]()
                torch_data = _preprocess(numpy_data, spec)
                _train_single_model(spec, torch_data, split_name, device)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data loading and preprocessing (private)
# ---------------------------------------------------------------------------


def _preprocess(data: SplitArrays, spec: MLPSpec) -> TorchSplitData:
    """
    Convert numpy arrays to float32 tensors with mean imputation (mlp_impute)
    or zero-fill plus binary missingness indicators (mlp_mask).
    """
    # Compute column means from training set only (NaN-aware).
    col_means = np.nanmean(data["X_train"], axis=0).astype(np.float32)

    # Indices of nullable protohalo features in FEATURE_COLS.
    nullable_idx = [FEATURE_COLS.index(f) for f in NULLABLE_FEATURES]

    def _apply(X: np.ndarray) -> np.ndarray:
        if spec.use_mask:
            # Build binary indicator columns before filling nulls.
            isnan = np.isnan(X)
            indicators = isnan[:, nullable_idx].astype(np.float32)
            X_filled = np.where(isnan, 0.0, X).astype(np.float32)
            return np.hstack([X_filled, indicators])
        # Mean imputation: replace NaN with training-set column mean.
        return np.where(np.isnan(X), col_means, X).astype(np.float32)

    def _label(arr: np.ndarray) -> torch.Tensor:
        # BCEWithLogitsLoss requires float; y arrives as int8 from load_split_data.
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)

    return {
        "X_train": torch.from_numpy(_apply(data["X_train"])),
        "y_train": _label(data["y_train"]),
        "X_val": torch.from_numpy(_apply(data["X_val"])),
        "y_val": _label(data["y_val"]),
        "X_trainval": torch.from_numpy(_apply(data["X_trainval"])),
        "y_trainval": _label(data["y_trainval"]),
        "X_test": torch.from_numpy(_apply(data["X_test"])),
        "y_test": _label(data["y_test"]),
        "col_means": col_means,
    }


# ---------------------------------------------------------------------------
# Training helpers (private)
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compute_pos_weight(y_train: torch.Tensor) -> torch.Tensor:
    # pos_weight = n_genuine / n_spurious; equivalent to class_weight='balanced' in sklearn.
    n_spurious = float(y_train.sum())
    n_genuine = float(len(y_train) - n_spurious)
    return torch.tensor([n_genuine / n_spurious], dtype=torch.float32)


def _train_epoch(
    model: MLP,
    loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    criterion: nn.BCEWithLogitsLoss,
    optimiser: torch.optim.Adam,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean batch loss."""
    _ = model.train()
    total_loss = 0.0
    total_samples = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        _ = optimiser.step()
        n = len(X_batch)
        total_loss += float(loss.item()) * n
        total_samples += n
    return total_loss / total_samples


def _evaluate(
    model: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (loss, y_true, y_pred_proba) without computing gradients."""
    _ = model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        loss = float(criterion(logits, y.to(device)).item())
        proba = torch.sigmoid(logits).cpu().numpy().ravel()
    return loss, y.numpy().ravel(), proba


def _train_single_model(
    spec: MLPSpec,
    data: TorchSplitData,
    split_name: str,
    device: torch.device,
) -> None:
    """
    Train one MLP variant, apply early stopping, evaluate on test, log to MLflow.

    Training uses train+val for the final refit after early stopping determines
    the optimal number of epochs. This mirrors the sklearn approach of refitting
    on train+val after GridSearchCV selects hyperparameters.

    Parameters
    ----------
    spec:
        MLPSpec from the registry.
    data:
        Preprocessed tensors from _preprocess.
    split_name:
        Used for run naming and artefact paths.
    device:
        Active compute device.
    """
    run_name = f"{spec.name}_{split_name}"
    log.info("Training %s ...", run_name)

    # Phase 1: train on train split with early stopping to find best epoch.
    model = build_model(spec).to(device)
    pos_weight = _compute_pos_weight(data["y_train"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=_LR_FACTOR, patience=_LR_PATIENCE
    )

    dataset = TensorDataset(data["X_train"], data["y_train"])
    loader = DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = _MODELS_DIR / f"{run_name}_checkpoint.pt"

    for epoch in range(1, _MAX_EPOCHS + 1):
        train_loss = _train_epoch(model, loader, criterion, optimiser, device)
        val_loss, _, _ = _evaluate(model, data["X_val"], data["y_val"], criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            log.info(
                "  epoch %3d  train_loss=%.4f  val_loss=%.4f  best_epoch=%d",
                epoch,
                train_loss,
                val_loss,
                best_epoch,
            )

        if patience_counter >= _PATIENCE:
            log.info("  Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    log.info("  Best val loss: %.4f at epoch %d", best_val_loss, best_epoch)

    # Phase 2: refit on train+val for best_epoch epochs, then evaluate on test.
    model_final = build_model(spec).to(device)
    pos_weight_tv = _compute_pos_weight(data["y_trainval"]).to(device)
    criterion_tv = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tv)
    optimiser_tv = torch.optim.Adam(model_final.parameters(), lr=_LEARNING_RATE)

    dataset_tv = TensorDataset(data["X_trainval"], data["y_trainval"])
    loader_tv = DataLoader(dataset_tv, batch_size=_BATCH_SIZE, shuffle=True)

    for _ in range(best_epoch):
        _ = _train_epoch(model_final, loader_tv, criterion_tv, optimiser_tv, device)

    _, y_true, y_pred_proba = _evaluate(
        model_final, data["X_test"], data["y_test"], criterion_tv, device
    )
    metrics = compute_metrics(y_true, y_pred_proba)

    params: dict[str, object] = {
        "split_name": split_name,
        "model": spec.name,
        "use_mask": spec.use_mask,
        "input_dim": spec.input_dim,
        "hidden_dims": str(spec.hidden_dims),
        "dropout": spec.dropout,
        "best_epoch": best_epoch,
        "lr": _LEARNING_RATE,
        "batch_size": _BATCH_SIZE,
    }
    with mlflow.start_run(run_name=run_name):
        _ = mlflow.log_params(params)
        _ = mlflow.log_metrics(metrics)
        _ = mlflow_pytorch_log_model(model_final, name="model")

    artefact_path = _MODELS_DIR / f"{run_name}.pt"
    torch.save(model_final.state_dict(), artefact_path)

    # Remove the phase-1 checkpoint; only the final model is kept.
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    log.info(
        "  [%s] avg_precision=%.4f  roc_auc=%.4f  f1=%.4f",
        run_name,
        metrics["test_average_precision"],
        metrics["test_roc_auc"],
        metrics["test_f1"],
    )
    log.info("  Saved to %s", artefact_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PyTorch MLP models for the spurious halo classifier."
    )
    _ = parser.add_argument(
        "--model",
        choices=[*ALL_MLP_NAMES, "all"],
        default="all",
        help="Model variant to train. 'all' trains both mlp_impute and mlp_mask.",
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
    model_names = ALL_MLP_NAMES if model_arg == "all" else [model_arg]
    split_names = _ALL_SPLITS if split_arg == "all" else [split_arg]

    cfg = load_config()
    log.info("Database: %s", cfg.database_path)
    log.info("Models:   %s", model_names)
    log.info("Splits:   %s", split_names)

    train_pytorch(cfg, model_names, split_names)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
