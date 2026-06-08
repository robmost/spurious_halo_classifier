"""
mlp.py: PyTorch MLP model definitions for the spurious halo classifier.

Two variants are defined and registered:

  - mlp_impute:     MLP trained on mean-imputed features. Input dimension matches
                    FEATURE_COLS exactly (15 features). Directly comparable to the
                    sklearn models since preprocessing is identical.

  - mlp_mask:       MLP trained on mean-imputed features plus three binary indicator
                    columns flagging missing protohalo data (sphericity_s, log10_m_hmm,
                    a_hmm). Input dimension is 18. Tests whether explicit missingness
                    signals improve genuine-class performance over blind imputation.

Architecture per hidden layer: Linear -> BatchNorm -> ReLU -> Dropout.
BatchNorm precedes ReLU (pre-activation normalisation) for training stability.
Output layer: single logit (no sigmoid) for use with BCEWithLogitsLoss.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import torch
import torch.nn as nn

from src.gold.features import FEATURE_COLS

log = logging.getLogger(__name__)

# Random seed for weight initialisation. Separate from the data split seed
# in config.yaml, which controls which halos end up in train/val/test.
_RANDOM_STATE: int = 55667788

# Protohalo features that are nullable. mlp_mask appends one binary indicator
# column per feature in this list, in order.
NULLABLE_FEATURES: list[str] = ["sphericity_s", "log10_m_hmm", "a_hmm"]

# Default hidden layer sizes. Applied to both variants.
_HIDDEN_DIMS: list[int] = [64, 32, 16]

# Default dropout probability applied after each hidden layer.
_DROPOUT: float = 0.3


# ---------------------------------------------------------------------------
# MLPSpec
# ---------------------------------------------------------------------------


@dataclass
class MLPSpec:
    """
    Container for an MLP variant name, architecture, and preprocessing flag.
    """

    name: str
    input_dim: int
    hidden_dims: list[int]
    dropout: float
    use_mask: bool  # if True, binary missingness indicators are appended to input


# ---------------------------------------------------------------------------
# MLP network
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """
    Fully connected MLP for binary spurious halo classification.

    Each hidden layer follows the pattern:
        Linear -> BatchNorm -> ReLU -> Dropout

    The output layer is a single linear unit with no activation.
    Use BCEWithLogitsLoss during training, which applies sigmoid internally
    for numerical stability.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dims:
        Sequence of hidden layer widths (e.g. [64, 32, 16]).
    dropout:
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim

        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = out_dim

        # Single logit output, with no sigmoid, as BCEWithLogitsLoss applies it.
        layers.append(nn.Linear(in_dim, 1))

        self.network: nn.Sequential = nn.Sequential(*layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------


def mlp_impute() -> MLPSpec:
    """Return MLPSpec for the mean-imputation variant (use_mask=False)."""
    return MLPSpec(
        name="mlp_impute",
        input_dim=len(FEATURE_COLS),
        hidden_dims=_HIDDEN_DIMS,
        dropout=_DROPOUT,
        use_mask=False,
    )


def mlp_mask() -> MLPSpec:
    """Return MLPSpec for the missingness-indicator variant (use_mask=True)."""
    return MLPSpec(
        name="mlp_mask",
        input_dim=len(FEATURE_COLS) + len(NULLABLE_FEATURES),
        hidden_dims=_HIDDEN_DIMS,
        dropout=_DROPOUT,
        use_mask=True,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps CLI model names to factory functions.
MLP_REGISTRY: dict[str, Callable[[], MLPSpec]] = {
    "mlp_impute": mlp_impute,
    "mlp_mask": mlp_mask,
}

# All registered MLP variant names. Imported by train_pytorch.py for CLI.
ALL_MLP_NAMES: list[str] = list(MLP_REGISTRY)


def get_mlp(name: str) -> MLPSpec:
    """
    Instantiate an MLPSpec by name.

    Parameters
    ----------
    name:
        One of 'mlp_impute', 'mlp_mask'.

    Returns
    -------
    MLPSpec

    Raises
    ------
    ValueError
        If name is not in the registry.
    """
    if name not in MLP_REGISTRY:
        raise ValueError(f"Unknown MLP variant '{name}'. Valid options: {list(MLP_REGISTRY)}")
    return MLP_REGISTRY[name]()


def build_model(spec: MLPSpec, seed: int = _RANDOM_STATE) -> MLP:
    """
    Instantiate and initialise an MLP from an MLPSpec.

    Weights are initialised with Kaiming uniform initialisation (He init),
    appropriate for ReLU activations. The seed is set before initialisation
    for reproducibility.

    Parameters
    ----------
    spec:
        MLPSpec produced by a factory function.
    seed:
        Random seed for weight initialisation.

    Returns
    -------
    MLP
        Initialised MLP ready for training.
    """
    _ = torch.manual_seed(seed)
    model = MLP(
        input_dim=spec.input_dim,
        hidden_dims=spec.hidden_dims,
        dropout=spec.dropout,
    )
    # Apply Kaiming uniform init to all linear layers, zeros for biases.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            _ = nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                _ = nn.init.zeros_(module.bias)
    log.info(
        "Built MLP '%s': input_dim=%d, hidden_dims=%s, dropout=%.2f",
        spec.name,
        spec.input_dim,
        spec.hidden_dims,
        spec.dropout,
    )
    return model
