"""
Tests for src/models/evaluate.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.evaluate import compute_metrics, load_mlp_model, load_model


class TestComputeMetrics:
    def _binary_data(self) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        return y_true, y_proba

    def test_returns_all_eight_keys(self) -> None:
        metrics = compute_metrics(*self._binary_data())
        assert set(metrics.keys()) == {
            "test_average_precision",
            "test_roc_auc",
            "test_f1",
            "test_precision",
            "test_recall",
            "test_f1_genuine",
            "test_precision_genuine",
            "test_recall_genuine",
        }

    def test_all_values_are_floats(self) -> None:
        for v in compute_metrics(*self._binary_data()).values():
            assert isinstance(v, float)

    def test_perfect_classifier_has_roc_auc_one(self) -> None:
        metrics = compute_metrics(*self._binary_data())
        assert metrics["test_roc_auc"] == pytest.approx(1.0)

    def test_roc_auc_within_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=100)
        y_proba = rng.random(100)
        metrics = compute_metrics(y_true, y_proba)
        assert 0.0 <= metrics["test_roc_auc"] <= 1.0

    def test_threshold_controls_binary_predictions(self) -> None:
        # All spurious haloes have proba=0.4
        y_true = np.array([0, 1, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.4, 0.4])
        # High threshold -> all predicted genuine -> spurious recall = 0
        metrics_high = compute_metrics(y_true, y_proba, threshold=0.5)
        # Low threshold -> all predicted spurious -> spurious recall = 1
        metrics_low = compute_metrics(y_true, y_proba, threshold=0.3)
        assert metrics_high["test_recall"] == pytest.approx(0.0)
        assert metrics_low["test_recall"] == pytest.approx(1.0)

    def test_genuine_metrics_use_pos_label_zero(self) -> None:
        # Perfect classifier for genuine class
        y_true = np.array([0, 0, 0, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.9])
        metrics = compute_metrics(y_true, y_proba, threshold=0.5)
        assert metrics["test_recall_genuine"] == pytest.approx(1.0)
        assert metrics["test_precision_genuine"] == pytest.approx(1.0)


class TestLoadModel:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            _ = load_model(tmp_path / "missing.joblib")


class TestLoadMlpModel:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            _ = load_mlp_model("mlp_impute", tmp_path / "missing.pt")
