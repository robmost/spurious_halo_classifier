"""
data.py: shared data loading for model training scripts.

Both train_sklearn.py and train_pytorch.py call load_split_data to get the
same numpy arrays. Each script applies its own dtype conversion and any
additional fields (e.g. test_fold for PredefinedSplit in sklearn) on top.
"""

from __future__ import annotations

from typing import TypedDict

import duckdb
import numpy as np
import polars as pl

from src.gold.features import FEATURE_COLS


class SplitArrays(TypedDict):
    """Raw numpy arrays for one train/val/test split."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_trainval: np.ndarray
    y_trainval: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_split_data(conn: duckdb.DuckDBPyConnection, split_name: str) -> SplitArrays:
    """Load features, labels, and split assignments for one split into numpy arrays."""
    feature_cols_sql = ", ".join(f"f.{c}" for c in FEATURE_COLS)

    query = f"""
        SELECT
            {feature_cols_sql},
            l.is_spurious_cdm_match AS label,
            s.split_role
        FROM gold.features f
        JOIN gold.labels l
            ON f.halo_id = l.halo_id AND f.simulation_id = l.simulation_id
        JOIN gold.train_test_splits s
            ON f.halo_id = s.halo_id AND f.simulation_id = s.simulation_id
        WHERE s.split_name = ?
          AND l.is_spurious_cdm_match IS NOT NULL
    """
    df: pl.DataFrame = conn.execute(query, [split_name]).pl()

    train = df.filter(pl.col("split_role") == "train")
    val = df.filter(pl.col("split_role") == "val")
    test = df.filter(pl.col("split_role") == "test")

    X_train = train.select(FEATURE_COLS).to_numpy()
    y_train = train["label"].cast(pl.Int8).to_numpy()
    X_val = val.select(FEATURE_COLS).to_numpy()
    y_val = val["label"].cast(pl.Int8).to_numpy()
    X_test = test.select(FEATURE_COLS).to_numpy()
    y_test = test["label"].cast(pl.Int8).to_numpy()

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_trainval": X_trainval,
        "y_trainval": y_trainval,
        "X_test": X_test,
        "y_test": y_test,
    }
