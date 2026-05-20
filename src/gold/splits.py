"""
gold/splits.py: build gold.train_test_splits from gold.labels and config.

Three split types are implemented:

  - within_sim:       Stratified 60/20/20 train/val/test split within a single
                      simulation. Standard benchmark, not a test of generalisation.

  - cross_softening:  Train on fixed softening, test on tidal adaptive (same
                      z_ini). Val is carved from the train simulation (in-distribution).
                      Tests whether the classifier generalises across softening schemes.

  - cross_z_ini:      Train on z_ini=39, test on z_ini=99. Val is carved from the
                      train simulation (in-distribution). Most scientifically
                      interesting split. The paper (Fig. 7) shows the empirical
                      sphericity relation breaks down at z_ini=99, so structured
                      classifier failure is expected and worth quantifying.

All splits are stratified on is_spurious_cdm_match to preserve the high fraction (~81%)
of spurious CDM haloes across train, val, and test. Val is always in-distribution (carved
from the train simulation) so it can be used for GridSearchCV hyperparameter
tuning without leaking test information.

Only WDM haloes present in gold.labels with a non-null is_spurious_cdm_match
are assigned splits. Label-agreement filtering happens at train time.
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl
from sklearn.model_selection import train_test_split

from src.config import AppConfig, SplitConfig

log = logging.getLogger(__name__)


def build_splits(conn: duckdb.DuckDBPyConnection, cfg: AppConfig) -> None:
    """
    Build gold.train_test_splits from gold.labels and split config.

    Writes results to gold.train_test_splits, replacing any existing table.

    Parameters
    ----------
    conn:
        Open DuckDB connection with gold.labels populated.
    cfg:
        Loaded application configuration.
    """
    log.info("Building gold.train_test_splits ...")

    train_fraction = cfg.splits.train_fraction
    val_fraction = cfg.splits.val_fraction

    # Fetch halo IDs and primary label for stratification.
    # is_spurious_cdm_match is used for stratification only, it is not written to output.
    labels: pl.DataFrame = conn.execute("""
        SELECT halo_id, simulation_id, is_spurious_cdm_match
        FROM gold.labels
        WHERE is_spurious_cdm_match IS NOT NULL
    """).pl()

    splits: list[pl.DataFrame] = []

    splits.append(
        _build_within_sim_split(
            labels, cfg.splits.within_sim, train_fraction, val_fraction, cfg.splits.random_seed
        )
    )
    splits.append(
        _build_cross_split(
            labels,
            cfg.splits.cross_softening,
            "cross_softening",
            val_fraction,
            cfg.splits.random_seed,
        )
    )
    splits.append(
        _build_cross_split(
            labels, cfg.splits.cross_z_ini, "cross_z_ini", val_fraction, cfg.splits.random_seed
        )
    )

    result = pl.concat(splits, how="diagonal").select(
        ["split_name", "halo_id", "simulation_id", "split_role"]
    )

    _ = conn.register("_gold_splits", result)
    _ = conn.execute("CREATE OR REPLACE TABLE gold.train_test_splits AS SELECT * FROM _gold_splits")
    _ = conn.unregister("_gold_splits")

    log.info("gold.train_test_splits: %d rows written", len(result))
    _log_split_summary(result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_within_sim_split(
    labels: pl.DataFrame,
    split_cfg: SplitConfig,
    train_fraction: float,
    val_fraction: float,
    random_seed: int,
) -> pl.DataFrame:
    """
    Stratified three-way split within the simulation(s) listed in split_cfg.

    Parameters
    ----------
    labels:
        Labelled WDM haloes (halo_id, simulation_id, is_spurious_cdm_match).
    split_cfg:
        Split configuration from config.yaml. train and test reference the
        same simulation ID for a within-simulation split.
    train_fraction:
        Fraction of haloes assigned to train (e.g. 0.6).
    val_fraction:
        Fraction of haloes assigned to val (e.g. 0.2). Remainder → test.

    Returns
    -------
    pl.DataFrame
        Rows with columns: split_name, halo_id, simulation_id,
        split_role, is_spurious_cdm_match.
    """
    all_sim_ids = list(set(split_cfg.train + split_cfg.test))
    halos = labels.filter(pl.col("simulation_id").is_in(all_sim_ids))

    y = halos["is_spurious_cdm_match"].to_numpy()

    # First split: carve out test from the full set.
    test_fraction = 1.0 - train_fraction - val_fraction
    idx_trainval, idx_test = train_test_split(
        range(len(halos)),
        test_size=test_fraction,
        stratify=y,
        random_state=random_seed,
    )

    # Second split: carve val from the train+val remainder.
    # val_fraction is relative to the full set, so adjust for the smaller pool.
    val_fraction_of_trainval = val_fraction / (train_fraction + val_fraction)
    y_trainval = y[list(idx_trainval)]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=random_seed,
    )

    role_map = (
        {i: "train" for i in idx_train}
        | {i: "val" for i in idx_val}
        | {i: "test" for i in idx_test}
    )
    roles = [role_map[i] for i in range(len(halos))]

    return halos.with_columns(
        pl.lit("within_sim").alias("split_name"),
        pl.Series("split_role", roles),
    )


def _build_cross_split(
    labels: pl.DataFrame,
    split_cfg: SplitConfig,
    split_name: str,
    val_fraction: float,
    random_seed: int,
) -> pl.DataFrame:
    """
    Cross-simulation split with stratified in-distribution validation.

    Val is carved from the train simulation so it can be used for
    GridSearchCV without leaking information from the test simulation.

    Parameters
    ----------
    labels:
        Labelled WDM haloes (halo_id, simulation_id, is_spurious_cdm_match).
    split_cfg:
        Split configuration from config.yaml.
    split_name:
        Name written to the split_name column (e.g. 'cross_softening').
    val_fraction:
        Fraction of train-simulation haloes assigned to val (e.g. 0.2).
        Remainder of train simulation → train. All of test simulation → test.

    Returns
    -------
    pl.DataFrame
        Rows with columns: split_name, halo_id, simulation_id,
        split_role, is_spurious_cdm_match.
    """
    train_sim_halos = labels.filter(pl.col("simulation_id").is_in(split_cfg.train))
    y_train_sim = train_sim_halos["is_spurious_cdm_match"].to_numpy()

    _, idx_val = train_test_split(
        range(len(train_sim_halos)),
        test_size=val_fraction,
        stratify=y_train_sim,
        random_state=random_seed,
    )

    roles = ["val" if i in set(idx_val) else "train" for i in range(len(train_sim_halos))]

    train_val = train_sim_halos.with_columns(
        pl.lit(split_name).alias("split_name"),
        pl.Series("split_role", roles),
    )

    test = labels.filter(pl.col("simulation_id").is_in(split_cfg.test)).with_columns(
        pl.lit(split_name).alias("split_name"),
        pl.lit("test").alias("split_role"),
    )

    return pl.concat([train_val, test], how="diagonal")


def _log_split_summary(result: pl.DataFrame) -> None:
    """
    Log train/val/test counts per split for a quick sanity check.
    """
    summary = (
        result.group_by(["split_name", "split_role"])
        .agg(pl.len().alias("n"))
        .sort(["split_name", "split_role"])
    )
    for row in summary.iter_rows(named=True):
        log.info(
            "  %-20s %-5s %d haloes",
            str(row["split_name"]),
            str(row["split_role"]),
            int(row["n"]),
        )
