"""
gold/features.py: build gold.features from silver tables.

Computes all ML-facing features. Three sources:

  - AHF catalogue features:  derived algebraically from silver.halos columns.
  - Protohalo features:      sphericity_s, log10_m_hmm, a_hmm from silver.protohalos.
                             Use null where has_protohalo_data is False.
  - Spatial features:        nearest-neighbour distance and neighbour count, computed
                             per simulation using a periodic KD-tree (scipy.spatial.KDTree).

All haloes (WDM and CDM) are included. Label filtering happens at train time.
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import polars as pl
from scipy.spatial import KDTree

from src.config import AppConfig

log = logging.getLogger(__name__)

# Feature column names written to gold.features.
# Changing these requires updating gold/splits.py and all model code.
_KEY_COLS = [
    "halo_id",
    "simulation_id",
    "cosmology",
    "softening",
    "z_ini",
    "n_part_per_side",
]

FEATURE_COLS = [
    "log10_m200",
    "log10_npart",
    "x_norm",
    "y_norm",
    "z_norm",
    "v_disp_sigv",
    "concentration_cNFW",
    "spin_bullock",
    "axis_ratio_b_over_a",
    "axis_ratio_c_over_a",
    "sphericity_s",
    "log10_m_hmm",
    "a_hmm",
    "dist_nearest_neighbour_mpc_h",
    "count_neighbours_within_radius_mpc_h",
]


def build_features(conn: duckdb.DuckDBPyConnection, cfg: AppConfig) -> None:
    """
    Build gold.features from silver.protohalos.

    Writes results to gold.features, replacing any existing table.

    Parameters
    ----------
    conn:
        Open DuckDB connection with silver schema populated.
    cfg:
        Loaded application configuration.
    """
    log.info("Building gold.features ...")

    # silver.protohalos contains all silver.halos columns plus protohalo
    # shape columns (null where has_protohalo_data is False).
    df: pl.DataFrame = conn.execute("SELECT * FROM silver.protohalos").pl()

    df = _compute_ahf_features(df)
    spatial = _compute_spatial_features(df, cfg.gold.spatial_features.neighbour_radius_mpc_h)
    df = df.join(spatial, on=["halo_id", "simulation_id"], how="left")

    features = df.select(_KEY_COLS + FEATURE_COLS)

    _ = conn.execute("CREATE SCHEMA IF NOT EXISTS gold")
    _ = conn.register("_gold_features", features)
    _ = conn.execute("CREATE OR REPLACE TABLE gold.features AS SELECT * FROM _gold_features")
    _ = conn.unregister("_gold_features")

    log.info(
        "gold.features: %d rows, %d feature columns",
        len(features),
        len(FEATURE_COLS),
    )
    n_null_sphericity = int(features["sphericity_s"].is_null().sum())
    log.info(
        "  sphericity_s null (no protohalo data): %d / %d (%.1f%%)",
        n_null_sphericity,
        len(features),
        100 * n_null_sphericity / len(features) if len(features) > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Internal (private) helpers
# ---------------------------------------------------------------------------


def _compute_ahf_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add algebraic AHF-derived features to the DataFrame.

    Parameters
    ----------
    df:
        silver.protohalos DataFrame.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with additional feature columns appended.
    """
    return df.with_columns(
        # Mass and particle count on log scale.
        pl.col("m_halo_msun_h").log(base=10).alias("log10_m200"),
        pl.col("n_particles").cast(pl.Float64).log(base=10).alias("log10_npart"),
        # Positions normalised by box size. Positions are in h^-1 kpc,
        # box_size_mpc_h is in h^-1 Mpc. Multiply by 1000 to match units.
        (pl.col("x_kpc_h") / (pl.col("box_size_mpc_h") * 1000)).alias("x_norm"),
        (pl.col("y_kpc_h") / (pl.col("box_size_mpc_h") * 1000)).alias("y_norm"),
        (pl.col("z_kpc_h") / (pl.col("box_size_mpc_h") * 1000)).alias("z_norm"),
        # Kinematic and structural properties. The column names match silver directly.
        pl.col("sigma_v_km_s").alias("v_disp_sigv"),
        pl.col("c_nfw").alias("concentration_cNFW"),
        pl.col("spin_bullock"),
        # Axis ratios are already b/a and c/a in silver.
        pl.col("axis_ratio_b").alias("axis_ratio_b_over_a"),
        pl.col("axis_ratio_c").alias("axis_ratio_c_over_a"),
        # Protohalo features: null where has_protohalo_data is False.
        pl.col("sphericity_s"),
        pl.col("m_hmm").log(base=10).alias("log10_m_hmm"),
        pl.col("a_hmm"),
    )


def _compute_spatial_features(
    df: pl.DataFrame,
    neighbour_radius_mpc_h: float,
) -> pl.DataFrame:
    """
    Compute per-halo spatial environment features using a periodic KD-tree.

    Builds one KD-tree per simulation to avoid cross-simulation contamination.
    Periodic boundary conditions match the simulation box topology.

    Parameters
    ----------
    df:
        silver.protohalos DataFrame; must contain x_kpc_h, y_kpc_h, z_kpc_h,
        box_size_mpc_h, halo_id, simulation_id.
    neighbour_radius_mpc_h:
        Search radius for the neighbour count feature, in h^-1 Mpc.
        Set in config.yaml as gold.spatial_features.neighbour_radius_mpc_h.

    Returns
    -------
    pl.DataFrame
        One row per halo with columns: halo_id, simulation_id,
        dist_nearest_neighbour_mpc_h, count_neighbours_within_radius_mpc_h.
    """
    results: list[pl.DataFrame] = []

    for (sim_id,), group in df.group_by(["simulation_id"], maintain_order=False):
        sim_id = str(sim_id)
        box_size: float = float(group["box_size_mpc_h"][0])

        # Convert positions from h^-1 kpc to h^-1 Mpc for the spatial computation.
        positions = np.column_stack(
            [
                group["x_kpc_h"].to_numpy() / 1000.0,
                group["y_kpc_h"].to_numpy() / 1000.0,
                group["z_kpc_h"].to_numpy() / 1000.0,
            ]
        )

        # boxsize enables periodic boundary conditions
        tree = KDTree(positions, boxsize=box_size)

        # k=2: as the first result is the point itself (distance 0), we take the second.
        dists, _ = tree.query(positions, k=2, workers=-1)
        dist_nn: np.ndarray = dists[:, 1]

        # return_length=True returns the count directly, avoiding large list allocation.
        # Subtract 1 to exclude the query point itself.
        counts_raw = tree.query_ball_point(
            positions,
            r=neighbour_radius_mpc_h,
            return_length=True,
            workers=-1,
        )
        count_within: np.ndarray = np.asarray(counts_raw) - 1

        results.append(
            pl.DataFrame(
                {
                    "halo_id": group["halo_id"],
                    "simulation_id": group["simulation_id"],
                    "dist_nearest_neighbour_mpc_h": dist_nn,
                    "count_neighbours_within_radius_mpc_h": count_within,
                }
            )
        )

        log.info("  Spatial features computed for simulation '%s': %d haloes", sim_id, len(group))

    return pl.concat(results, how="diagonal")
