"""
build_silver.py: pipeline entrypoint for the silver layer.

Invoked by `make silver` as `python -m src.silver.build_silver`.

Calls the three silver modules in dependency order, then builds
silver.simulation_pairs from config metadata.

The DuckDB ingestion follows the same pattern:
    1. Register a virtual table (view) from a DataFrame
    2. Create a table from the virtual table
    3. Unregister the virtual table
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

from src.config import AppConfig, configure_logging, load_config
from src.db import get_connection, log_row_counts
from src.silver.clean_halos import clean_halos
from src.silver.filter_matches import filter_matches
from src.silver.join_protohalos import join_protohalos

configure_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def build_silver(cfg: AppConfig) -> None:
    """
    Populate all silver tables from bronze.

    All tables are dropped and recreated on each run, silver is idempotent
    (i.e. same output regardless of how many times it is run).

    Parameters
    ----------
    cfg:
        Loaded application configuration.
    """
    conn = get_connection(cfg.database_path)

    try:
        clean_halos(conn)
        join_protohalos(conn)
        filter_matches(conn, cfg.silver.merit_threshold)
        _build_simulation_pairs(conn, cfg)
        log_row_counts(conn, _SILVER_TABLES, "Silver")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Internal (private) helpers
# ---------------------------------------------------------------------------


def _build_simulation_pairs(conn: duckdb.DuckDBPyConnection, cfg: AppConfig) -> None:
    """
    Build silver.simulation_pairs from crossmatch config.

    Maps each simulation_pair_id (e.g. 'z39_adapt') to the corresponding
    WDM and CDM simulation ids. This is the bridge table that lets gold join
    silver.halos to silver.wdm_cdm_matches.
    """
    rows = []
    for cm in cfg.crossmatch:
        # Find the matching WDM and CDM simulations by z_ini and softening.
        wdm_sims = [
            sim
            for sim in cfg.simulations
            if sim.cosmology == "WDM" and sim.z_ini == cm.z_ini and sim.softening == cm.softening
        ]
        cdm_sims = [
            sim
            for sim in cfg.simulations
            if sim.cosmology == "CDM" and sim.z_ini == cm.z_ini and sim.softening == cm.softening
        ]

        if len(wdm_sims) != 1:
            raise ValueError(
                f"Expected exactly one WDM simulation for crossmatch '{cm.id}' "
                + f"(z_ini={cm.z_ini}, softening='{cm.softening}'), found {len(wdm_sims)}"
            )
        if len(cdm_sims) != 1:
            raise ValueError(
                f"Expected exactly one CDM simulation for crossmatch '{cm.id}' "
                + f"(z_ini={cm.z_ini}, softening='{cm.softening}'), found {len(cdm_sims)}"
            )

        rows.append(
            {
                "simulation_pair_id": cm.id,
                "wdm_simulation_id": wdm_sims[0].id,
                "cdm_simulation_id": cdm_sims[0].id,
                "z_ini": cm.z_ini,
                "softening": cm.softening,
                "n_part_per_side": wdm_sims[0].n_part_per_side,
            }
        )

    df = pl.DataFrame(
        rows,
        schema={
            "simulation_pair_id": pl.String,
            "wdm_simulation_id": pl.String,
            "cdm_simulation_id": pl.String,
            "z_ini": pl.Int32,
            "softening": pl.String,
            "n_part_per_side": pl.Int32,
        },
    )

    _ = conn.register("_sim_pairs", df)
    _ = conn.execute("CREATE OR REPLACE TABLE silver.simulation_pairs AS SELECT * FROM _sim_pairs")
    _ = conn.unregister("_sim_pairs")
    log.info("silver.simulation_pairs: %d rows written", len(rows))


_SILVER_TABLES = [
    "silver.halos",
    "silver.protohalos",
    "silver.wdm_cdm_matches",
    "silver.simulation_pairs",
]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    log.info("Database: %s", cfg.database_path)
    build_silver(cfg)
    log.info("Silver layer complete.")
