"""
load_bronze.py: pipeline entrypoint for the bronze layer.

Invoked by `make bronze` as `python -m src.bronze.load_bronze`.

Loops over all simulations and crossmatch configs defined in config.yaml,
calls the three parsers (parse_ahf_halos, parse_matches, parse_protohalos), and writes the results
into the four bronze tables in the DuckDB database.

The DuckDB ingestion follows the same pattern:
    1. Register a virtual table (view) from a DataFrame
    2. Create a table from the virtual table
    3. Unregister the virtual table
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

from src.bronze.parse_ahf import parse_ahf_halos
from src.bronze.parse_matches import parse_matches
from src.bronze.parse_protohalos import parse_protohalos
from src.config import AppConfig, configure_logging, load_config
from src.db import get_connection

configure_logging()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def load_bronze(cfg: AppConfig) -> None:
    """
    Populate all four bronze tables from raw data files described in cfg.

    All tables are dropped and recreated on each run, bronze is idempotent
    (i.e. same output regardless of how many times it is run).

    Parameters
    ----------
    cfg:
        Loaded application configuration.
    """
    conn = get_connection(cfg.database_path)

    try:
        _load_simulations(conn, cfg)
        _load_ahf_halos(conn, cfg)
        _load_protohalo_shapes(conn, cfg)
        _load_wdm_cdm_matches(conn, cfg)
        _log_row_counts(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Internal (private) loaders, one per bronze table
# ---------------------------------------------------------------------------


def _load_simulations(conn: duckdb.DuckDBPyConnection, cfg: AppConfig) -> None:
    """
    Build bronze.simulations from config metadata.
    """
    rows = [
        {
            "simulation_id": sim.id,
            "cosmology": sim.cosmology,
            "softening": sim.softening,
            "n_part_per_side": sim.n_part_per_side,
            "z_ini": sim.z_ini,
            "box_size_mpc_h": sim.box_size_mpc_h,
            "particle_mass_msun_h": sim.particle_mass_msun_h,
            "raw_file_path": str(sim.ahf_halos_dir),
        }
        for sim in cfg.simulations
    ]
    df = pl.DataFrame(
        rows,
        schema={
            "simulation_id": pl.String,
            "cosmology": pl.String,
            "softening": pl.String,
            "n_part_per_side": pl.Int32,
            "z_ini": pl.Float32,
            "box_size_mpc_h": pl.Float32,
            "particle_mass_msun_h": pl.Float32,
            "raw_file_path": pl.String,
        },
    )
    _ = conn.register("_sims", df)
    _ = conn.execute("CREATE OR REPLACE TABLE bronze.simulations AS SELECT * FROM _sims")
    _ = conn.unregister("_sims")
    log.info("bronze.simulations: %d rows written", len(rows))


def _load_ahf_halos(conn: duckdb.DuckDBPyConnection, cfg: AppConfig) -> None:
    """
    Parse AHF halo catalogues for all simulations and write bronze.ahf_halos.
    """
    first = True
    for sim in cfg.simulations:
        log.info("Parsing AHF halos for %s ...", sim.id)
        df = parse_ahf_halos(sim.ahf_halos_dir, sim.id)
        _ = conn.register("_ahf", df)
        if first:
            _ = conn.execute(
                "CREATE OR REPLACE TABLE bronze.ahf_halos AS SELECT * FROM _ahf WHERE false"
            )
            first = False
        _ = conn.execute("INSERT INTO bronze.ahf_halos SELECT * FROM _ahf")
        _ = conn.unregister("_ahf")
        log.info("  -> %d rows", len(df))


def _load_protohalo_shapes(conn: duckdb.DuckDBPyConnection, cfg: AppConfig) -> None:
    """
    Parse protohalo shape files for all simulations and write bronze.protohalo_shapes.
    """
    first = True
    for sim in cfg.simulations:
        log.info("Parsing protohalo shapes for %s ...", sim.id)
        df = parse_protohalos(sim.sphericity_path, sim.id)
        _ = conn.register("_proto", df)
        if first:
            _ = conn.execute(
                "CREATE OR REPLACE TABLE bronze.protohalo_shapes"
                + " AS SELECT * FROM _proto WHERE false"
            )
            first = False
        _ = conn.execute("INSERT INTO bronze.protohalo_shapes SELECT * FROM _proto")
        _ = conn.unregister("_proto")
        log.info("  -> %d rows", len(df))


def _load_wdm_cdm_matches(conn: duckdb.DuckDBPyConnection, cfg: AppConfig) -> None:
    """
    Parse WDM->CDM crossmatch files for all configs and write bronze.wdm_cdm_matches.
    """
    first = True
    for cm in cfg.crossmatch:
        log.info("Parsing crossmatch for %s ...", cm.id)
        df = parse_matches(cm.wdm_cdm_path, cm.id)
        _ = conn.register("_matches", df)
        if first:
            _ = conn.execute(
                "CREATE OR REPLACE TABLE bronze.wdm_cdm_matches"
                + " AS SELECT * FROM _matches WHERE false"
            )
            first = False
        _ = conn.execute("INSERT INTO bronze.wdm_cdm_matches SELECT * FROM _matches")
        _ = conn.unregister("_matches")
        log.info("  -> %d rows", len(df))


def _log_row_counts(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Log row counts for all bronze tables.
    """
    tables = [
        "bronze.simulations",
        "bronze.ahf_halos",
        "bronze.protohalo_shapes",
        "bronze.wdm_cdm_matches",
    ]
    log.info("--- Bronze layer row counts ---")
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        count = row[0] if row is not None else 0
        log.info("  %-35s %d", table, count)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    log.info("Database: %s", cfg.database_path)
    load_bronze(cfg)
    log.info("Bronze layer complete.")
