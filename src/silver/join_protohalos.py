"""
join_protohalos.py: build silver.protohalos from silver.halos and bronze.protohalo_shapes.

Left-joins protohalo shape records onto the full halo catalogue. Halos with
no matching shape record (roughly 25 to 44% depending on simulation) are retained with
null shape columns and has_protohalo_data=False.

Coverage is limited by the extra requirements of a 10-snapshot minimum merger tree history
and the half-maximum-mass selection criteria applied when computing protohalo shapes
(see paper Section 3.2.2).
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

log = logging.getLogger(__name__)


def join_protohalos(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Build silver.protohalos from silver.halos and bronze.protohalo_shapes.

    Writes results to silver.protohalos, replacing any existing table.

    Parameters
    ----------
    conn:
        Open DuckDB connection with bronze and silver.halos populated.
    """
    log.info("Building silver.protohalos ...")

    halos: pl.DataFrame = conn.execute("SELECT * FROM silver.halos").pl()
    shapes: pl.DataFrame = conn.execute(
        "SELECT halo_id, simulation_id, sphericity_s, a_hmm, m_hmm FROM bronze.protohalo_shapes"
    ).pl()

    # Left join: every halo is retained. Shape columns are null where no
    # protohalo record exists for that halo.
    protohalos = halos.join(
        shapes,
        on=["halo_id", "simulation_id"],
        how="left",
    )

    # Flag halos that have a protohalo record. Null sphericity_s is the
    # reliable indicator, it is never null for matched rows.
    protohalos = protohalos.with_columns(
        pl.col("sphericity_s").is_not_null().alias("has_protohalo_data"),
    )

    _ = conn.register("_silver_protohalos", protohalos)
    _ = conn.execute(
        "CREATE OR REPLACE TABLE silver.protohalos AS SELECT * FROM _silver_protohalos"
    )
    _ = conn.unregister("_silver_protohalos")

    n_total = len(protohalos)
    n_matched = protohalos["has_protohalo_data"].sum()
    log.info(
        f"silver.protohalos: {n_total} rows written ({n_matched} with protohalo data,"
        + f" {n_total - n_matched} without,"
        + f" {100 * n_matched / n_total if n_total > 0 else 0.0}% coverage)"
    )
