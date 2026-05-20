"""
filter_matches.py: build silver.wdm_cdm_matches from bronze.wdm_cdm_matches.

Applies the merit threshold, reduces to one row per WDM halo (keeping the
best CDM counterpart by merit), and adds the has_cdm_counterpart boolean.

WDM haloes absent from bronze entirely (zero particle overlap with any CDM
halo) are not handled here, they appear as unmatched rows when gold.labels
left-joins silver.halos against this table via silver.simulation_pairs.
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

log = logging.getLogger(__name__)


def filter_matches(conn: duckdb.DuckDBPyConnection, merit_threshold: float) -> None:
    """
    Build silver.wdm_cdm_matches from bronze.wdm_cdm_matches.

    Writes results to silver.wdm_cdm_matches, replacing any existing table.

    Parameters
    ----------
    conn:
        Open DuckDB connection with bronze schema populated.
    merit_threshold:
        Minimum merit score for a CDM counterpart to be considered a match.
        Sourced from config.yaml, not hardcoded here.
    """
    log.info("Building silver.wdm_cdm_matches (merit threshold: %.3f) ...", merit_threshold)

    matches: pl.DataFrame = conn.execute(
        "SELECT simulation_pair_id, wdm_halo_id, cdm_halo_id, merit FROM bronze.wdm_cdm_matches"
    ).pl()

    # Separate pairs that pass the threshold from those that do not.
    above_matches = matches.filter(pl.col("merit") >= merit_threshold)

    # For WDM haloes with multiple CDM counterparts above threshold, keep the
    # best match (highest merit). Sort descending then take first per group.
    best_match = (
        above_matches.sort("merit", descending=True)
        .group_by(["simulation_pair_id", "wdm_halo_id"], maintain_order=True)
        .first()
        .rename({"cdm_halo_id": "best_cdm_halo_id", "merit": "best_merit"})
        .with_columns(pl.lit(True).alias("has_cdm_counterpart"))
    )

    # Handle WDM haloes present in bronze but with no match above threshold.
    # These have some particle overlap with CDM haloes but it is too weak
    # to count as a genuine counterpart.
    all_wdm = matches.select(["simulation_pair_id", "wdm_halo_id"]).unique()
    no_match = all_wdm.join(
        best_match.select(["simulation_pair_id", "wdm_halo_id"]),
        on=["simulation_pair_id", "wdm_halo_id"],
        how="anti",
    ).with_columns(
        pl.lit(None, dtype=pl.Int64).alias("best_cdm_halo_id"),
        pl.lit(None, dtype=pl.Float64).alias("best_merit"),
        pl.lit(False).alias("has_cdm_counterpart"),
    )

    result = pl.concat([best_match, no_match], how="diagonal")

    _ = conn.register("_silver_matches", result)
    _ = conn.execute(
        "CREATE OR REPLACE TABLE silver.wdm_cdm_matches AS SELECT * FROM _silver_matches"
    )
    _ = conn.unregister("_silver_matches")

    n_total = len(result)
    n_matched = result["has_cdm_counterpart"].sum()
    log.info(
        f"silver.wdm_cdm_matches: {n_total} WDM haloes ({n_matched} genuine,"
        + f" {n_total - n_matched} spurious by merit,"
        + f" {100 * (n_total - n_matched) / n_total if n_total > 0 else 0.0}%)"
    )
