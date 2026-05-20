"""
gold/labels.py: compute gold.labels for WDM haloes.

Two independent binary labels are derived for each WDM halo:

 - is_spurious_empirical: Reproduces equation 4 from Mostoghiu Paun et al. (2025).
                          A halo is spurious if its protohalo sphericity falls below
                          the empirical mean sphericity–mass relation at half-maximum mass.
                          Only applied below the mass threshold (default log10(M) = 10.0 h^-1 Msun);
                          haloes above the threshold are assumed genuine.

Caveat: the empirical relation was derived by visually separating the
mean sphericity trends of CDM and WDM samples (Fig. 6 of the paper).
Because the underlying sphericity distributions overlap significantly,
the relation classifies a large fraction of haloes as spurious and
should be interpreted as a population-level diagnostic rather than a
per-halo ground truth. It serves as the baseline label only.

- is_spurious_cdm_match: Primary ML label. A WDM halo is spurious if it has no CDM counterpart
                         above the merit threshold in the matched simulation pair. Independent of
                         sphericity, avoiding the circularity in the empirical approach.

CDM haloes receive null for both labels. They are the reference population,
not training examples.
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

from src.config import AppConfig

log = logging.getLogger(__name__)


def build_labels(conn: duckdb.DuckDBPyConnection, cfg: AppConfig) -> None:
    """
    Build gold.labels from silver tables.

    Writes results to gold.labels, replacing any existing table.
    WDM haloes only; CDM haloes are excluded from this table.

    Parameters
    ----------
    conn:
        Open DuckDB connection with silver schema populated.
    cfg:
        Loaded application configuration (provides empirical label parameters).
    """
    log.info("Building gold.labels ...")

    emp = cfg.gold.empirical_label
    alpha = emp.alpha
    beta = emp.beta
    mass_threshold = 10**emp.mass_threshold_log10  # h^-1 M_sun

    # Join WDM protohalos with CDM counterpart information via simulation_pairs.
    # A left join ensures WDM haloes absent from wdm_cdm_matches are retained
    # with null has_cdm_counterpart, which maps to is_spurious_cdm_match=True.
    df: pl.DataFrame = conn.execute("""
        SELECT
            h.halo_id,
            h.simulation_id,
            h.has_protohalo_data,
            h.sphericity_s,
            h.m_hmm,
            m.has_cdm_counterpart
        FROM silver.protohalos h
        LEFT JOIN silver.simulation_pairs sp
            ON h.simulation_id = sp.wdm_simulation_id
        LEFT JOIN silver.wdm_cdm_matches m
            ON sp.simulation_pair_id = m.simulation_pair_id
            AND h.halo_id = m.wdm_halo_id
        WHERE h.cosmology = 'WDM'
    """).pl()

    # --- is_spurious_empirical ---
    # Null if no protohalo data (sphericity unknown).
    # False if above mass threshold (assumed genuine regardless of sphericity).
    # Otherwise compare sphericity to the empirical mean relation.
    s_bar = alpha * (pl.col("m_hmm").log(base=10)) + beta
    df = df.with_columns(
        pl.when(~pl.col("has_protohalo_data"))
        .then(None)
        .when(pl.col("m_hmm") >= mass_threshold)
        .then(False)
        .when(pl.col("sphericity_s") < s_bar)
        .then(True)
        .otherwise(False)
        .alias("is_spurious_empirical")
        .cast(pl.Boolean)
    )

    # --- is_spurious_cdm_match ---
    # Null has_cdm_counterpart means the halo shares zero particles with any
    # CDM halo — the strongest spurious signal. Treat as False (no counterpart).
    df = df.with_columns(
        (~pl.col("has_cdm_counterpart").fill_null(False)).alias("is_spurious_cdm_match")
    )

    # --- label_agreement ---
    # Null if either label is null; Polars == propagates nulls naturally.
    df = df.with_columns(
        (pl.col("is_spurious_empirical") == pl.col("is_spurious_cdm_match")).alias(
            "label_agreement"
        )
    )

    # Keep only the label columns plus keys.
    labels = df.select(
        [
            "halo_id",
            "simulation_id",
            "is_spurious_empirical",
            "is_spurious_cdm_match",
            "label_agreement",
        ]
    )

    _ = conn.execute("CREATE SCHEMA IF NOT EXISTS gold")
    _ = conn.register("_gold_labels", labels)
    _ = conn.execute("CREATE OR REPLACE TABLE gold.labels AS SELECT * FROM _gold_labels")
    _ = conn.unregister("_gold_labels")

    n_total = len(labels)
    n_null_emp = int(labels["is_spurious_empirical"].is_null().sum())
    n_null_cdm = int(labels["is_spurious_cdm_match"].is_null().sum())
    n_null_agree = int(labels["label_agreement"].is_null().sum())
    n_spurious_emp = int(labels["is_spurious_empirical"].fill_null(False).sum())
    n_spurious_cdm = int(labels["is_spurious_cdm_match"].fill_null(False).sum())
    n_agree = int(labels["label_agreement"].fill_null(False).sum())

    log.info("gold.labels: %d WDM haloes", n_total)
    log.info(
        "  is_spurious_empirical : %d spurious, %d genuine, %d null",
        n_spurious_emp,
        n_total - n_spurious_emp - n_null_emp,
        n_null_emp,
    )
    log.info(
        "  is_spurious_cdm_match : %d spurious, %d genuine, %d null",
        n_spurious_cdm,
        n_total - n_spurious_cdm - n_null_cdm,
        n_null_cdm,
    )
    log.info(
        "  label_agreement       : %d agree, %d disagree, %d null",
        n_agree,
        n_total - n_agree - n_null_agree,
        n_null_agree,
    )
