-- label_counts.sql
-- Spurious/genuine label counts per WDM simulation for both label strategies.
-- Run against the DuckDB file built by `make gold`.
--
-- Usage (DuckDB CLI):
--   duckdb data/spurious_halos.duckdb < sql/queries/label_counts.sql

SELECT
    f.simulation_id,
    f.softening,
    f.z_ini,
    COUNT(*)                                                          AS n_wdm_halos,
    -- CDM-match label (primary ML label)
    SUM(CASE WHEN l.is_spurious_cdm_match = true  THEN 1 ELSE 0 END) AS cdm_match_spurious,
    SUM(CASE WHEN l.is_spurious_cdm_match = false THEN 1 ELSE 0 END) AS cdm_match_genuine,
    ROUND(
        100.0 * SUM(CASE WHEN l.is_spurious_cdm_match = true THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                                 AS pct_spurious_cdm_match,
    -- Empirical label (Mostoghiu Paun et al. 2025, eq. 4)
    SUM(CASE WHEN l.is_spurious_empirical = true  THEN 1 ELSE 0 END) AS empirical_spurious,
    SUM(CASE WHEN l.is_spurious_empirical = false THEN 1 ELSE 0 END) AS empirical_genuine,
    SUM(CASE WHEN l.is_spurious_empirical IS NULL THEN 1 ELSE 0 END) AS empirical_null,
    -- Label agreement
    SUM(CASE WHEN l.label_agreement = true  THEN 1 ELSE 0 END)       AS labels_agree,
    SUM(CASE WHEN l.label_agreement = false THEN 1 ELSE 0 END)        AS labels_disagree,
    SUM(CASE WHEN l.label_agreement IS NULL THEN 1 ELSE 0 END)        AS agreement_null
FROM gold.features f
JOIN gold.labels l
    ON f.halo_id = l.halo_id AND f.simulation_id = l.simulation_id
WHERE f.cosmology = 'WDM'
GROUP BY f.simulation_id, f.softening, f.z_ini
ORDER BY f.simulation_id;
