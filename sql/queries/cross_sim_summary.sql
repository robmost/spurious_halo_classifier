-- cross_sim_summary.sql
-- Cross-simulation comparison of key population statistics.
-- Useful for verifying that the three evaluation splits (within_sim,
-- cross_softening, cross_z_ini) differ in the expected ways.
-- Run against the DuckDB file built by `make gold`.
--
-- Usage (DuckDB CLI):
--   duckdb data/spurious_halos.duckdb < sql/queries/cross_sim_summary.sql

-- ---------------------------------------------------------------------------
-- 1. Halo counts and spurious fractions per simulation
-- ---------------------------------------------------------------------------
SELECT
    f.simulation_id,
    f.cosmology,
    f.softening,
    f.z_ini,
    COUNT(*)                                                           AS n_halos,
    SUM(CASE WHEN f.cosmology = 'WDM' THEN 1 ELSE 0 END)             AS n_wdm,
    SUM(CASE WHEN f.cosmology = 'CDM' THEN 1 ELSE 0 END)             AS n_cdm,
    SUM(CASE WHEN l.is_spurious_cdm_match = true  THEN 1 ELSE 0 END) AS n_spurious,
    SUM(CASE WHEN l.is_spurious_cdm_match = false THEN 1 ELSE 0 END) AS n_genuine,
    ROUND(
        100.0 * SUM(CASE WHEN l.is_spurious_cdm_match = true THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN f.cosmology = 'WDM' THEN 1 ELSE 0 END), 0), 1
    )                                                                  AS pct_spurious
FROM gold.features f
LEFT JOIN gold.labels l
    ON f.halo_id = l.halo_id AND f.simulation_id = l.simulation_id
GROUP BY f.simulation_id, f.cosmology, f.softening, f.z_ini
ORDER BY f.cosmology, f.z_ini, f.softening;

-- ---------------------------------------------------------------------------
-- 2. Train/val/test set sizes per split
-- ---------------------------------------------------------------------------
SELECT
    s.split_name,
    s.split_role,
    f.simulation_id,
    COUNT(*)                                                           AS n_halos,
    SUM(CASE WHEN l.is_spurious_cdm_match = true  THEN 1 ELSE 0 END) AS n_spurious,
    SUM(CASE WHEN l.is_spurious_cdm_match = false THEN 1 ELSE 0 END) AS n_genuine,
    ROUND(
        100.0 * SUM(CASE WHEN l.is_spurious_cdm_match = true THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                                  AS pct_spurious
FROM gold.train_test_splits s
JOIN gold.features f
    ON s.halo_id = f.halo_id AND s.simulation_id = f.simulation_id
JOIN gold.labels l
    ON s.halo_id = l.halo_id AND s.simulation_id = l.simulation_id
GROUP BY s.split_name, s.split_role, f.simulation_id
ORDER BY s.split_name, s.split_role, f.simulation_id;

-- ---------------------------------------------------------------------------
-- 3. Key feature shifts between train and test simulations
--    (the metrics that drive cross-split generalisation behaviour)
-- ---------------------------------------------------------------------------
SELECT
    f.simulation_id,
    f.softening,
    f.z_ini,
    ROUND(AVG(f.log10_m200),      3) AS mean_log10_m200,
    ROUND(AVG(f.v_disp_sigv),     2) AS mean_v_disp_sigv,
    ROUND(AVG(f.a_hmm),           4) AS mean_a_hmm,         -- formation time (earlier in adaptive runs)
    ROUND(AVG(f.sphericity_s),    3) AS mean_sphericity_s,  -- ~6% lower at z_ini=99 than z_ini=39
    COUNT(*)                         AS n_wdm_halos
FROM gold.features f
JOIN gold.labels l
    ON f.halo_id = l.halo_id AND f.simulation_id = l.simulation_id
WHERE f.cosmology = 'WDM'
  AND l.is_spurious_cdm_match IS NOT NULL
GROUP BY f.simulation_id, f.softening, f.z_ini
ORDER BY f.z_ini, f.softening;
