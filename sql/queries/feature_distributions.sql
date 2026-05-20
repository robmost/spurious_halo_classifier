-- feature_distributions.sql
-- Null fractions and basic distribution statistics for all 15 gold features,
-- split by cosmology and spurious label.
-- Run against the DuckDB file built by `make gold`.
--
-- Usage (DuckDB CLI):
--   duckdb data/spurious_halos.duckdb < sql/queries/feature_distributions.sql

-- ---------------------------------------------------------------------------
-- 1. Null fractions per feature (all haloes)
-- ---------------------------------------------------------------------------
SELECT
    'log10_m200'                           AS feature, COUNT(*) AS n_total, SUM(CASE WHEN log10_m200                           IS NULL THEN 1 ELSE 0 END) AS n_null FROM gold.features
UNION ALL SELECT 'log10_npart',                            COUNT(*), SUM(CASE WHEN log10_npart                            IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'x_norm',                                 COUNT(*), SUM(CASE WHEN x_norm                                 IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'y_norm',                                 COUNT(*), SUM(CASE WHEN y_norm                                 IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'z_norm',                                 COUNT(*), SUM(CASE WHEN z_norm                                 IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'v_disp_sigv',                            COUNT(*), SUM(CASE WHEN v_disp_sigv                            IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'concentration_cNFW',                     COUNT(*), SUM(CASE WHEN concentration_cNFW                     IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'spin_bullock',                           COUNT(*), SUM(CASE WHEN spin_bullock                           IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'axis_ratio_b_over_a',                    COUNT(*), SUM(CASE WHEN axis_ratio_b_over_a                    IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'axis_ratio_c_over_a',                    COUNT(*), SUM(CASE WHEN axis_ratio_c_over_a                    IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'sphericity_s',                           COUNT(*), SUM(CASE WHEN sphericity_s                           IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'log10_m_hmm',                            COUNT(*), SUM(CASE WHEN log10_m_hmm                            IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'a_hmm',                                  COUNT(*), SUM(CASE WHEN a_hmm                                  IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'dist_nearest_neighbour_mpc_h',           COUNT(*), SUM(CASE WHEN dist_nearest_neighbour_mpc_h           IS NULL THEN 1 ELSE 0 END) FROM gold.features
UNION ALL SELECT 'count_neighbours_within_radius_mpc_h',   COUNT(*), SUM(CASE WHEN count_neighbours_within_radius_mpc_h   IS NULL THEN 1 ELSE 0 END) FROM gold.features
ORDER BY n_null DESC;

-- ---------------------------------------------------------------------------
-- 2. Key feature statistics by cosmology and spurious label (WDM only)
-- ---------------------------------------------------------------------------
SELECT
    f.cosmology,
    l.is_spurious_cdm_match,
    COUNT(*)                                    AS n,
    ROUND(AVG(f.log10_m200),          3)        AS mean_log10_m200,
    ROUND(STDDEV(f.log10_m200),       3)        AS std_log10_m200,
    ROUND(AVG(f.v_disp_sigv),         2)        AS mean_v_disp_sigv,
    ROUND(STDDEV(f.v_disp_sigv),      2)        AS std_v_disp_sigv,
    ROUND(AVG(f.sphericity_s),        3)        AS mean_sphericity_s,
    ROUND(STDDEV(f.sphericity_s),     3)        AS std_sphericity_s,
    ROUND(AVG(f.concentration_cNFW),  3)        AS mean_concentration,
    ROUND(AVG(f.dist_nearest_neighbour_mpc_h), 4) AS mean_dist_nn_mpc_h
FROM gold.features f
JOIN gold.labels l
    ON f.halo_id = l.halo_id AND f.simulation_id = l.simulation_id
WHERE f.cosmology = 'WDM'
  AND l.is_spurious_cdm_match IS NOT NULL
GROUP BY f.cosmology, l.is_spurious_cdm_match
ORDER BY f.cosmology, l.is_spurious_cdm_match;
