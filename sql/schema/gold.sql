-- gold.sql
-- DDL for the gold layer: ML-ready labels, features, and train/test splits.
-- Tables are created and populated by `make gold` (src/gold/build_gold.py).
-- This file is reference documentation; it is not executed by the pipeline.

CREATE SCHEMA IF NOT EXISTS gold;

-- ---------------------------------------------------------------------------
-- gold.labels
-- Two independent binary spurious-halo labels for every WDM halo.
-- CDM haloes are excluded — they are the reference population, not training examples.
--
-- is_spurious_empirical
--   Reproduces eq. 4 of Mostoghiu Paun et al. (2025): a halo is spurious if
--   its protohalo sphericity falls below the empirical mean sphericity–mass
--   relation at half-maximum mass. NULL where has_protohalo_data = FALSE
--   (sphericity unknown). FALSE unconditionally above log10(M_hmm) = 10.0.
--
-- is_spurious_cdm_match
--   Primary ML label. A WDM halo is spurious if it has no CDM counterpart
--   above the merit threshold in the matched simulation pair. Independent of
--   sphericity; avoids the circularity in the empirical approach.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.labels (
    halo_id                BIGINT,
    simulation_id          VARCHAR,
    is_spurious_empirical  BOOLEAN,  -- NULL if no protohalo data
    is_spurious_cdm_match  BOOLEAN,  -- primary ML label; never NULL
    label_agreement        BOOLEAN,  -- NULL if either label is NULL
    PRIMARY KEY (halo_id, simulation_id)
);

-- ---------------------------------------------------------------------------
-- gold.features
-- All 15 ML features for every halo (WDM and CDM). Label filtering is applied
-- at train time by joining gold.labels. Three protohalo-derived features
-- (sphericity_s, log10_m_hmm, a_hmm) are NULL for ~25–44% of haloes
-- depending on simulation.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.features (
    -- Keys and metadata
    halo_id                              BIGINT,
    simulation_id                        VARCHAR,
    cosmology                            VARCHAR,   -- 'CDM' or 'WDM'
    softening                            VARCHAR,
    z_ini                                FLOAT,
    n_part_per_side                      INTEGER,
    -- AHF-derived features
    log10_m200                           DOUBLE,    -- log10(M_halo / h^-1 Msun)
    log10_npart                          DOUBLE,    -- log10(n_particles); collinear with log10_m200
    x_norm                               DOUBLE,    -- x / box_size (dimensionless, periodic [0,1))
    y_norm                               DOUBLE,
    z_norm                               DOUBLE,
    v_disp_sigv                          DOUBLE,    -- 3D velocity dispersion [km/s]
    concentration_cNFW                   DOUBLE,    -- NFW concentration parameter
    spin_bullock                         DOUBLE,
    axis_ratio_b_over_a                  DOUBLE,    -- b/a (intermediate/major)
    axis_ratio_c_over_a                  DOUBLE,    -- c/a (minor/major)
    -- Protohalo-derived features (NULL where has_protohalo_data = FALSE)
    sphericity_s                         DOUBLE,    -- s = lambda_c / lambda_a at half-max mass
    log10_m_hmm                          DOUBLE,    -- log10(half-max mass / h^-1 Msun)
    a_hmm                                DOUBLE,    -- scale factor at half-max mass
    -- Spatial environment features (periodic KD-tree, radius = 1.0 h^-1 Mpc)
    dist_nearest_neighbour_mpc_h         DOUBLE,
    count_neighbours_within_radius_mpc_h BIGINT,
    PRIMARY KEY (halo_id, simulation_id)
);

-- ---------------------------------------------------------------------------
-- gold.train_test_splits
-- Stratified split assignments for three evaluation scenarios.
-- Only WDM haloes with a non-null is_spurious_cdm_match are assigned.
--
-- split_name values:
--   within_sim       train/val/test within wdm_512_z39_fixed (60/20/20)
--   cross_softening  train on z39_fixed, test on z39_adapt
--   cross_z_ini      train on z39_fixed, test on z99_fixed
--
-- split_role values: 'train', 'val', 'test'
-- Val is always in-distribution (from the train simulation) in all splits.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gold.train_test_splits (
    split_name    VARCHAR,
    halo_id       BIGINT,
    simulation_id VARCHAR,
    split_role    VARCHAR,
    PRIMARY KEY (split_name, halo_id, simulation_id)
);
