-- silver.sql
-- DDL for the silver layer: cleaned, renamed, and joined data.
-- Tables are created and populated by `make silver` (src/silver/build_silver.py).
-- This file is reference documentation; it is not executed by the pipeline.

CREATE SCHEMA IF NOT EXISTS silver;

-- ---------------------------------------------------------------------------
-- silver.halos
-- Cleaned AHF catalogue with readable column names, host/subhalo flags,
-- particle-count quality flag, and simulation metadata joined from
-- bronze.simulations. Gas/star columns (all zero in DM-only runs) are dropped.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.halos (
    simulation_id               VARCHAR,
    halo_id                     BIGINT,
    -- Structural
    host_halo_id                BIGINT,   -- 0 = field halo (no parent)
    n_substructure              BIGINT,
    m_halo_msun_h               DOUBLE,   -- halo mass [h^-1 Msun]
    n_particles                 BIGINT,
    -- Position [h^-1 kpc]
    x_kpc_h                     DOUBLE,
    y_kpc_h                     DOUBLE,
    z_kpc_h                     DOUBLE,
    -- Velocity [km/s]
    vx_km_s                     DOUBLE,
    vy_km_s                     DOUBLE,
    vz_km_s                     DOUBLE,
    -- Radii [h^-1 kpc]
    r_halo_kpc_h                DOUBLE,
    r_max_kpc_h                 DOUBLE,
    r_scale_kpc_h               DOUBLE,
    mbp_offset_kpc_h            DOUBLE,
    com_offset_kpc_h            DOUBLE,
    -- Kinematics [km/s]
    v_max_km_s                  DOUBLE,
    v_esc_km_s                  DOUBLE,
    sigma_v_km_s                DOUBLE,   -- 3D velocity dispersion
    -- Spin
    spin_bullock                DOUBLE,
    spin_peebles                DOUBLE,
    -- Angular momentum (unit vector)
    l_x                         DOUBLE,
    l_y                         DOUBLE,
    l_z                         DOUBLE,
    -- Shape
    axis_ratio_b                DOUBLE,   -- b/a (intermediate/major)
    axis_ratio_c                DOUBLE,   -- c/a (minor/major)
    -- Principal axes of moment of inertia tensor
    ea_x DOUBLE, ea_y DOUBLE, ea_z DOUBLE,
    eb_x DOUBLE, eb_y DOUBLE, eb_z DOUBLE,
    ec_x DOUBLE, ec_y DOUBLE, ec_z DOUBLE,
    -- Profile / thermodynamic
    overdensity                 DOUBLE,
    n_bins                      BIGINT,
    f_mass_highres              DOUBLE,
    e_kin_msun_km2_s2_h         DOUBLE,
    e_pot_msun_km2_s2_h         DOUBLE,
    surf_pressure_msun_km2_s2_h DOUBLE,
    phi0_km2_s2                 DOUBLE,
    c_nfw                       DOUBLE,
    -- Derived flags
    is_host                     BOOLEAN,  -- TRUE if host_halo_id == 0
    is_subhalo                  BOOLEAN,  -- TRUE if host_halo_id != 0
    below_100_part_limit        BOOLEAN,  -- TRUE if n_particles < 100
    -- Simulation metadata (joined from bronze.simulations)
    cosmology                   VARCHAR,
    softening                   VARCHAR,
    n_part_per_side             INTEGER,
    z_ini                       FLOAT,
    box_size_mpc_h              FLOAT,
    particle_mass_msun_h        FLOAT,
    PRIMARY KEY (simulation_id, halo_id)
);

-- ---------------------------------------------------------------------------
-- silver.protohalos
-- silver.halos extended with protohalo shape columns from bronze.protohalo_shapes
-- via a left join. Halos with no matching protohalo record retain null shape
-- columns; has_protohalo_data = FALSE for these rows.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.protohalos (
    -- All columns from silver.halos (see above) ...
    simulation_id               VARCHAR,
    halo_id                     BIGINT,
    host_halo_id                BIGINT,
    n_substructure              BIGINT,
    m_halo_msun_h               DOUBLE,
    n_particles                 BIGINT,
    x_kpc_h                     DOUBLE,
    y_kpc_h                     DOUBLE,
    z_kpc_h                     DOUBLE,
    vx_km_s                     DOUBLE,
    vy_km_s                     DOUBLE,
    vz_km_s                     DOUBLE,
    r_halo_kpc_h                DOUBLE,
    r_max_kpc_h                 DOUBLE,
    r_scale_kpc_h               DOUBLE,
    mbp_offset_kpc_h            DOUBLE,
    com_offset_kpc_h            DOUBLE,
    v_max_km_s                  DOUBLE,
    v_esc_km_s                  DOUBLE,
    sigma_v_km_s                DOUBLE,
    spin_bullock                DOUBLE,
    spin_peebles                DOUBLE,
    l_x                         DOUBLE,
    l_y                         DOUBLE,
    l_z                         DOUBLE,
    axis_ratio_b                DOUBLE,
    axis_ratio_c                DOUBLE,
    ea_x DOUBLE, ea_y DOUBLE, ea_z DOUBLE,
    eb_x DOUBLE, eb_y DOUBLE, eb_z DOUBLE,
    ec_x DOUBLE, ec_y DOUBLE, ec_z DOUBLE,
    overdensity                 DOUBLE,
    n_bins                      BIGINT,
    f_mass_highres              DOUBLE,
    e_kin_msun_km2_s2_h         DOUBLE,
    e_pot_msun_km2_s2_h         DOUBLE,
    surf_pressure_msun_km2_s2_h DOUBLE,
    phi0_km2_s2                 DOUBLE,
    c_nfw                       DOUBLE,
    is_host                     BOOLEAN,
    is_subhalo                  BOOLEAN,
    below_100_part_limit        BOOLEAN,
    cosmology                   VARCHAR,
    softening                   VARCHAR,
    n_part_per_side             INTEGER,
    z_ini                       FLOAT,
    box_size_mpc_h              FLOAT,
    particle_mass_msun_h        FLOAT,
    -- Protohalo shape columns (NULL where has_protohalo_data = FALSE)
    sphericity_s                DOUBLE,   -- s = lambda_c / lambda_a at half-max mass
    a_hmm                       DOUBLE,   -- scale factor at half-max mass
    m_hmm                       DOUBLE,   -- half-maximum mass [h^-1 Msun]
    -- Coverage flag
    has_protohalo_data          BOOLEAN,
    PRIMARY KEY (simulation_id, halo_id)
);

-- ---------------------------------------------------------------------------
-- silver.wdm_cdm_matches
-- Best CDM counterpart per WDM halo above the merit threshold (default 0.1),
-- one row per WDM halo present in bronze.wdm_cdm_matches.
-- WDM haloes absent from bronze entirely (zero particle overlap) are NOT here;
-- they appear as unmatched rows when gold.labels left-joins via simulation_pairs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.wdm_cdm_matches (
    simulation_pair_id  VARCHAR,
    wdm_halo_id         BIGINT,
    best_cdm_halo_id    BIGINT,   -- NULL if no counterpart above threshold
    best_merit          DOUBLE,   -- NULL if no counterpart above threshold
    has_cdm_counterpart BOOLEAN,  -- FALSE = spurious by merit criterion
    PRIMARY KEY (simulation_pair_id, wdm_halo_id)
);

-- ---------------------------------------------------------------------------
-- silver.simulation_pairs
-- Bridge table mapping each crossmatch configuration id (e.g. 'z39_adapt')
-- to its constituent WDM and CDM simulation ids.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.simulation_pairs (
    simulation_pair_id VARCHAR PRIMARY KEY,
    wdm_simulation_id  VARCHAR,
    cdm_simulation_id  VARCHAR,
    z_ini              INTEGER,
    softening          VARCHAR,
    n_part_per_side    INTEGER
);
