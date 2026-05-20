-- bronze.sql
-- DDL for the bronze layer: raw ingested data.
-- Tables are created and populated by `make bronze` (src/bronze/load_bronze.py).
-- This file is reference documentation; it is not executed by the pipeline.

CREATE SCHEMA IF NOT EXISTS bronze;

-- ---------------------------------------------------------------------------
-- bronze.simulations
-- One row per simulation. Populated from config.yaml metadata.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.simulations (
    simulation_id        VARCHAR PRIMARY KEY,  -- e.g. 'cdm_512_z39_adapt'
    cosmology            VARCHAR,              -- 'CDM' or 'WDM'
    softening            VARCHAR,              -- 'fixed' or 'adaptive'
    n_part_per_side      INTEGER,              -- particles per side (512)
    z_ini                FLOAT,               -- initial redshift (39 or 99)
    box_size_mpc_h       FLOAT,               -- comoving box size in h^-1 Mpc
    particle_mass_msun_h FLOAT,               -- particle mass in h^-1 Msun
    raw_file_path        VARCHAR              -- absolute path to AHF halo catalogue directory
);

-- ---------------------------------------------------------------------------
-- bronze.ahf_halos
-- Raw AHF halo catalogue, one row per halo per simulation.
-- Column names are original AHF names; renamed to snake_case in silver.halos.
-- Gas and star columns (all zero in DM-only runs) are preserved here and
-- dropped during the silver transform.
-- Units: positions/radii in h^-1 kpc; masses in h^-1 Msun; velocities in km/s.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.ahf_halos (
    simulation_id        VARCHAR,
    halo_id              BIGINT,
    -- Structural
    hostHalo             BIGINT,   -- host halo ID; 0 = field halo (no parent)
    numSubStruct         BIGINT,   -- number of subhaloes
    Mhalo                DOUBLE,   -- halo mass [h^-1 Msun]
    npart                BIGINT,   -- number of particles
    -- Position [h^-1 kpc]
    Xc                   DOUBLE,
    Yc                   DOUBLE,
    Zc                   DOUBLE,
    -- Velocity [km/s]
    VXc                  DOUBLE,
    VYc                  DOUBLE,
    VZc                  DOUBLE,
    -- Radii [h^-1 kpc]
    Rhalo                DOUBLE,   -- halo radius
    Rmax                 DOUBLE,   -- radius of rotation curve maximum
    r2                   DOUBLE,   -- radius where rho*r^2 peaks (scale radius proxy)
    mbp_offset           DOUBLE,   -- most bound particle offset
    com_offset           DOUBLE,   -- centre of mass offset
    -- Kinematics [km/s]
    Vmax                 DOUBLE,   -- maximum circular velocity
    v_esc                DOUBLE,   -- escape velocity at Rhalo
    sigV                 DOUBLE,   -- 3D velocity dispersion
    -- Spin
    lambda               DOUBLE,   -- Bullock spin parameter
    lambdaE              DOUBLE,   -- Peebles spin parameter
    -- Angular momentum (unit vector components)
    Lx                   DOUBLE,
    Ly                   DOUBLE,
    Lz                   DOUBLE,
    -- Shape (axis ratios: b/a and c/a)
    b                    DOUBLE,
    c                    DOUBLE,
    -- Principal axis eigenvectors (moment of inertia tensor)
    Eax                  DOUBLE,   Eay DOUBLE, Eaz DOUBLE,  -- major axis
    Ebx                  DOUBLE,   Eby DOUBLE, Ebz DOUBLE,  -- intermediate axis
    Ecx                  DOUBLE,   Ecy DOUBLE, Ecz DOUBLE,  -- minor axis
    -- Thermodynamic / profile
    ovdens               DOUBLE,   -- overdensity at Rhalo
    nbins                BIGINT,   -- number of profile bins
    fMhires              DOUBLE,   -- high-resolution mass fraction (zoom sims)
    -- Energies [h^-1 Msun (km/s)^2]
    Ekin                 DOUBLE,
    Epot                 DOUBLE,
    SurfP                DOUBLE,   -- surface pressure term
    Phi0                 DOUBLE,   -- potential for unbinding [km^2/s^2]
    -- Concentration
    cNFW                 DOUBLE,
    -- Gas columns (all zero in DM-only runs; dropped in silver)
    n_gas                BIGINT,   M_gas DOUBLE,
    lambda_gas           DOUBLE,   lambdaE_gas DOUBLE,
    Lx_gas               DOUBLE,   Ly_gas DOUBLE,   Lz_gas DOUBLE,
    b_gas                DOUBLE,   c_gas DOUBLE,
    Eax_gas              DOUBLE,   Eay_gas DOUBLE,  Eaz_gas DOUBLE,
    Ebx_gas              DOUBLE,   Eby_gas DOUBLE,  Ebz_gas DOUBLE,
    Ecx_gas              DOUBLE,   Ecy_gas DOUBLE,  Ecz_gas DOUBLE,
    Ekin_gas             DOUBLE,   Epot_gas DOUBLE,
    -- Star columns (all zero in DM-only runs; dropped in silver)
    n_star               BIGINT,   M_star DOUBLE,
    lambda_star          DOUBLE,   lambdaE_star DOUBLE,
    Lx_star              DOUBLE,   Ly_star DOUBLE,   Lz_star DOUBLE,
    b_star               DOUBLE,   c_star DOUBLE,
    Eax_star             DOUBLE,   Eay_star DOUBLE,  Eaz_star DOUBLE,
    Ebx_star             DOUBLE,   Eby_star DOUBLE,  Ebz_star DOUBLE,
    Ecx_star             DOUBLE,   Ecy_star DOUBLE,  Ecz_star DOUBLE,
    Ekin_star            DOUBLE,   Epot_star DOUBLE,
    mean_z_gas           DOUBLE,   mean_z_star DOUBLE,
    n_star_excised       BIGINT,   M_star_excised DOUBLE, mean_z_star_excised DOUBLE,
    PRIMARY KEY (simulation_id, halo_id)
);

-- ---------------------------------------------------------------------------
-- bronze.protohalo_shapes
-- Half-maximum-mass protohalo sphericity, one row per halo with sufficient
-- merger tree history (≥10 snapshots, half-maximum-mass criterion).
-- ~25–44% of z=0 haloes per simulation have a matching record.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.protohalo_shapes (
    simulation_id VARCHAR,
    halo_id       BIGINT,
    sphericity_s  DOUBLE,  -- s = lambda_c / lambda_a (minor/major axis ratio at half-max mass)
    snap_hmm      BIGINT,  -- snapshot number at half-maximum mass
    a_hmm         DOUBLE,  -- scale factor at half-maximum mass
    m_hmm         DOUBLE,  -- half-maximum mass [h^-1 Msun]
    PRIMARY KEY (simulation_id, halo_id)
);

-- ---------------------------------------------------------------------------
-- bronze.wdm_cdm_matches
-- AHF MergerTree output: all (WDM halo, CDM halo) pairs with non-zero
-- shared particle count, one row per pair per crossmatch configuration.
-- Merit M = N^2_(A∩B) / (N_A * N_B); filtered to merit >= 0.1 in silver.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.wdm_cdm_matches (
    simulation_pair_id VARCHAR,  -- e.g. 'z39_adapt' (links to silver.simulation_pairs)
    wdm_halo_id        BIGINT,
    wdm_npart          BIGINT,   -- number of particles in WDM halo
    cdm_halo_id        BIGINT,
    cdm_npart          BIGINT,   -- number of particles in CDM halo
    n_shared           BIGINT,   -- number of shared particles
    merit              DOUBLE    -- merit score M = n_shared^2 / (wdm_npart * cdm_npart)
);
