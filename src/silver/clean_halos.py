"""
clean_halos.py: build silver.halos from bronze.ahf_halos.

Renames AHF columns to readable snake_case names, drops gas/star columns
(which are all zero in dark-matter-only runs), adds host/subhalo booleans and
particle-count flags, and joins simulation metadata from bronze.simulations.
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column rename map: bronze name -> silver name.
# Only columns that need renaming are listed, all others pass through as-is.
#
# Unit notes:
#   Positions (Xc/Yc/Zc), radii (Rhalo/Rmax/r2), offsets: h^-1 kpc
#   Masses (Mhalo): h^-1 Msun
#   Velocities (VXc/VYc/VZc/Vmax/v_esc/sigV): km s^-1
#   Energies (Ekin/Epot/SurfP/Phi0): h^-1 Msun (km s^-1)^2
#   b, c: dimensionless axis ratios b/a and c/a (values confirmed 0-1)
#   Eax..Ecz: dimensionless unit vectors along principal axes
# ---------------------------------------------------------------------------
_COLUMN_RENAMES: dict[str, str] = {
    "hostHalo": "host_halo_id",  # ID of the host halo; 0 = no host halo (field host)
    "numSubStruct": "n_substructure",  # number of subhaloes in the halo
    "Mhalo": "m_halo_msun_h",  # halo mass in h^-1 Msun
    "npart": "n_particles",  # number of particles in the halo
    "Xc": "x_kpc_h",  # x position in h^-1 kpc
    "Yc": "y_kpc_h",  # y position in h^-1 kpc
    "Zc": "z_kpc_h",  # z position in h^-1 kpc
    "VXc": "vx_km_s",  # x peculiar velocity in km s^-1
    "VYc": "vy_km_s",  # y peculiar velocity in km s^-1
    "VZc": "vz_km_s",  # z peculiar velocity in km s^-1
    "Rhalo": "r_halo_kpc_h",  # halo radius in h^-1 kpc
    "Rmax": "r_max_kpc_h",  # position of rotation curve maximum in h^-1 kpc
    "r2": "r_scale_kpc_h",  # radius where \rho*r^2 peaks
    "mbp_offset": "mbp_offset_kpc_h",  # most bound particle offset in h^-1 kpc
    "com_offset": "com_offset_kpc_h",  # center of mass offset in h^-1 kpc
    "Vmax": "v_max_km_s",  # maximum of rotation curve in km s^-1
    "v_esc": "v_esc_km_s",  # escape velocity at Rhalo in km s^-1
    "sigV": "sigma_v_km_s",  # 3D velocity dispersion in km s^-1
    "lambda": "spin_bullock",  # Bullock spin parameter
    "lambdaE": "spin_peebles",  # Peebles spin parameter
    "Lx": "l_x",  # normalised angular momentum (x component)
    "Ly": "l_y",  # normalised angular momentum (y component)
    "Lz": "l_z",  # normalised angular momentum (z component)
    "b": "axis_ratio_b",  # intermediate axis / major axis (b/a)
    "c": "axis_ratio_c",  # minor axis / major axis (c/a)
    "Eax": "ea_x",  # largest eigenvec moment of inertia tensor (x component)
    "Eay": "ea_y",  # largest eigenvec moment of inertia tensor (y component)
    "Eaz": "ea_z",  # largest eigenvec moment of inertia tensor (z component)
    "Ebx": "eb_x",  # intermediate eigenvec moment of inertia tensor (x component)
    "Eby": "eb_y",  # intermediate eigenvec moment of inertia tensor (y component)
    "Ebz": "eb_z",  # intermediate eigenvec moment of inertia tensor (z component)
    "Ecx": "ec_x",  # minor eigenvec moment of inertia tensor (x component)
    "Ecy": "ec_y",  # minor eigenvec moment of inertia tensor (y component)
    "Ecz": "ec_z",  # minor eigenvec moment of inertia tensor (z component)
    "ovdens": "overdensity",  # overdensity at Rhalo
    "nbins": "n_bins",  # number of bins used in *AHF_profiles calculation
    "fMhires": "f_mass_highres",  # mass fraction in high resolution particles for zoom sims
    "Ekin": "e_kin_msun_km2_s2_h",  # kinetic energy in h^-1 Msun km^2 s^-2 (halo frame)
    "Epot": "e_pot_msun_km2_s2_h",  # potential energy in h^-1 Msun km^2 s^-2 (halo frame)
    "SurfP": "surf_pressure_msun_km2_s2_h",  # surface pressure in h^-1 Msun km^2 s^-2
    "Phi0": "phi0_km2_s2",  # potential for unbinding procedure in km^2 s^-2
    "cNFW": "c_nfw",  # NFW concentration parameter
}

# Gas and star columns are all zero in dark-matter-only runs.
# We drop them here rather than carry 45 dead columns into gold.
_GAS_STAR_COLUMNS = (
    "n_gas",
    "M_gas",
    "lambda_gas",
    "lambdaE_gas",
    "Lx_gas",
    "Ly_gas",
    "Lz_gas",
    "b_gas",
    "c_gas",
    "Eax_gas",
    "Eay_gas",
    "Eaz_gas",
    "Ebx_gas",
    "Eby_gas",
    "Ebz_gas",
    "Ecx_gas",
    "Ecy_gas",
    "Ecz_gas",
    "Ekin_gas",
    "Epot_gas",
    "n_star",
    "M_star",
    "lambda_star",
    "lambdaE_star",
    "Lx_star",
    "Ly_star",
    "Lz_star",
    "b_star",
    "c_star",
    "Eax_star",
    "Eay_star",
    "Eaz_star",
    "Ebx_star",
    "Eby_star",
    "Ebz_star",
    "Ecx_star",
    "Ecy_star",
    "Ecz_star",
    "Ekin_star",
    "Epot_star",
    "mean_z_gas",
    "mean_z_star",
    "n_star_excised",
    "M_star_excised",
    "mean_z_star_excised",
)

# Simulation metadata columns joined in from bronze.simulations.
_SIM_META_COLUMNS = [
    "cosmology",
    "softening",
    "n_part_per_side",
    "z_ini",
    "box_size_mpc_h",
    "particle_mass_msun_h",
]


def clean_halos(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Build silver.halos from bronze.ahf_halos and bronze.simulations.

    Writes results to silver.halos, replacing any existing table.

    Parameters
    ----------
    conn:
        Open DuckDB connection with bronze schema populated.
    """
    log.info("Building silver.halos ...")

    halos: pl.DataFrame = conn.execute("SELECT * FROM bronze.ahf_halos").pl()
    sims: pl.DataFrame = conn.execute(
        f"SELECT simulation_id, {', '.join(_SIM_META_COLUMNS)} FROM bronze.simulations"
    ).pl()

    # Drop gas/star columns, which are all zero in DM-only runs.
    drop_cols = [col for col in halos.columns if col in _GAS_STAR_COLUMNS]
    halos = halos.drop(drop_cols)

    # Rename AHF columns to readable names and skip any absent in this dataset
    # to guard against minor schema differences between AHF versions.
    rename_map = {key: val for key, val in _COLUMN_RENAMES.items() if key in halos.columns}
    halos = halos.rename(rename_map)

    # Derive host/subhalo booleans.
    # host_halo_id == 0 means the halo has no host and is a field host itself.
    halos = halos.with_columns(
        (pl.col("host_halo_id") == 0).alias("is_host"),
        (pl.col("host_halo_id") != 0).alias("is_subhalo"),
    )

    # Derive particle-count flag used for quality filtering downstream.
    halos = halos.with_columns(
        (pl.col("n_particles") < 100).alias("below_100_part_limit"),
    )

    # Join simulation metadata
    # Every halo row has a simulation_id so this should not introduce nulls
    # on the right-hand side.
    halos = halos.join(sims, on="simulation_id", how="left")

    _ = conn.execute("CREATE SCHEMA IF NOT EXISTS silver")
    _ = conn.register("_silver_halos", halos)
    _ = conn.execute("CREATE OR REPLACE TABLE silver.halos AS SELECT * FROM _silver_halos")
    _ = conn.unregister("_silver_halos")

    log.info("silver.halos: %d rows written", len(halos))
