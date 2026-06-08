"""
parse_protohalos.py: parser for SPHERICITY_ALL_INFO.*.hdf5 protohalo shape files.

Each file is a pandas HDF5 store written with pd.DataFrame.to_hdf() in
'fixed' format. It contains one row per z=0 halo that survived the
10-snapshot and half-maximum-mass selection criteria from Mostoghiu Paun
et al. (2025).

NOTE: Not every z=0 halo has a record, so the has_protohalo_data
flag in silver marks this gap.

We extract three quantities and discard the rest:
  - halo_id      : AHF halo ID (from the DataFrame index)
  - sphericity_s : axis ratio s = lambda_c / lambda_a of the Lagrangian volume
  - snap_hmm     : snapshot number at half-maximum mass (Int64)
  - a_hmm        : scale factor at half-maximum mass, computed as
                   1 / (1 + z) from the redshift encoded in the raw
                   string column `halfmaxmass` (e.g. "snap_056-z0.000")
  - m_hmm        : halo mass at half-maximum mass (h^-1 M_sun)

The `halfmaxmass` column encodes both snapshot number and redshift as a
composite string. It is parsed into `snap_hmm` (Int64) and `a_hmm` (Float64)
at bronze level so that all downstream layers see clean typed columns.

NOTE: The partid_parttype_halfmaxmass column (serialised particle ID arrays,
~700 MB uint8 blob) is loaded but immediately discarded. They are not needed,
as shapes are already computed, and particle IDs are not ML features.
Column selection is not possible because the store uses 'fixed' (not 'table') format.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import cast

import pandas as pd
import polars as pl
from pandas import DataFrame

# ---------------------------------------------------------------------------
# HDF5 key and column mapping
# ---------------------------------------------------------------------------

_HDF_KEY = "sphericity"

# Maps HDF5 column names to intermediate names before type-specific parsing.
# halfmaxmass is kept as a raw string column and parsed into snap_hmm and
# a_hmm after conversion to Polars. It is not the final column name.
_COLUMN_MAP: dict[str, str] = {
    "sphericity_halfmaxmass": "sphericity_s",
    "halfmaxmass": "halfmaxmass_raw",  # composite string, parsed below
    "mhalo_halfmaxmass": "m_hmm",
}

_REQUIRED_COLUMNS: set[str] = set(_COLUMN_MAP.keys())

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def parse_protohalos(sphericity_path: Path, simulation_id: str) -> pl.DataFrame:
    """
    Parse the SPHERICITY_ALL_INFO.*.hdf5 file at sphericity_path.

    Parameters
    ----------
    sphericity_path:
        Path to the SPHERICITY_ALL_INFO.*.hdf5 file.
    simulation_id:
        Identifier string attached as a column to every row.

    Returns
    -------
    pl.DataFrame
        One row per protohalo with columns:
        simulation_id, halo_id, sphericity_s, snap_hmm, a_hmm, m_hmm.

    Raises
    ------
    FileNotFoundError
        If sphericity_path does not exist.
    ValueError
        If expected columns are missing from the HDF5 store.
    """
    if not sphericity_path.exists():
        raise FileNotFoundError(f"Sphericity file not found: '{sphericity_path}'")

    # The store is fixed format, so column selection is not supported. Read the full DataFrame.
    # NOTE: The partid_parttype_halfmaxmass object column triggers a
    # PyTables PerformanceWarning about pickling, which is expected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        df_pd = cast(DataFrame, pd.read_hdf(sphericity_path, key=_HDF_KEY))

    missing = _REQUIRED_COLUMNS - set(df_pd.columns)
    if missing:
        raise ValueError(f"Expected columns missing from '{sphericity_path}': {missing}")

    # The DataFrame index holds the AHF halo IDs.
    # Add it as a column and rename it to 'halo_id', then rename the other columns.
    df_pd.index.name = "halo_id"
    df_pd = df_pd.reset_index()[["halo_id"] + list(_COLUMN_MAP.keys())]
    df_pd = df_pd.rename(columns=_COLUMN_MAP)  # pyright: ignore[reportCallIssue]

    return (
        pl.from_pandas(df_pd)
        .with_columns(pl.lit(simulation_id).alias("simulation_id"))
        .with_columns(_parse_halfmaxmass())
        .drop("halfmaxmass_raw")
        .select(["simulation_id", "halo_id", "sphericity_s", "snap_hmm", "a_hmm", "m_hmm"])
    )


# ---------------------------------------------------------------------------
# Internal (private) helpers
# ---------------------------------------------------------------------------


def _parse_halfmaxmass() -> list[pl.Expr]:
    """Parse 'snap_056-z0.000' into snap_hmm (Int64) and a_hmm = 1/(1+z) (Float64)."""
    # Split on "-": ["snap_056", "z0.000"]
    parts = pl.col("halfmaxmass_raw").str.split("-")

    snap_expr = (
        parts.list.get(0)  # "snap_056"
        .str.strip_prefix("snap_")  # "056"
        .cast(pl.Int64)
        .alias("snap_hmm")
    )

    # Scale factor a = 1 / (1 + z); redshift is the numeric part after "z".
    z_expr = (
        parts.list.get(1)  # "z0.000"
        .str.strip_prefix("z")  # "0.000"
        .cast(pl.Float64)
    )
    a_expr = (1.0 / (1.0 + z_expr)).alias("a_hmm")

    return [snap_expr, a_expr]
