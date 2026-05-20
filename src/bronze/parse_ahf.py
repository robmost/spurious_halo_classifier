"""
parse_ahf.py: parser for AHF halo catalogue files (.AHF_halos).

Reads one merged .AHF_halos file per simulation directory and returns a
Polars DataFrame with all the 88 columns + a simulation_id.

The filtering or unit conversion gets done by the silver layer.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Column name mapping
# ---------------------------------------------------------------------------

# AHF headers look like "ID(1)\thostHalo(2)\t...". Strip the "(N)" suffix so
# downstream SQL and Python code uses clean names.
# Regex pattern to match the "(N)" suffix: \(\d+\)$
_INDEX_SUFFIX = re.compile(r"\(\d+\)$")


def _clean_column_name(raw: str) -> str:
    """
    Strip the trailing column-index suffix from an AHF header token.

    Examples
    --------
    >>> _clean_column_name("ID(1)")
    'halo_id'
    >>> _clean_column_name("hostHalo(2)")
    'hostHalo'
    """
    name = _INDEX_SUFFIX.sub("", raw).strip()
    # Rename ID -> halo_id to avoid collision with SQL reserved word.
    if name == "ID":
        return "halo_id"
    return name


# Columns that should be stored as integers. Everything else is float.
_INT_COLUMNS = frozenset(
    {
        "halo_id",
        "hostHalo",
        "numSubStruct",
        "npart",
        "nbins",
        "n_gas",
        "n_star",
        "n_star_excised",
    }
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def parse_ahf_halos(ahf_halos_dir: Path, simulation_id: str) -> pl.DataFrame:
    """
    Parse the merged .AHF_halos file in ahf_halos_dir.

    Parameters
    ----------
    ahf_halos_dir:
        Directory containing exactly one .AHF_halos file.
    simulation_id:
        Identifier string attached as a column to every row.

    Returns
    -------
    pl.DataFrame
        88 data columns (with cleaned names) + a simulation_id. All rows from
        the file are returned. No particle-count filtering gets done at bronze.

    Raises
    ------
    FileNotFoundError
        If no .AHF_halos file exists in ahf_halos_dir.
    ValueError
        If more than one .AHF_halos file is found, or if the parsed column
        count does not match the header column count.
    """
    ahf_file = _find_ahf_file(ahf_halos_dir)
    column_names = _parse_header(ahf_file)
    schema = _build_schema(column_names)

    df = pl.read_csv(
        ahf_file,
        separator="\t",
        comment_prefix="#",
        has_header=False,
        new_columns=column_names,
        schema_overrides=schema,
        # A trailing tab in some rows produces an empty last token, so we ignore it.
        truncate_ragged_lines=True,
    )

    if df.width != len(column_names):
        raise ValueError(
            f"Expected {len(column_names)} columns from header but parsed "
            + f"{df.width} in '{ahf_file}'"
        )

    # Attach simulation identifier as the first column.
    return df.with_columns(pl.lit(simulation_id).alias("simulation_id")).select(
        ["simulation_id"] + column_names
    )


# ---------------------------------------------------------------------------
# Internal (private) helpers
# ---------------------------------------------------------------------------


def _find_ahf_file(directory: Path) -> Path:
    """
    Return the single .AHF_halos file in directory.
    """
    matches = list(directory.glob("*.AHF_halos"))
    if not matches:
        raise FileNotFoundError(f"No .AHF_halos file found in '{directory}'")
    if len(matches) > 1:
        raise ValueError(
            f"Expected one .AHF_halos file in '{directory}', found {len(matches)}: "
            + ", ".join(m.name for m in matches)
        )
    return matches[0]


def _parse_header(ahf_file: Path) -> list[str]:
    """
    Read the '#'-prefixed header line and return clean column names.
    """
    with ahf_file.open() as f:
        for line in f:
            if line.startswith("#"):
                raw_names = line.lstrip("#").strip().split("\t")
                return [
                    _clean_column_name(tok)
                    for tok in raw_names
                    if tok.strip()  # skip empty tokens from trailing tabs
                ]
    raise ValueError(f"No header line found in '{ahf_file}'")


def _build_schema(column_names: list[str]) -> dict[str, pl.DataType]:
    """
    Return a complete Polars dtype map for all columns.

    To prevent Polars from inferring String for AHF whitespace-padded
    numeric values (e.g. ' 7.13775e+12'), a full schema
    (not just overrides for int columns) is required.
    """
    return {col: pl.Int64() if col in _INT_COLUMNS else pl.Float64() for col in column_names}
