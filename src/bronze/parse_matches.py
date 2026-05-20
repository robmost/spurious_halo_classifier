"""
parse_matches.py: parser for WDM to CDM MergerTree cross-correlation files.

Reads one *_croco file per setup directory from WDM_CDM_crossmatch/.
Each file lists WDM haloes as primaries (unindented lines) and their CDM
counterparts as children (indented lines). WDM haloes absent from the file
altogether share zero particles with any CDM halo.

Output is one row per (WDM halo, CDM halo) pair, returning all pairs regardless of
merit. Merit filtering and the has_cdm_counterpart flag are computed in
silver, not here.

File format
-----------
# header lines (prefixed with #)
<wdm_halo_id>  <wdm_npart>  <n_progenitors>       ← primary line (no indent)
  <n_shared>  <cdm_halo_id>  <cdm_npart>  <merit>  ← child line (2-space indent)
  ...
<next_wdm_halo_id>  ...
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def parse_matches(wdm_cdm_dir: Path, simulation_pair_id: str) -> pl.DataFrame:
    """
    Parse the *_croco file in wdm_cdm_dir.

    Parameters
    ----------
    wdm_cdm_dir:
        Directory containing exactly one *_croco file
        (e.g. WDM_CDM_crossmatch/z39_adapt/).
    simulation_pair_id:
        Identifier for the WDM-CDM simulation pair, e.g. 'z39_adapt'.
        Attached as a column to every row.

    Returns
    -------
    pl.DataFrame
        One row per (WDM halo, CDM halo) pair with columns:
        simulation_pair_id, wdm_halo_id, wdm_npart, cdm_halo_id,
        cdm_npart, n_shared, merit.

        WDM haloes with no CDM counterpart at all are absent from the output, the silver
        layer handles them via a left join against silver.halos.

    Raises
    ------
    FileNotFoundError
        If no *_croco file exists in wdm_cdm_dir.
    ValueError
        If more than one *_croco file is found.
    """
    croco_file = _find_croco_file(wdm_cdm_dir)
    rows = _parse_croco(croco_file)

    # Return an empty frame if no rows were parsed
    if not rows:
        return _empty_frame()

    df = (
        pl.DataFrame(
            rows,
            schema={
                "wdm_halo_id": pl.Int64,
                "wdm_npart": pl.Int64,
                "cdm_halo_id": pl.Int64,
                "cdm_npart": pl.Int64,
                "n_shared": pl.Int64,
                "merit": pl.Float64,
            },
        )
        .with_columns(pl.lit(simulation_pair_id).alias("simulation_pair_id"))
        .select(
            [
                "simulation_pair_id",
                "wdm_halo_id",
                "wdm_npart",
                "cdm_halo_id",
                "cdm_npart",
                "n_shared",
                "merit",
            ]
        )
    )

    return df


# ---------------------------------------------------------------------------
# Internal (private) helpers
# ---------------------------------------------------------------------------


def _find_croco_file(directory: Path) -> Path:
    """
    Return the single *_croco file in directory.
    """
    matches = list(directory.glob("*_croco"))
    if not matches:
        raise FileNotFoundError(f"No *_croco file found in '{directory}'")
    if len(matches) > 1:
        raise ValueError(
            f"Expected one *_croco file in '{directory}', found {len(matches)}: "
            + ", ".join(m.name for m in matches)
        )
    return matches[0]


def _parse_croco(croco_file: Path) -> list[dict[str, int | float | None]]:
    """
    Parse the two-level text format of a _croco file.

    Returns a flat list of dicts, one per (WDM primary, CDM child) pair.
    WDM primaries with no children are excluded, MergerTree does not write
    primaries with zero children to the file.
    """
    rows: list[dict[str, int | float | None]] = []
    current_wdm_id: int | None = None
    current_wdm_npart: int | None = None

    with croco_file.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            if line.startswith("  "):
                # Child line: n_shared  cdm_halo_id  cdm_npart  merit
                if current_wdm_id is None:
                    continue
                parts = line.split()
                rows.append(
                    {
                        "wdm_halo_id": current_wdm_id,
                        "wdm_npart": current_wdm_npart,
                        "cdm_halo_id": int(parts[1]),
                        "cdm_npart": int(parts[2]),
                        "n_shared": int(parts[0]),
                        "merit": float(parts[3]),
                    }
                )
            else:
                # Primary line: wdm_halo_id  wdm_npart  n_progenitors
                parts = line.split()
                current_wdm_id = int(parts[0])
                current_wdm_npart = int(parts[1])

    return rows


def _empty_frame() -> pl.DataFrame:
    """
    Return a correctly typed empty DataFrame for a directory with no matches.
    """
    return pl.DataFrame(
        schema={
            "simulation_pair_id": pl.String,
            "wdm_halo_id": pl.Int64,
            "wdm_npart": pl.Int64,
            "cdm_halo_id": pl.Int64,
            "cdm_npart": pl.Int64,
            "n_shared": pl.Int64,
            "merit": pl.Float64,
        }
    )
