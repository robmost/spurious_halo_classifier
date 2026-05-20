"""
Tests for src/bronze/parse_ahf.py.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.bronze.parse_ahf import (
    _build_schema,  # pyright: ignore[reportPrivateUsage]
    _clean_column_name,  # pyright: ignore[reportPrivateUsage]
    _find_ahf_file,  # pyright: ignore[reportPrivateUsage]
    _parse_header,  # pyright: ignore[reportPrivateUsage]
    parse_ahf_halos,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCleanColumnName:
    def test_strips_numeric_index_suffix(self) -> None:
        assert _clean_column_name("Mvir(4)") == "Mvir"
        assert _clean_column_name("hostHalo(2)") == "hostHalo"

    def test_renames_id_to_halo_id(self) -> None:
        assert _clean_column_name("ID(1)") == "halo_id"

    def test_no_suffix_returned_unchanged(self) -> None:
        assert _clean_column_name("sigV") == "sigV"

    def test_strips_trailing_whitespace_after_suffix_removal(self) -> None:
        assert _clean_column_name("Mvir(4) ") == "Mvir(4)"  # trailing space, regex $-anchored


class TestBuildSchema:
    def test_integer_columns_get_int64(self) -> None:
        schema = _build_schema(["halo_id", "npart"])
        assert schema["halo_id"] == pl.Int64()
        assert schema["npart"] == pl.Int64()

    def test_other_columns_get_float64(self) -> None:
        schema = _build_schema(["Mvir", "Xc", "sigV"])
        for col in ("Mvir", "Xc", "sigV"):
            assert schema[col] == pl.Float64()

    def test_all_known_int_columns_classified(self) -> None:
        int_cols = [
            "halo_id",
            "hostHalo",
            "numSubStruct",
            "npart",
            "nbins",
            "n_gas",
            "n_star",
            "n_star_excised",
        ]
        schema = _build_schema(int_cols)
        for col in int_cols:
            assert schema[col] == pl.Int64(), f"Expected Int64 for '{col}'"


class TestFindAhfFile:
    def test_returns_single_file(self, tmp_path: Path) -> None:
        ahf = tmp_path / "sim.AHF_halos"
        ahf.touch()
        assert _find_ahf_file(tmp_path) == ahf

    def test_raises_when_no_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"\.AHF_halos"):
            _ = _find_ahf_file(tmp_path)

    def test_raises_when_multiple_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.AHF_halos").touch()
        (tmp_path / "b.AHF_halos").touch()
        with pytest.raises(ValueError, match="Expected one"):
            _ = _find_ahf_file(tmp_path)


class TestParseHeader:
    def test_returns_clean_column_names(self, tmp_path: Path) -> None:
        ahf = tmp_path / "sim.AHF_halos"
        _ = ahf.write_text("#ID(1)\thostHalo(2)\tMvir(3)\n1\t0\t1.0e12\n")
        names = _parse_header(ahf)
        assert names == ["halo_id", "hostHalo", "Mvir"]

    def test_raises_when_no_header_line(self, tmp_path: Path) -> None:
        ahf = tmp_path / "sim.AHF_halos"
        _ = ahf.write_text("1\t0\t1.0e12\n")
        with pytest.raises(ValueError, match="No header line"):
            _ = _parse_header(ahf)


class TestParseAhfHalos:
    def test_returns_correct_row_count(self) -> None:
        df = parse_ahf_halos(FIXTURES_DIR, "test_sim")
        assert len(df) == 2

    def test_simulation_id_is_first_column(self) -> None:
        df = parse_ahf_halos(FIXTURES_DIR, "test_sim")
        assert df.columns[0] == "simulation_id"
        assert df.columns[1] == "halo_id"

    def test_simulation_id_value_attached(self) -> None:
        df = parse_ahf_halos(FIXTURES_DIR, "test_sim")
        assert df["simulation_id"].unique().to_list() == ["test_sim"]

    def test_integer_columns_have_int64_dtype(self) -> None:
        df = parse_ahf_halos(FIXTURES_DIR, "test_sim")
        assert df["halo_id"].dtype == pl.Int64
        assert df["npart"].dtype == pl.Int64

    def test_float_columns_have_float64_dtype(self) -> None:
        df = parse_ahf_halos(FIXTURES_DIR, "test_sim")
        assert df["Mvir"].dtype == pl.Float64
        assert df["Xc"].dtype == pl.Float64

    def test_raises_when_directory_has_no_ahf_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _ = parse_ahf_halos(tmp_path, "sim")
