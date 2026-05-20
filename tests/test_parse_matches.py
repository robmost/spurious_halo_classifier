"""
Tests for src/bronze/parse_matches.py.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.bronze.parse_matches import (
    _empty_frame,  # pyright: ignore[reportPrivateUsage]
    _find_croco_file,  # pyright: ignore[reportPrivateUsage]
    _parse_croco,  # pyright: ignore[reportPrivateUsage]
    parse_matches,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestFindCrocoFile:
    def test_returns_single_file(self, tmp_path: Path) -> None:
        croco = tmp_path / "sim_croco"
        croco.touch()
        assert _find_croco_file(tmp_path) == croco

    def test_raises_when_no_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="_croco"):
            _ = _find_croco_file(tmp_path)

    def test_raises_when_multiple_files(self, tmp_path: Path) -> None:
        (tmp_path / "a_croco").touch()
        (tmp_path / "b_croco").touch()
        with pytest.raises(ValueError, match="Expected one"):
            _ = _find_croco_file(tmp_path)


class TestParseCroco:
    def test_correct_row_count(self) -> None:
        # sample_croco has 2 WDM primaries: 2 children + 1 child = 3 rows
        rows = _parse_croco(FIXTURES_DIR / "sample_croco")
        assert len(rows) == 3

    def test_wdm_halo_ids_correct(self) -> None:
        rows = _parse_croco(FIXTURES_DIR / "sample_croco")
        assert {r["wdm_halo_id"] for r in rows} == {100, 200}

    def test_merit_values_are_float(self) -> None:
        rows = _parse_croco(FIXTURES_DIR / "sample_croco")
        for row in rows:
            assert isinstance(row["merit"], float)

    def test_n_shared_values_are_int(self) -> None:
        rows = _parse_croco(FIXTURES_DIR / "sample_croco")
        for row in rows:
            assert isinstance(row["n_shared"], int)

    def test_skips_comment_lines(self, tmp_path: Path) -> None:
        croco = tmp_path / "t_croco"
        _ = croco.write_text("# header\n100  500  1\n  250  200  400  0.3\n")
        rows = _parse_croco(croco)
        assert len(rows) == 1

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        croco = tmp_path / "t_croco"
        _ = croco.write_text("\n100  500  1\n  250  200  400  0.3\n\n")
        rows = _parse_croco(croco)
        assert len(rows) == 1


class TestEmptyFrame:
    def test_has_correct_schema(self) -> None:
        df = _empty_frame()
        assert df.schema["simulation_pair_id"] == pl.String
        assert df.schema["wdm_halo_id"] == pl.Int64
        assert df.schema["merit"] == pl.Float64

    def test_has_zero_rows(self) -> None:
        assert len(_empty_frame()) == 0


class TestParseMatches:
    def test_returns_correct_row_count(self) -> None:
        df = parse_matches(FIXTURES_DIR, "z39_fixed")
        assert len(df) == 3

    def test_simulation_pair_id_is_first_column(self) -> None:
        df = parse_matches(FIXTURES_DIR, "z39_fixed")
        assert df.columns[0] == "simulation_pair_id"

    def test_simulation_pair_id_attached(self) -> None:
        df = parse_matches(FIXTURES_DIR, "z39_fixed")
        assert df["simulation_pair_id"].unique().to_list() == ["z39_fixed"]

    def test_merit_values_parsed_correctly(self) -> None:
        df = parse_matches(FIXTURES_DIR, "z39_fixed")
        assert sorted(df["merit"].to_list()) == pytest.approx(sorted([0.35, 0.15, 0.42]))

    def test_raises_when_no_croco_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _ = parse_matches(tmp_path, "z39_fixed")
