"""
Tests for the _parse_crossmatch_id helper in src/config.py.
"""

from __future__ import annotations

import pytest

from src.config import _parse_crossmatch_id  # pyright: ignore[reportPrivateUsage]


class TestParseCrossmatchId:
    def test_z39_adapt_returns_adaptive(self) -> None:
        z, s = _parse_crossmatch_id("z39_adapt")
        assert z == 39
        assert s == "adaptive"

    def test_z99_fixed_returns_fixed(self) -> None:
        z, s = _parse_crossmatch_id("z99_fixed")
        assert z == 99
        assert s == "fixed"

    def test_no_underscore_raises(self) -> None:
        with pytest.raises(ValueError, match="pattern"):
            _ = _parse_crossmatch_id("z39")

    def test_no_z_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="pattern"):
            _ = _parse_crossmatch_id("39_fixed")

    def test_non_integer_z_raises(self) -> None:
        with pytest.raises(ValueError, match="z_ini"):
            _ = _parse_crossmatch_id("zabc_fixed")

    def test_unknown_softening_alias_raises(self) -> None:
        with pytest.raises(ValueError, match="softening"):
            _ = _parse_crossmatch_id("z39_parallel")
