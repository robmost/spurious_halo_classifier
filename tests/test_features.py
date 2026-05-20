"""
Tests for src/gold/features.py.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.gold.features import (
    FEATURE_COLS,
    _compute_ahf_features,  # pyright: ignore[reportPrivateUsage]
    _compute_spatial_features,  # pyright: ignore[reportPrivateUsage]
)


def _minimal_proto_df(**overrides: object) -> pl.DataFrame:
    """Two-row silver.protohalos-like DataFrame for AHF feature tests."""
    defaults: dict[str, object] = {
        "halo_id": [1, 2],
        "simulation_id": ["sim_a", "sim_a"],
        "m_halo_msun_h": [1e10, 1e11],
        "n_particles": [100, 1000],
        "x_kpc_h": [5000.0, 10000.0],
        "y_kpc_h": [5000.0, 10000.0],
        "z_kpc_h": [5000.0, 10000.0],
        "box_size_mpc_h": [25.0, 25.0],
        "sigma_v_km_s": [100.0, 200.0],
        "c_nfw": [10.0, 12.0],
        "spin_bullock": [0.03, 0.05],
        "axis_ratio_b": [0.8, 0.7],
        "axis_ratio_c": [0.6, 0.5],
        "sphericity_s": [0.5, None],
        "m_hmm": [1e9, None],
        "a_hmm": [0.5, None],
    }
    defaults.update(overrides)
    return pl.DataFrame(defaults)


class TestComputeAhfFeatures:
    def test_log10_m200_correct(self) -> None:
        df = _compute_ahf_features(_minimal_proto_df())
        assert df["log10_m200"].to_list() == pytest.approx([10.0, 11.0])

    def test_log10_npart_correct(self) -> None:
        df = _compute_ahf_features(_minimal_proto_df())
        assert df["log10_npart"].to_list() == pytest.approx([2.0, 3.0])

    def test_position_normalisation(self) -> None:
        # x_kpc_h=5000, box_size_mpc_h=25 → 5000 / (25 × 1000) = 0.2
        df = _compute_ahf_features(_minimal_proto_df())
        assert df["x_norm"][0] == pytest.approx(0.2)
        assert df["y_norm"][0] == pytest.approx(0.2)
        assert df["z_norm"][0] == pytest.approx(0.2)

    def test_v_disp_sigv_alias(self) -> None:
        df = _compute_ahf_features(_minimal_proto_df())
        assert "v_disp_sigv" in df.columns
        assert df["v_disp_sigv"].to_list() == [100.0, 200.0]

    def test_axis_ratio_aliases_renamed(self) -> None:
        df = _compute_ahf_features(_minimal_proto_df())
        assert "axis_ratio_b_over_a" in df.columns
        assert "axis_ratio_c_over_a" in df.columns
        assert df["axis_ratio_b_over_a"].to_list() == [0.8, 0.7]
        assert df["axis_ratio_c_over_a"].to_list() == [0.6, 0.5]

    def test_log10_m_hmm_null_propagated(self) -> None:
        df = _compute_ahf_features(_minimal_proto_df())
        assert df["log10_m_hmm"][0] == pytest.approx(9.0)
        assert df["log10_m_hmm"][1] is None

    def test_sphericity_passed_through(self) -> None:
        df = _compute_ahf_features(_minimal_proto_df())
        assert df["sphericity_s"][0] == pytest.approx(0.5)
        assert df["sphericity_s"][1] is None

    def test_a_hmm_passed_through(self) -> None:
        df = _compute_ahf_features(_minimal_proto_df())
        assert df["a_hmm"][0] == pytest.approx(0.5)
        assert df["a_hmm"][1] is None


class TestComputeSpatialFeatures:
    def test_returns_expected_columns(self) -> None:
        result = _compute_spatial_features(_minimal_proto_df(), neighbour_radius_mpc_h=5.0)
        assert {
            "halo_id",
            "simulation_id",
            "dist_nearest_neighbour_mpc_h",
            "count_neighbours_within_radius_mpc_h",
        }.issubset(set(result.columns))

    def test_one_row_per_halo(self) -> None:
        result = _compute_spatial_features(_minimal_proto_df(), neighbour_radius_mpc_h=5.0)
        assert len(result) == 2

    def test_nearest_neighbour_distance_positive(self) -> None:
        result = _compute_spatial_features(_minimal_proto_df(), neighbour_radius_mpc_h=5.0)
        assert (result["dist_nearest_neighbour_mpc_h"] > 0).all()

    def test_no_cross_simulation_contamination(self) -> None:
        """
        Each simulation must get its own KD-tree.
        Inter-sim distances are never computed.
        """
        df = pl.DataFrame(
            {
                "halo_id": [1, 2, 3, 4],
                "simulation_id": ["sim_a", "sim_a", "sim_b", "sim_b"],
                "x_kpc_h": [1000.0, 2000.0, 10000.0, 11000.0],
                "y_kpc_h": [0.0, 0.0, 0.0, 0.0],
                "z_kpc_h": [0.0, 0.0, 0.0, 0.0],
                "box_size_mpc_h": [25.0, 25.0, 25.0, 25.0],
            }
        )
        result = _compute_spatial_features(df, neighbour_radius_mpc_h=0.5)
        assert sorted(result["simulation_id"].unique().to_list()) == ["sim_a", "sim_b"]

    def test_neighbour_count_zero_for_small_radius(self) -> None:
        """
        With a search radius smaller than the halo separation no neighbours are found.
        """
        result = _compute_spatial_features(_minimal_proto_df(), neighbour_radius_mpc_h=0.001)
        assert (result["count_neighbours_within_radius_mpc_h"] == 0).all()


class TestFeatureCols:
    def test_fifteen_features_defined(self) -> None:
        assert len(FEATURE_COLS) == 15

    def test_key_feature_names_present(self) -> None:
        assert {"log10_m200", "sphericity_s", "dist_nearest_neighbour_mpc_h"}.issubset(
            set(FEATURE_COLS)
        )

    def test_no_duplicate_names(self) -> None:
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
