"""
Tests for src/config.py.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.config import AppConfig, load_config

# ---------------------------------------------------------------------------
# Minimal valid YAML used as the base for all config tests.
# Tests that need to deviate from the happy path make targeted string
# replacements on this template.
# ---------------------------------------------------------------------------

_BASE_YAML = textwrap.dedent("""\
    database:
      path: data/test.duckdb

    simulations:
      - id: cdm_sim
        cosmology: CDM
        softening: fixed
        n_part_per_side: 512
        z_ini: 39
        box_size_mpc_h: 25.0
        particle_mass_msun_h: 1.0e8
        ahf_halos_dir: data/raw
        sphericity_path: data/raw/s.hdf5
      - id: wdm_sim
        cosmology: WDM
        softening: fixed
        n_part_per_side: 512
        z_ini: 39
        box_size_mpc_h: 25.0
        particle_mass_msun_h: 1.0e8
        ahf_halos_dir: data/raw
        sphericity_path: data/raw/s.hdf5

    crossmatch:
      configurations:
        - id: z39_fixed
          path: data/raw/WDM_CDM_crossmatch/z39_fixed

    silver:
      merit_threshold: 0.1

    gold:
      empirical_label:
        alpha: 0.0624
        beta: -0.0988
        mass_threshold_log10: 10.0
      spatial_features:
        neighbour_radius_mpc_h: 1.0

    splits:
      random_seed: 42
      train_fraction: 0.6
      val_fraction: 0.2
      within_sim:
        train: [wdm_sim]
        test: [wdm_sim]
      cross_softening:
        train: [wdm_sim]
        test: [wdm_sim]
      cross_z_ini:
        train: [wdm_sim]
        test: [wdm_sim]

    mlflow:
      tracking_uri: sqlite:///mlruns/mlflow.db
      experiment: test_experiment
""")


def _write_yaml(tmp_path: Path, yaml_text: str) -> Path:
    cfg_file = tmp_path / "config.yaml"
    _ = cfg_file.write_text(yaml_text)
    return cfg_file


class TestLoadConfig:
    def test_happy_path_returns_app_config(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _BASE_YAML))
        assert isinstance(cfg, AppConfig)
        assert len(cfg.simulations) == 2
        assert cfg.silver.merit_threshold == pytest.approx(0.1)

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _ = load_config("/nonexistent/path/config.yaml")

    def test_paths_resolved_to_absolute(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _BASE_YAML))
        assert cfg.database_path.is_absolute()

    def test_duplicate_simulation_id_raises(self, tmp_path: Path) -> None:
        yaml = _BASE_YAML.replace(
            "  - id: wdm_sim\n    cosmology: WDM",
            "  - id: cdm_sim\n    cosmology: WDM",
        )
        with pytest.raises(ValueError, match="Duplicate simulation id"):
            _ = load_config(_write_yaml(tmp_path, yaml))

    def test_invalid_cosmology_raises(self, tmp_path: Path) -> None:
        yaml = _BASE_YAML.replace("cosmology: CDM", "cosmology: INVALID", 1)
        with pytest.raises(ValueError, match="cosmology"):
            _ = load_config(_write_yaml(tmp_path, yaml))

    def test_invalid_softening_raises(self, tmp_path: Path) -> None:
        yaml = _BASE_YAML.replace("softening: fixed", "softening: INVALID", 1)
        with pytest.raises(ValueError, match="softening"):
            _ = load_config(_write_yaml(tmp_path, yaml))

    def test_merit_threshold_zero_raises(self, tmp_path: Path) -> None:
        yaml = _BASE_YAML.replace("merit_threshold: 0.1", "merit_threshold: 0.0")
        with pytest.raises(ValueError, match="merit_threshold"):
            _ = load_config(_write_yaml(tmp_path, yaml))

    def test_merit_threshold_one_raises(self, tmp_path: Path) -> None:
        yaml = _BASE_YAML.replace("merit_threshold: 0.1", "merit_threshold: 1.0")
        with pytest.raises(ValueError, match="merit_threshold"):
            _ = load_config(_write_yaml(tmp_path, yaml))

    def test_split_fractions_exceed_one_raises(self, tmp_path: Path) -> None:
        yaml = _BASE_YAML.replace("val_fraction: 0.2", "val_fraction: 0.5")
        with pytest.raises(ValueError, match="train_fraction"):
            _ = load_config(_write_yaml(tmp_path, yaml))

    def test_split_train_fraction_zero_raises(self, tmp_path: Path) -> None:
        yaml = _BASE_YAML.replace("train_fraction: 0.6", "train_fraction: 0.0")
        with pytest.raises(ValueError, match="train_fraction"):
            _ = load_config(_write_yaml(tmp_path, yaml))

    def test_split_references_unknown_sim_raises(self, tmp_path: Path) -> None:
        yaml = _BASE_YAML.replace(
            "  within_sim:\n    train: [wdm_sim]\n    test: [wdm_sim]",
            "  within_sim:\n    train: [nonexistent_sim]\n    test: [wdm_sim]",
        )
        with pytest.raises(ValueError, match="unknown simulation id"):
            _ = load_config(_write_yaml(tmp_path, yaml))


class TestAppConfigLookups:
    def test_simulation_found(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _BASE_YAML))
        sim = cfg.simulation("wdm_sim")
        assert sim.cosmology == "WDM"
        assert sim.z_ini == 39

    def test_simulation_not_found_raises(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _BASE_YAML))
        with pytest.raises(KeyError):
            _ = cfg.simulation("nonexistent")

    def test_simulations_by_cosmology_wdm(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _BASE_YAML))
        wdm = cfg.simulations_by_cosmology("WDM")
        assert len(wdm) == 1
        assert wdm[0].id == "wdm_sim"

    def test_simulations_by_cosmology_cdm(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _BASE_YAML))
        cdm = cfg.simulations_by_cosmology("CDM")
        assert len(cdm) == 1
        assert cdm[0].id == "cdm_sim"

    def test_crossmatch_for_found(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _BASE_YAML))
        cm = cfg.crossmatch_for("z39_fixed")
        assert cm.id == "z39_fixed"

    def test_crossmatch_for_not_found_raises(self, tmp_path: Path) -> None:
        cfg = load_config(_write_yaml(tmp_path, _BASE_YAML))
        with pytest.raises(KeyError):
            _ = cfg.crossmatch_for("nonexistent")
