"""
Shared pytest fixtures for the spurious halo classifier test suite.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import duckdb
import pytest

from src.config import (
    AppConfig,
    CrossmatchConfig,
    EmpiricalLabelConfig,
    GoldConfig,
    MLflowConfig,
    SilverConfig,
    SimulationConfig,
    SpatialFeaturesConfig,
    SplitConfig,
    SplitsConfig,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mem_conn() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    In-memory DuckDB connection with bronze, silver, and gold schemas created.
    """
    conn = duckdb.connect(":memory:")
    for schema in ("bronze", "silver", "gold"):
        _ = conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    yield conn
    conn.close()


@pytest.fixture
def minimal_cfg(tmp_path: Path) -> AppConfig:
    """
    Minimal AppConfig built directly from dataclasses, no I/O required.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return AppConfig(
        database_path=data_dir / "test.duckdb",
        simulations=[
            SimulationConfig(
                id="cdm_512_z39_fixed",
                cosmology="CDM",
                softening="fixed",
                n_part_per_side=512,
                z_ini=39,
                box_size_mpc_h=25.0,
                particle_mass_msun_h=1e8,
                ahf_halos_dir=data_dir,
                sphericity_path=data_dir / "sphericity.hdf5",
            ),
            SimulationConfig(
                id="wdm_512_z39_fixed",
                cosmology="WDM",
                softening="fixed",
                n_part_per_side=512,
                z_ini=39,
                box_size_mpc_h=25.0,
                particle_mass_msun_h=1e8,
                ahf_halos_dir=data_dir,
                sphericity_path=data_dir / "sphericity.hdf5",
            ),
        ],
        crossmatch=[
            CrossmatchConfig(
                id="z39_fixed",
                wdm_cdm_path=data_dir / "WDM_CDM_crossmatch" / "z39_fixed",
                cdm_wdm_path=data_dir / "CDM_WDM_crossmatch" / "z39_fixed",
            )
        ],
        silver=SilverConfig(merit_threshold=0.1),
        gold=GoldConfig(
            empirical_label=EmpiricalLabelConfig(
                alpha=0.0624,
                beta=-0.0988,
                mass_threshold_log10=10.0,
            ),
            spatial_features=SpatialFeaturesConfig(neighbour_radius_mpc_h=1.0),
        ),
        splits=SplitsConfig(
            random_seed=42,
            train_fraction=0.6,
            val_fraction=0.2,
            within_sim=SplitConfig(
                train=["wdm_512_z39_fixed"],
                test=["wdm_512_z39_fixed"],
            ),
            cross_softening=SplitConfig(
                train=["wdm_512_z39_fixed"],
                test=["wdm_512_z39_fixed"],
            ),
            cross_z_ini=SplitConfig(
                train=["wdm_512_z39_fixed"],
                test=["wdm_512_z39_fixed"],
            ),
        ),
        mlflow=MLflowConfig(
            tracking_uri="sqlite:///mlruns/mlflow.db",
            experiment="test_experiment",
        ),
    )
