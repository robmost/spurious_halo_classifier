"""
config.py: loads config.yaml and exposes typed configuration objects.

All other modules import from here. Nothing in src/ reads config.yaml directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuration for a single simulation.
    """

    id: str
    cosmology: Literal["CDM", "WDM"]
    softening: Literal["fixed", "adaptive"]
    n_part_per_side: int
    z_ini: int
    box_size_mpc_h: float
    particle_mass_msun_h: float
    ahf_halos_dir: Path
    sphericity_path: Path


@dataclass(frozen=True)
class CrossmatchConfig:
    """
    Paths for one WDM-CDM crossmatch setup directory.
    """

    id: str  # e.g. "z39_adapt"
    wdm_cdm_path: Path  # WDM_CDM_crossmatch/<id>, used for is_spurious_cdm_match
    cdm_wdm_path: Path  # CDM_WDM_crossmatch/<id>, kept for reference


@dataclass(frozen=True)
class SilverConfig:
    """
    Configuration for the silver layer.
    """

    merit_threshold: float  # minimum WDM-CDM/CDM-WDM merit for has_cdm_counterpart = True


@dataclass(frozen=True)
class EmpiricalLabelConfig:
    """
    Configuration for the empirical sphericity label.
    """

    alpha: float  # slope of eq. 4, Mostoghiu Paun et al. (2025)
    beta: float  # intercept of eq. 4, Mostoghiu Paun et al. (2025)
    mass_threshold_log10: float  # log10(h^-1 M_sun), apply label only below this mass


@dataclass(frozen=True)
class SpatialFeaturesConfig:
    """
    Configuration for spatial features.
    """

    neighbour_radius_mpc_h: float


@dataclass(frozen=True)
class GoldConfig:
    """
    Configuration for the gold layer.
    """

    empirical_label: EmpiricalLabelConfig
    spatial_features: SpatialFeaturesConfig


@dataclass(frozen=True)
class SplitConfig:
    """
    Train/validation/test simulation IDs for a single split type.
    """

    train: list[str]
    test: list[str]


@dataclass(frozen=True)
class SplitsConfig:
    """
    Configuration for the train/validation/test splits.
    """

    random_seed: int
    train_fraction: float
    val_fraction: float
    within_sim: SplitConfig
    cross_softening: SplitConfig
    cross_z_ini: SplitConfig
    # cross_resolution is null, 256^3 data unavailable


@dataclass(frozen=True)
class MLflowConfig:
    """
    Configuration for MLflow experiment tracking.
    """

    tracking_uri: str  # e.g. "sqlite:///mlruns/mlflow.db" for local SQLite
    experiment: str  # MLflow experiment name


@dataclass(frozen=True)
class AppConfig:
    """
    Top level configuration for the application.
    """

    database_path: Path
    simulations: list[SimulationConfig]
    crossmatch: list[CrossmatchConfig]
    silver: SilverConfig
    gold: GoldConfig
    splits: SplitsConfig
    mlflow: MLflowConfig

    # ---------------------------------------------------------------------------
    # Convenience lookups
    # ---------------------------------------------------------------------------

    def simulation(self, sim_id: str) -> SimulationConfig:
        """
        Return a single SimulationConfig by id.
        Raises KeyError if not found.
        """
        for sim in self.simulations:
            if sim.id == sim_id:
                return sim
        raise KeyError(f"No simulation with id '{sim_id}'")

    def simulations_by_cosmology(self, cosmology: Literal["CDM", "WDM"]) -> list[SimulationConfig]:
        """
        Return all simulations for a given cosmology.
        """
        return [sim for sim in self.simulations if sim.cosmology == cosmology]

    def crossmatch_for(self, setup_id: str) -> CrossmatchConfig:
        """
        Return crossmatch config for a setup id (e.g. 'z39_adapt').
        Raises KeyError if not found.
        """
        for cmatch in self.crossmatch:
            if cmatch.id == setup_id:
                return cmatch
        raise KeyError(f"No crossmatch config for setup id '{setup_id}'")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> AppConfig:
    """
    Load and validate the config.yaml file.

    Parameters
    ----------
    path:
        Path to config.yaml. Defaults to the CONFIG_PATH environment variable,
        then to 'config.yaml' relative to the current working directory.

    Returns
    -------
    AppConfig
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist at the resolved path.
    ValueError
        If required fields are missing or values are invalid.
    """
    if path is None:
        path = os.environ.get("CONFIG_PATH", "config.yaml")
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found at '{path.resolve()}'")

    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"config.yaml must be a YAML mapping, got {type(raw).__name__}")

    return _parse(raw, config_dir=path.parent)


# ---------------------------------------------------------------------------
# Internal (private) parsing helpers
# ---------------------------------------------------------------------------


def _parse(raw: dict, config_dir: Path) -> AppConfig:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    """
    Parse the raw yaml dict into typed dataclasses.
    """

    def resolve(p: str) -> Path:
        """
        Resolve a path relative to the directory containing config.yaml.
        """
        return (config_dir / p).resolve()

    # --- Database ---
    database_path = resolve(raw["database"]["path"])

    # --- Simulations ---
    simulations: list[SimulationConfig] = []
    seen_ids: set[str] = set()
    for entry in raw["simulations"]:
        sim_id = entry["id"]
        if sim_id in seen_ids:
            raise ValueError(f"Duplicate simulation id '{sim_id}' in config.yaml")
        seen_ids.add(sim_id)

        cosmology = entry["cosmology"]
        if cosmology not in ("CDM", "WDM"):
            raise ValueError(
                f"Simulation '{sim_id}': cosmology must be 'CDM' or 'WDM', got '{cosmology}'"
            )

        softening = entry["softening"]
        if softening not in ("fixed", "adaptive"):
            raise ValueError(
                f"Simulation '{sim_id}': softening must be 'fixed' or 'adaptive', got '{softening}'"
            )

        simulations.append(
            SimulationConfig(
                id=sim_id,
                cosmology=cosmology,
                softening=softening,
                n_part_per_side=int(entry["n_part_per_side"]),
                z_ini=int(entry["z_ini"]),
                box_size_mpc_h=float(entry["box_size_mpc_h"]),
                particle_mass_msun_h=float(entry["particle_mass_msun_h"]),
                ahf_halos_dir=resolve(entry["ahf_halos_dir"]),
                sphericity_path=resolve(entry["sphericity_path"]),
            )
        )

    # --- Crossmatch ---
    # NOTE: CDM_WDM paths are derived by substituting the WDM_CDM directory name.
    # This avoids duplicating the path list in config.yaml, as the CDM_WDM path is predictable.
    crossmatch: list[CrossmatchConfig] = []
    raw_cm_base = raw["crossmatch"]["configurations"][0]["path"]
    # Extract the base crossmatch root by stripping the known subdirectory pattern
    # e.g. "data/raw/WDM_CDM_crossmatch/z39_adapt" -> "data/raw"
    wdm_cdm_root = resolve(raw_cm_base).parent.parent

    for entry in raw["crossmatch"]["configurations"]:
        setup_id = entry["id"]
        wdm_cdm_path = resolve(entry["path"])
        # CDM_WDM_crossmatch sits alongside WDM_CDM_crossmatch under the same root
        cdm_wdm_path = wdm_cdm_root / "CDM_WDM_crossmatch" / setup_id

        crossmatch.append(
            CrossmatchConfig(
                id=setup_id,
                wdm_cdm_path=wdm_cdm_path,
                cdm_wdm_path=cdm_wdm_path,
            )
        )

    # --- Silver ---
    merit_threshold = float(raw["silver"]["merit_threshold"])
    if not 0.0 < merit_threshold < 1.0:
        raise ValueError(f"silver.merit_threshold must be in (0, 1), got {merit_threshold}")
    silver = SilverConfig(merit_threshold=merit_threshold)

    # --- Gold ---
    emp_lab = raw["gold"]["empirical_label"]
    gold = GoldConfig(
        empirical_label=EmpiricalLabelConfig(
            alpha=float(emp_lab["alpha"]),
            beta=float(emp_lab["beta"]),
            mass_threshold_log10=float(emp_lab["mass_threshold_log10"]),
        ),
        spatial_features=SpatialFeaturesConfig(
            neighbour_radius_mpc_h=float(raw["gold"]["spatial_features"]["neighbour_radius_mpc_h"]),
        ),
    )

    # --- Splits ---
    raw_splits = raw["splits"]

    train_fraction = float(raw_splits["train_fraction"])
    val_fraction = float(raw_splits["val_fraction"])
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"splits.train_fraction must be in (0, 1), got {train_fraction}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"splits.val_fraction must be in (0, 1), got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            "splits.train_fraction + val_fraction must be < 1.0,"
            + f" got {train_fraction + val_fraction}"
        )

    splits = SplitsConfig(
        random_seed=int(raw_splits["random_seed"]),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        within_sim=_parse_split(raw_splits["within_sim"], "within_sim"),
        cross_softening=_parse_split(raw_splits["cross_softening"], "cross_softening"),
        cross_z_ini=_parse_split(raw_splits["cross_z_ini"], "cross_z_ini"),
    )

    # Validate that every simulation id referenced in splits exists
    all_sim_ids = {s.id for s in simulations}
    for split_name, split in [
        ("within_sim", splits.within_sim),
        ("cross_softening", splits.cross_softening),
        ("cross_z_ini", splits.cross_z_ini),
    ]:
        for sim_id in split.train + split.test:
            if sim_id not in all_sim_ids:
                raise ValueError(f"splits.{split_name} references unknown simulation id '{sim_id}'")

    # --- MLflow ---
    raw_mlflow = raw["mlflow"]
    mlflow_cfg = MLflowConfig(
        tracking_uri=str(raw_mlflow["tracking_uri"]),
        experiment=str(raw_mlflow["experiment"]),
    )

    return AppConfig(
        database_path=database_path,
        simulations=simulations,
        crossmatch=crossmatch,
        silver=silver,
        gold=gold,
        splits=splits,
        mlflow=mlflow_cfg,
    )


def _parse_split(raw_split: dict, name: str) -> SplitConfig:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    """
    Parse a single split entry, validating that train and test are non-empty lists.
    """
    train = raw_split.get("train")
    test = raw_split.get("test")
    if not train or not isinstance(train, list):
        raise ValueError(f"splits.{name}.train must be a non-empty list")
    if not test or not isinstance(test, list):
        raise ValueError(f"splits.{name}.test must be a non-empty list")
    return SplitConfig(train=train, test=test)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s  %(levelname)s  %(message)s"


def configure_logging(*, force: bool = False) -> None:
    """
    Configure the root logger for the application.

    Should be called once at the start of each entry-point module. All other
    modules inherit this configuration via `logging.getLogger(__name__)`.

    Parameters
    ----------
    force:
        Passed to `logging.basicConfig`. Set to True when an external
        library (e.g. MLflow) has already configured the root logger and
        the application format must take precedence.
    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, force=force)
