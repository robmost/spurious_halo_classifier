"""
build_gold.py: pipeline entrypoint for the gold layer.

Invoked by `make gold` as `python -m src.gold.build_gold`.

Calls the three gold modules in dependency order:
  1. labels: requires silver.protohalos and silver.wdm_cdm_matches
  2. features: requires silver.protohalos
  3. splits: requires gold.labels
"""

from __future__ import annotations

import logging

from src.config import AppConfig, configure_logging, load_config
from src.db import get_connection, log_row_counts
from src.gold.features import build_features
from src.gold.labels import build_labels
from src.gold.splits import build_splits

configure_logging()
log = logging.getLogger(__name__)


def build_gold(cfg: AppConfig) -> None:
    """
    Populate all gold tables from silver.

    All tables are dropped and recreated on each run; gold is idempotent.

    Parameters
    ----------
    cfg:
        Loaded application configuration.
    """
    conn = get_connection(cfg.database_path)

    try:
        build_labels(conn, cfg)
        build_features(conn, cfg)
        build_splits(conn, cfg)
        log_row_counts(conn, _GOLD_TABLES, "Gold")
    finally:
        conn.close()


_GOLD_TABLES = [
    "gold.labels",
    "gold.features",
    "gold.train_test_splits",
]


if __name__ == "__main__":
    cfg = load_config()
    log.info("Database: %s", cfg.database_path)
    build_gold(cfg)
    log.info("Gold layer complete.")
