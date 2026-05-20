"""
db.py: DuckDB connection and schema initialisation.

All modules that need a database connection import get_connection() from here.
Note that no table definitions live here, those are instead described in sql/schema/.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Schema namespaces
# ---------------------------------------------------------------------------

_SCHEMAS = ("bronze", "silver", "gold")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def get_connection(database_path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection and ensure all schema namespaces exist.

    Parameters
    ----------
    database_path:
        Path to the .duckdb file. The parent directory must already exist.
    read_only:
        Open the database in read-only mode. Schema creation is skipped.
        Useful in notebooks that should not accidentally modify the database.

    Returns
    -------
    duckdb.DuckDBPyConnection
        An open connection. The caller is responsible for closing it.

    Raises
    ------
    FileNotFoundError
        If the parent directory of database_path does not exist.
    """
    if not database_path.parent.exists():
        raise FileNotFoundError(
            f"Database directory does not exist: '{database_path.parent}'."
            + " Must be created before calling get_connection()."
        )

    conn = duckdb.connect(str(database_path), read_only=read_only)

    if not read_only:
        _create_schemas(conn)

    return conn


def create_schemas(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create the bronze, silver, and gold schema namespaces if they do not exist.
    Safe to call multiple times because it uses IF NOT EXISTS.

    Parameters
    ----------
    conn:
        An open DuckDB connection.
    """
    _create_schemas(conn)


def reset_database(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Drop and recreate all three schema namespaces.

    Drops bronze, silver, and gold with CASCADE (so it removes all tables within
    them), then recreates the empty schemas. Use during development to rerun
    the full pipeline from scratch.

    Parameters
    ----------
    conn:
        An open DuckDB connection.
    """
    for schema in _SCHEMAS:
        _ = conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    _create_schemas(conn)


# ---------------------------------------------------------------------------
# Internal (private) helpers
# ---------------------------------------------------------------------------


def _create_schemas(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create all schema namespaces.
    """
    for schema in _SCHEMAS:
        _ = conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging

    from src.config import configure_logging, load_config

    configure_logging()
    log = logging.getLogger(__name__)

    cfg = load_config()
    if cfg.database_path.exists():
        cfg.database_path.unlink()
        log.info("Deleted %s", cfg.database_path)
    else:
        log.info("Nothing to delete: %s did not exist", cfg.database_path)
    log.info("Database reset complete. Run 'make bronze' to repopulate.")
