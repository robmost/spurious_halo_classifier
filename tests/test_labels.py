"""
Tests for src/gold/labels.py.
"""

from __future__ import annotations

import duckdb
import polars as pl

from src.config import AppConfig
from src.gold.labels import build_labels


def _setup_silver_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Populate the minimum silver tables required by build_labels.

    Halo catalogue (all WDM except halo 5):
        1  has protohalo data, sphericity 0.5 above s_bar ~= 0.463 -> empirical genuine
           has CDM counterpart                                     -> cdm_match genuine
        2  has protohalo data, sphericity 0.1 below s_bar ~= 0.463 -> empirical spurious
           no CDM counterpart                                      -> cdm_match spurious
        3  no protohalo data                                       -> empirical null
           absent from wdm_cdm_matches                             -> cdm_match spurious
        4  above mass threshold (m_hmm=1.5e10 ≥ 10^10)             -> empirical genuine (forced)
           absent from wdm_cdm_matches                             -> cdm_match spurious
        5  CDM halo — must not appear in gold.labels
    """
    pairs = pl.DataFrame(
        {
            "simulation_pair_id": ["z39_fixed"],
            "wdm_simulation_id": ["wdm_512_z39_fixed"],
            "cdm_simulation_id": ["cdm_512_z39_fixed"],
        }
    )
    _ = conn.register("_pairs", pairs)
    _ = conn.execute("CREATE TABLE silver.simulation_pairs AS SELECT * FROM _pairs")
    _ = conn.unregister("_pairs")

    matches = pl.DataFrame(
        {
            "simulation_pair_id": ["z39_fixed", "z39_fixed"],
            "wdm_halo_id": [1, 2],
            "has_cdm_counterpart": [True, False],
        }
    )
    _ = conn.register("_matches", matches)
    _ = conn.execute("CREATE TABLE silver.wdm_cdm_matches AS SELECT * FROM _matches")
    _ = conn.unregister("_matches")

    protohalos = pl.DataFrame(
        {
            "halo_id": [1, 2, 3, 4, 5],
            "simulation_id": ["wdm_512_z39_fixed"] * 4 + ["cdm_512_z39_fixed"],
            "cosmology": ["WDM", "WDM", "WDM", "WDM", "CDM"],
            "has_protohalo_data": [True, True, False, True, False],
            "sphericity_s": [0.5, 0.1, None, 0.5, None],
            "m_hmm": [1e9, 1e9, None, 1.5e10, None],
        }
    )
    _ = conn.register("_protos", protohalos)
    _ = conn.execute("CREATE TABLE silver.protohalos AS SELECT * FROM _protos")
    _ = conn.unregister("_protos")


class TestBuildLabels:
    def test_wdm_haloes_only_in_output(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        labels = mem_conn.execute("SELECT * FROM gold.labels").pl()
        assert len(labels) == 4  # halo 5 (CDM) excluded

    def test_all_label_columns_present(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        labels = mem_conn.execute("SELECT * FROM gold.labels").pl()
        assert {
            "halo_id",
            "simulation_id",
            "is_spurious_empirical",
            "is_spurious_cdm_match",
            "label_agreement",
        }.issubset(set(labels.columns))

    # --- is_spurious_cdm_match ---

    def test_cdm_match_genuine_when_counterpart_exists(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute(
            "SELECT is_spurious_cdm_match FROM gold.labels WHERE halo_id = 1"
        ).pl()
        assert row["is_spurious_cdm_match"][0] is False

    def test_cdm_match_spurious_when_no_counterpart(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute(
            "SELECT is_spurious_cdm_match FROM gold.labels WHERE halo_id = 2"
        ).pl()
        assert row["is_spurious_cdm_match"][0] is True

    def test_cdm_match_spurious_when_absent_from_matches_table(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        """
        Null has_cdm_counterpart (halo absent from wdm_cdm_matches) -> treated as spurious.
        """
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute(
            "SELECT is_spurious_cdm_match FROM gold.labels WHERE halo_id = 3"
        ).pl()
        assert row["is_spurious_cdm_match"][0] is True

    # --- is_spurious_empirical ---

    def test_empirical_null_when_no_protohalo_data(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute(
            "SELECT is_spurious_empirical FROM gold.labels WHERE halo_id = 3"
        ).pl()
        assert row["is_spurious_empirical"][0] is None

    def test_empirical_genuine_when_sphericity_above_boundary(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        """
        Halo 1: s=0.5 > s_bar(1e9) ~= 0.463 -> genuine.
        """
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute(
            "SELECT is_spurious_empirical FROM gold.labels WHERE halo_id = 1"
        ).pl()
        assert row["is_spurious_empirical"][0] is False

    def test_empirical_spurious_when_sphericity_below_boundary(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        """
        Halo 2: s=0.1 < s_bar(1e9) ~= 0.463 -> spurious.
        """
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute(
            "SELECT is_spurious_empirical FROM gold.labels WHERE halo_id = 2"
        ).pl()
        assert row["is_spurious_empirical"][0] is True

    def test_empirical_genuine_forced_above_mass_threshold(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        """
        Halo 4: m_hmm=1.5e10 >= 10^10 -> label forced to False regardless of sphericity.
        """
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute(
            "SELECT is_spurious_empirical FROM gold.labels WHERE halo_id = 4"
        ).pl()
        assert row["is_spurious_empirical"][0] is False

    # --- label_agreement ---

    def test_agreement_null_when_empirical_label_null(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute("SELECT label_agreement FROM gold.labels WHERE halo_id = 3").pl()
        assert row["label_agreement"][0] is None

    def test_agreement_true_when_both_genuine(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        """
        Halo 1: empirical=False, cdm_match=False -> agree.
        """
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute("SELECT label_agreement FROM gold.labels WHERE halo_id = 1").pl()
        assert row["label_agreement"][0] is True

    def test_agreement_true_when_both_spurious(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        """
        Halo 2: empirical=True, cdm_match=True -> agree.
        """
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute("SELECT label_agreement FROM gold.labels WHERE halo_id = 2").pl()
        assert row["label_agreement"][0] is True

    def test_agreement_false_when_labels_disagree(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        """
        Halo 4: empirical=False (above threshold), cdm_match=True -> disagree.
        """
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        row = mem_conn.execute("SELECT label_agreement FROM gold.labels WHERE halo_id = 4").pl()
        assert row["label_agreement"][0] is False

    def test_idempotent_second_call_does_not_duplicate_rows(
        self, mem_conn: duckdb.DuckDBPyConnection, minimal_cfg: AppConfig
    ) -> None:
        _setup_silver_tables(mem_conn)
        build_labels(mem_conn, minimal_cfg)
        build_labels(mem_conn, minimal_cfg)
        labels = mem_conn.execute("SELECT * FROM gold.labels").pl()
        assert len(labels) == 4
