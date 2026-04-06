"""
Tests for TrafficDatabase.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from traffic_system.database import TrafficDatabase


@pytest.fixture
def db(tmp_path):
    """Provide a fresh in-temp-dir database for each test."""
    return TrafficDatabase(db_path=str(tmp_path / "test_traffic.db"))


class TestDatabaseInit:
    def test_tables_created(self, db):
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
        assert "traffic_logs" in tables
        assert "signal_events" in tables
        assert "vehicle_counts" in tables


class TestLogTraffic:
    def test_single_log(self, db):
        db.log_traffic(
            lane_id=0,
            vehicle_count=10,
            density=33.3,
            density_class="medium",
            green_duration=25,
            has_emergency=False,
        )
        df = db.get_recent_logs(limit=10)
        assert len(df) == 1
        assert df.iloc[0]["vehicle_count"] == 10
        assert df.iloc[0]["lane_id"] == 0

    def test_multiple_logs(self, db):
        for i in range(5):
            db.log_traffic(
                lane_id=i % 4,
                vehicle_count=i * 3,
                density=float(i * 10),
                density_class="low",
                green_duration=10,
                has_emergency=False,
            )
        df = db.get_recent_logs(limit=10)
        assert len(df) == 5

    def test_emergency_flag_stored(self, db):
        db.log_traffic(0, 5, 16.7, "low", 10, has_emergency=True)
        df = db.get_emergency_events()
        assert len(df) == 1


class TestLogVehicleCounts:
    def test_counts_stored(self, db):
        counts = {"car": 5, "bus": 2, "truck": 1, "bike": 3,
                  "ambulance": 0, "fire_truck": 0, "police": 0}
        db.log_vehicle_counts(lane_id=1, counts=counts)
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM vehicle_counts").fetchone()
        assert row["car"] == 5
        assert row["bus"] == 2


class TestLogSignalEvent:
    def test_event_stored(self, db):
        db.log_signal_event(lane_id=2, event_type="green_phase", duration=30)
        df = db.get_signal_events()
        assert len(df) == 1
        assert df.iloc[0]["event_type"] == "green_phase"
        assert df.iloc[0]["duration"] == 30


class TestGetTrafficSummary:
    def test_returns_dataframe(self, db):
        db.log_traffic(0, 10, 33.3, "medium", 25, False)
        db.log_traffic(1, 5, 16.7, "low", 10, False)
        summary = db.get_traffic_summary(hours=1)
        assert isinstance(summary, pd.DataFrame)

    def test_empty_when_no_data(self, db):
        summary = db.get_traffic_summary(hours=1)
        assert isinstance(summary, pd.DataFrame)
        # May be empty since no data
        assert "avg_vehicles" in summary.columns or summary.empty
