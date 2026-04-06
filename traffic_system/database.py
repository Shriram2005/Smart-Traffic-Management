"""
database.py
-----------
SQLite-backed data store for traffic logs, signal events, and vehicle counts.
Exposes helper methods to write real-time data and read aggregated analytics.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Optional

import pandas as pd

from config import DB_PATH, LANE_NAMES


class TrafficDatabase:
    """Manages persistent storage for the traffic management system."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create database tables if they don't already exist."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traffic_logs (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      TEXT    NOT NULL,
                    lane_id        INTEGER NOT NULL,
                    lane_name      TEXT    NOT NULL,
                    vehicle_count  INTEGER NOT NULL,
                    density        REAL    NOT NULL,
                    density_class  TEXT    NOT NULL,
                    green_duration INTEGER NOT NULL,
                    has_emergency  INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS signal_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT    NOT NULL,
                    lane_id    INTEGER NOT NULL,
                    event_type TEXT    NOT NULL,
                    duration   INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vehicle_counts (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT    NOT NULL,
                    lane_id    INTEGER NOT NULL,
                    car        INTEGER NOT NULL DEFAULT 0,
                    bus        INTEGER NOT NULL DEFAULT 0,
                    truck      INTEGER NOT NULL DEFAULT 0,
                    bike       INTEGER NOT NULL DEFAULT 0,
                    ambulance  INTEGER NOT NULL DEFAULT 0,
                    fire_truck INTEGER NOT NULL DEFAULT 0,
                    police     INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def log_traffic(
        self,
        lane_id: int,
        vehicle_count: int,
        density: float,
        density_class: str,
        green_duration: int,
        has_emergency: bool,
    ) -> None:
        """Insert one traffic-state record for a lane."""
        lane_name = LANE_NAMES[lane_id] if lane_id < len(LANE_NAMES) else str(lane_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traffic_logs
                    (timestamp, lane_id, lane_name, vehicle_count, density,
                     density_class, green_duration, has_emergency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    lane_id,
                    lane_name,
                    vehicle_count,
                    density,
                    density_class,
                    green_duration,
                    int(has_emergency),
                ),
            )

    def log_vehicle_counts(self, lane_id: int, counts: Dict[str, int]) -> None:
        """Insert per-type vehicle counts for a lane."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO vehicle_counts
                    (timestamp, lane_id, car, bus, truck, bike, ambulance, fire_truck, police)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    lane_id,
                    counts.get("car", 0),
                    counts.get("bus", 0),
                    counts.get("truck", 0),
                    counts.get("bike", 0),
                    counts.get("ambulance", 0),
                    counts.get("fire_truck", 0),
                    counts.get("police", 0),
                ),
            )

    def log_signal_event(
        self, lane_id: int, event_type: str, duration: int
    ) -> None:
        """Record a signal phase change event."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signal_events (timestamp, lane_id, event_type, duration)
                VALUES (?, ?, ?, ?)
                """,
                (datetime.now().isoformat(), lane_id, event_type, duration),
            )

    # ------------------------------------------------------------------
    # Read / analytics
    # ------------------------------------------------------------------

    def get_recent_logs(self, limit: int = 100) -> pd.DataFrame:
        """Return the *limit* most-recent traffic log rows as a DataFrame."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM traffic_logs ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(limit,),
            )

    def get_traffic_summary(self, hours: int = 1) -> pd.DataFrame:
        """Return per-lane aggregated stats for the past *hours* hours."""
        # Validate hours is a positive integer to prevent any injection
        hours = max(1, int(hours))
        cutoff = f"-{hours} hours"
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT
                    lane_name,
                    ROUND(AVG(vehicle_count), 1)  AS avg_vehicles,
                    ROUND(AVG(density), 1)         AS avg_density,
                    SUM(has_emergency)             AS total_emergencies,
                    COUNT(*)                       AS total_readings
                FROM traffic_logs
                WHERE timestamp >= datetime('now', ?)
                GROUP BY lane_name
                ORDER BY lane_id
                """,
                conn,
                params=(cutoff,),
            )

    def get_emergency_events(self, limit: int = 50) -> pd.DataFrame:
        """Return recent rows that had an emergency vehicle present."""
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT * FROM traffic_logs
                WHERE has_emergency = 1
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )

    def get_signal_events(self, limit: int = 50) -> pd.DataFrame:
        """Return recent signal-change events."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM signal_events ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(limit,),
            )
