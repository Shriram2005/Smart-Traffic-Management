"""
traffic_density.py
------------------
Calculates traffic density per lane and converts vehicle counts into
adaptive green-light durations.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from config import (
    DENSITY_LOW,
    DENSITY_MEDIUM,
    DENSITY_HIGH,
    LANE_CAPACITY,
    MIN_GREEN_TIME,
    MAX_GREEN_TIME,
)

# Density classification labels
DENSITY_CLASS_LOW = "low"
DENSITY_CLASS_MEDIUM = "medium"
DENSITY_CLASS_HIGH = "high"
DENSITY_CLASS_CRITICAL = "critical"


class TrafficDensityCalculator:
    """Converts raw vehicle counts into density metrics and green-light durations."""

    # ------------------------------------------------------------------
    # Density metrics
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_density(vehicle_count: int, capacity: int = LANE_CAPACITY) -> float:
        """Return density as a percentage of lane capacity (0–100)."""
        if capacity <= 0:
            return 0.0
        return min(vehicle_count / capacity, 1.0) * 100.0

    @staticmethod
    def classify_density(vehicle_count: int) -> str:
        """Return a human-readable density classification string."""
        if vehicle_count <= DENSITY_LOW:
            return DENSITY_CLASS_LOW
        if vehicle_count <= DENSITY_MEDIUM:
            return DENSITY_CLASS_MEDIUM
        if vehicle_count <= DENSITY_HIGH:
            return DENSITY_CLASS_HIGH
        return DENSITY_CLASS_CRITICAL

    # ------------------------------------------------------------------
    # Adaptive green-light duration
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_green_time(
        vehicle_count: int,
        min_time: int = MIN_GREEN_TIME,
        max_time: int = MAX_GREEN_TIME,
    ) -> int:
        """Return the optimal green-light duration (seconds) for a given vehicle count.

        Uses a linear model: ``base + rate * count``, clamped to [min_time, max_time].
        """
        base_time = min_time
        time_per_vehicle = 1.5  # seconds per vehicle
        green_time = base_time + vehicle_count * time_per_vehicle
        return int(np.clip(green_time, min_time, max_time))

    # ------------------------------------------------------------------
    # Multi-lane helpers
    # ------------------------------------------------------------------

    def get_all_densities(
        self, vehicle_counts: List[int]
    ) -> List[Tuple[float, str, int]]:
        """Return a list of (density_pct, density_class, green_time) for each lane."""
        result = []
        for count in vehicle_counts:
            density_pct = self.calculate_density(count)
            density_cls = self.classify_density(count)
            green_time = self.calculate_green_time(count)
            result.append((density_pct, density_cls, green_time))
        return result

    @staticmethod
    def busiest_lane(vehicle_counts: List[int]) -> int:
        """Return the index of the lane with the highest vehicle count."""
        return int(np.argmax(vehicle_counts))
