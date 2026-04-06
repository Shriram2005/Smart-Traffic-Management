"""
Tests for TrafficDensityCalculator.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from traffic_system.traffic_density import (
    TrafficDensityCalculator,
    DENSITY_CLASS_LOW,
    DENSITY_CLASS_MEDIUM,
    DENSITY_CLASS_HIGH,
    DENSITY_CLASS_CRITICAL,
)
from config import MIN_GREEN_TIME, MAX_GREEN_TIME


class TestCalculateDensity:
    def setup_method(self):
        self.calc = TrafficDensityCalculator()

    def test_zero_vehicles(self):
        assert self.calc.calculate_density(0) == 0.0

    def test_full_capacity(self):
        assert self.calc.calculate_density(30, capacity=30) == 100.0

    def test_over_capacity_clamped(self):
        assert self.calc.calculate_density(60, capacity=30) == 100.0

    def test_half_capacity(self):
        assert self.calc.calculate_density(15, capacity=30) == pytest.approx(50.0)

    def test_zero_capacity_returns_zero(self):
        assert self.calc.calculate_density(10, capacity=0) == 0.0


class TestClassifyDensity:
    def setup_method(self):
        self.calc = TrafficDensityCalculator()

    def test_low(self):
        assert self.calc.classify_density(0) == DENSITY_CLASS_LOW
        assert self.calc.classify_density(5) == DENSITY_CLASS_LOW

    def test_medium(self):
        assert self.calc.classify_density(6) == DENSITY_CLASS_MEDIUM
        assert self.calc.classify_density(15) == DENSITY_CLASS_MEDIUM

    def test_high(self):
        assert self.calc.classify_density(16) == DENSITY_CLASS_HIGH
        assert self.calc.classify_density(25) == DENSITY_CLASS_HIGH

    def test_critical(self):
        assert self.calc.classify_density(26) == DENSITY_CLASS_CRITICAL
        assert self.calc.classify_density(100) == DENSITY_CLASS_CRITICAL


class TestCalculateGreenTime:
    def setup_method(self):
        self.calc = TrafficDensityCalculator()

    def test_min_time_at_zero_vehicles(self):
        assert self.calc.calculate_green_time(0) == MIN_GREEN_TIME

    def test_does_not_exceed_max(self):
        assert self.calc.calculate_green_time(1000) == MAX_GREEN_TIME

    def test_increases_with_count(self):
        t1 = self.calc.calculate_green_time(5)
        t2 = self.calc.calculate_green_time(20)
        assert t2 > t1


class TestBusiestLane:
    def test_returns_correct_index(self):
        counts = [5, 20, 10, 15]
        assert TrafficDensityCalculator.busiest_lane(counts) == 1

    def test_first_lane_busiest(self):
        counts = [30, 0, 0, 0]
        assert TrafficDensityCalculator.busiest_lane(counts) == 0
