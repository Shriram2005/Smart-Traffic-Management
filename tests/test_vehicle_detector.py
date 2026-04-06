"""
Tests for VehicleDetector.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from traffic_system.vehicle_detector import VehicleDetector
from config import VEHICLE_TYPES, EMERGENCY_VEHICLES


class TestVehicleDetectorSimulation:
    def test_returns_list(self):
        det = VehicleDetector(lane_id=0)
        vehicles = det.detect_vehicles()
        assert isinstance(vehicles, list)

    def test_vehicle_has_required_keys(self):
        det = VehicleDetector(lane_id=0)
        for _ in range(5):
            vehicles = det.detect_vehicles()
            for v in vehicles:
                assert "type" in v
                assert "is_emergency" in v

    def test_vehicle_types_are_valid(self):
        det = VehicleDetector(lane_id=0)
        for _ in range(5):
            vehicles = det.detect_vehicles()
            for v in vehicles:
                assert v["type"] in VEHICLE_TYPES

    def test_emergency_flag_matches_type(self):
        det = VehicleDetector(lane_id=0)
        for _ in range(10):
            vehicles = det.detect_vehicles()
            for v in vehicles:
                if v["type"] in EMERGENCY_VEHICLES:
                    assert v["is_emergency"] is True
                else:
                    assert v["is_emergency"] is False


class TestHasEmergencyVehicle:
    def test_no_emergency(self):
        vehicles = [{"type": "car", "is_emergency": False}]
        assert VehicleDetector.has_emergency_vehicle(vehicles) is False

    def test_with_emergency(self):
        vehicles = [
            {"type": "car", "is_emergency": False},
            {"type": "ambulance", "is_emergency": True},
        ]
        assert VehicleDetector.has_emergency_vehicle(vehicles) is True

    def test_empty_list(self):
        assert VehicleDetector.has_emergency_vehicle([]) is False


class TestCountByType:
    def test_count_basic(self):
        vehicles = [
            {"type": "car", "is_emergency": False},
            {"type": "car", "is_emergency": False},
            {"type": "ambulance", "is_emergency": True},
        ]
        counts = VehicleDetector.count_by_type(vehicles)
        assert counts["car"] == 2
        assert counts["ambulance"] == 1

    def test_count_empty(self):
        counts = VehicleDetector.count_by_type([])
        for vtype in VEHICLE_TYPES:
            assert counts[vtype] == 0
