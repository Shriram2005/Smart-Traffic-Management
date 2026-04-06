"""
Tests for EmergencyHandler.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from traffic_system.emergency_handler import EmergencyHandler


def _make_vehicles(*types):
    """Helper to create a vehicle list from type names."""
    emergency = {"ambulance", "fire_truck", "police"}
    return [{"type": t, "is_emergency": t in emergency} for t in types]


class TestEmergencyHandler:
    def setup_method(self):
        self.handler = EmergencyHandler()

    def test_no_emergency_by_default(self):
        assert self.handler.has_emergency is False
        assert self.handler.get_priority_lane() is None

    def test_detects_ambulance(self):
        vehicles = _make_vehicles("car", "ambulance")
        result = self.handler.update(0, vehicles)
        assert result is True
        assert self.handler.has_emergency is True

    def test_no_emergency_vehicles(self):
        vehicles = _make_vehicles("car", "bus")
        result = self.handler.update(1, vehicles)
        assert result is False
        assert self.handler.has_emergency is False

    def test_priority_lane_highest_priority(self):
        # Lane 0 has police (priority 1), lane 1 has ambulance (priority 3)
        self.handler.update(0, _make_vehicles("police"))
        self.handler.update(1, _make_vehicles("ambulance"))
        assert self.handler.get_priority_lane() == 1  # ambulance wins

    def test_priority_fire_truck_over_police(self):
        self.handler.update(2, _make_vehicles("police"))
        self.handler.update(3, _make_vehicles("fire_truck"))
        assert self.handler.get_priority_lane() == 3  # fire_truck wins

    def test_clear_lane(self):
        self.handler.update(0, _make_vehicles("ambulance"))
        assert self.handler.has_emergency is True
        self.handler.clear_lane(0)
        assert self.handler.has_emergency is False

    def test_clear_all(self):
        self.handler.update(0, _make_vehicles("ambulance"))
        self.handler.update(1, _make_vehicles("police"))
        self.handler.clear_all()
        assert self.handler.has_emergency is False

    def test_update_removes_cleared_emergency(self):
        self.handler.update(0, _make_vehicles("ambulance"))
        # Now only normal vehicles remain
        self.handler.update(0, _make_vehicles("car"))
        assert self.handler.has_emergency is False

    def test_get_active_emergencies_returns_copy(self):
        self.handler.update(2, _make_vehicles("ambulance"))
        active = self.handler.get_active_emergencies()
        assert 2 in active
        # Mutating the returned dict should not affect internal state
        active[99] = ["police"]
        assert 99 not in self.handler.get_active_emergencies()
