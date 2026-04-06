"""
Tests for SignalController.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from traffic_system.signal_controller import SignalController, SIGNAL_GREEN, SIGNAL_RED
from config import NUM_LANES, LANE_NAMES


class TestSignalController:
    def setup_method(self):
        self.ctrl = SignalController()

    def test_initial_state_has_one_green(self):
        states = self.ctrl.get_state()["states"]
        assert states.count(SIGNAL_GREEN) == 1

    def test_initial_state_rest_red(self):
        states = self.ctrl.get_state()["states"]
        assert states.count(SIGNAL_RED) == NUM_LANES - 1

    def test_set_green_updates_correct_lane(self):
        for lane in range(NUM_LANES):
            self.ctrl.set_green(lane, 30)
            states = self.ctrl.get_state()["states"]
            assert states[lane] == SIGNAL_GREEN

    def test_set_green_makes_others_red(self):
        self.ctrl.set_green(2, 30)
        states = self.ctrl.get_state()["states"]
        for i, state in enumerate(states):
            if i == 2:
                assert state == SIGNAL_GREEN
            else:
                assert state == SIGNAL_RED

    def test_time_remaining_is_positive(self):
        self.ctrl.set_green(0, 30)
        assert self.ctrl.time_remaining > 0

    def test_emergency_override_sets_flag(self):
        self.ctrl.emergency_override(1)
        assert self.ctrl.emergency_override_active is True
        assert self.ctrl.get_state()["states"][1] == SIGNAL_GREEN

    def test_clear_emergency_override(self):
        self.ctrl.emergency_override(1)
        self.ctrl.clear_emergency_override()
        assert self.ctrl.emergency_override_active is False

    def test_get_signal_for_lane(self):
        self.ctrl.set_green(3, 20)
        assert self.ctrl.get_signal_for_lane(3) == SIGNAL_GREEN
        assert self.ctrl.get_signal_for_lane(0) == SIGNAL_RED

    def test_state_dict_has_expected_keys(self):
        state = self.ctrl.get_state()
        for key in ("states", "current_green_lane", "time_remaining",
                    "emergency_override", "green_duration"):
            assert key in state
