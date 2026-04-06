"""
Tests for TrafficRLOptimizer.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from traffic_system.rl_optimizer import TrafficRLOptimizer
from config import NUM_LANES, RL_NUM_DENSITY_LEVELS


class TestDiscretize:
    def test_low(self):
        assert TrafficRLOptimizer.discretize(0) == 0
        assert TrafficRLOptimizer.discretize(5) == 0

    def test_medium(self):
        assert TrafficRLOptimizer.discretize(6) == 1
        assert TrafficRLOptimizer.discretize(15) == 1

    def test_high(self):
        assert TrafficRLOptimizer.discretize(16) == 2
        assert TrafficRLOptimizer.discretize(25) == 2

    def test_critical(self):
        assert TrafficRLOptimizer.discretize(26) == 3
        assert TrafficRLOptimizer.discretize(100) == 3


class TestEncodeState:
    def setup_method(self):
        self.opt = TrafficRLOptimizer()

    def test_all_zero(self):
        assert self.opt.encode_state([0, 0, 0, 0]) == 0

    def test_consistent_encoding(self):
        s1 = self.opt.encode_state([5, 10, 20, 30])
        s2 = self.opt.encode_state([5, 10, 20, 30])
        assert s1 == s2

    def test_different_inputs_different_states(self):
        # Use counts above DENSITY_LOW threshold so density levels differ
        s1 = self.opt.encode_state([10, 0, 0, 0])  # lane 0 = medium (1), others = low (0)
        s2 = self.opt.encode_state([0, 10, 0, 0])  # lane 1 = medium (1), others = low (0)
        assert s1 != s2

    def test_state_within_bounds(self):
        max_state = RL_NUM_DENSITY_LEVELS ** NUM_LANES
        for _ in range(50):
            counts = list(np.random.randint(0, 40, NUM_LANES))
            state = self.opt.encode_state(counts)
            assert 0 <= state < max_state


class TestChooseAction:
    def setup_method(self):
        self.opt = TrafficRLOptimizer(epsilon=0.0)  # greedy for determinism

    def test_returns_valid_lane(self):
        counts = [5, 20, 10, 15]
        action = self.opt.choose_action(counts)
        assert 0 <= action < NUM_LANES

    def test_with_epsilon_one_returns_random(self):
        opt = TrafficRLOptimizer(epsilon=1.0)
        actions = {opt.choose_action([10] * NUM_LANES) for _ in range(50)}
        assert len(actions) > 1  # should explore multiple lanes


class TestCalculateReward:
    def setup_method(self):
        self.opt = TrafficRLOptimizer()

    def test_reward_is_float(self):
        reward = self.opt.calculate_reward([5, 10, 15, 20], 3)
        assert isinstance(reward, float)

    def test_higher_count_action_gives_less_negative_reward(self):
        counts = [5, 30, 5, 5]
        r_low = self.opt.calculate_reward(counts, 0)   # action on low lane
        r_high = self.opt.calculate_reward(counts, 1)  # action on high lane
        assert r_high > r_low


class TestUpdate:
    def setup_method(self):
        self.opt = TrafficRLOptimizer()

    def test_q_table_changes_after_update(self):
        counts = [10, 10, 10, 10]
        old_val = self.opt.q_table[self.opt.encode_state(counts), 0]
        self.opt.update(counts, 0, reward=5.0, next_vehicle_counts=counts)
        new_val = self.opt.q_table[self.opt.encode_state(counts), 0]
        assert old_val != new_val


class TestSaveLoad:
    def test_save_and_load(self):
        opt = TrafficRLOptimizer()
        opt.q_table[0, 0] = 42.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "q_table.npy")
            opt.save(path)
            opt2 = TrafficRLOptimizer()
            assert opt2.load(path) is True
            assert opt2.q_table[0, 0] == pytest.approx(42.0)

    def test_load_missing_file_returns_false(self):
        opt = TrafficRLOptimizer()
        assert opt.load("/nonexistent/q_table.npy") is False
