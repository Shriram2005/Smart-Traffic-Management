"""
rl_optimizer.py
---------------
Q-learning based traffic flow optimizer.

State space : density level (0-3) for each lane  → 4^N_LANES discrete states
Action space: choose which lane receives the green phase
Reward      : cleared-lane vehicles minus total waiting vehicles

The Q-table is small enough to be kept in memory and saved/loaded as a NumPy
array between simulation runs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np

from config import (
    NUM_LANES,
    RL_LEARNING_RATE,
    RL_DISCOUNT_FACTOR,
    RL_EPSILON,
    RL_NUM_DENSITY_LEVELS,
    DENSITY_LOW,
    DENSITY_MEDIUM,
    DENSITY_HIGH,
)


class TrafficRLOptimizer:
    """Q-learning optimizer for adaptive green-phase assignment."""

    def __init__(
        self,
        learning_rate: float = RL_LEARNING_RATE,
        discount_factor: float = RL_DISCOUNT_FACTOR,
        epsilon: float = RL_EPSILON,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # State: tuple of per-lane density levels (each 0..num_density_levels-1)
        state_count = RL_NUM_DENSITY_LEVELS ** NUM_LANES
        self.q_table = np.zeros((state_count, NUM_LANES), dtype=np.float64)

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    @staticmethod
    def discretize(vehicle_count: int) -> int:
        """Map a raw vehicle count to a density level index."""
        if vehicle_count <= DENSITY_LOW:
            return 0
        if vehicle_count <= DENSITY_MEDIUM:
            return 1
        if vehicle_count <= DENSITY_HIGH:
            return 2
        return 3

    def encode_state(self, vehicle_counts: List[int]) -> int:
        """Encode a per-lane vehicle count list into a single state index."""
        state = 0
        for i, count in enumerate(vehicle_counts):
            state += self.discretize(count) * (RL_NUM_DENSITY_LEVELS ** i)
        return state

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, vehicle_counts: List[int]) -> int:
        """Return the lane index that should receive the green signal.

        Uses an ε-greedy policy: with probability *epsilon* explore randomly,
        otherwise exploit the best-known Q-value.
        """
        if random.random() < self.epsilon:
            return random.randint(0, NUM_LANES - 1)
        state = self.encode_state(vehicle_counts)
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def calculate_reward(self, vehicle_counts: List[int], action: int) -> float:
        """Reward = vehicles cleared from green lane - total vehicles waiting.

        This incentivises the agent to clear congested lanes while keeping
        the overall intersection throughput high.
        """
        return float(vehicle_counts[action]) - float(sum(vehicle_counts))

    def update(
        self,
        vehicle_counts: List[int],
        action: int,
        reward: float,
        next_vehicle_counts: List[int],
    ) -> None:
        """Perform a single Q-learning update step."""
        state = self.encode_state(vehicle_counts)
        next_state = self.encode_state(next_vehicle_counts)

        current_q = self.q_table[state, action]
        max_next_q = float(np.max(self.q_table[next_state]))

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state, action] = new_q

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "q_table.npy") -> None:
        """Save the Q-table to a file."""
        np.save(path, self.q_table)

    def load(self, path: str = "q_table.npy") -> bool:
        """Load the Q-table from a file.  Returns True on success."""
        p = Path(path)
        if not p.exists():
            return False
        loaded = np.load(path)
        if loaded.shape == self.q_table.shape:
            self.q_table = loaded
            return True
        return False
