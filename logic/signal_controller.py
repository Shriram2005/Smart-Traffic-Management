"""
Centralized traffic signal controller.

Collects detection results from all four lanes and decides which lane
receives GREEN using a score-based algorithm with emergency override.
"""

from __future__ import annotations

import threading
from time import monotonic
from dataclasses import dataclass, field
from typing import Optional

from utils.config import (
    EMERGENCY_DURATION,
    GREEN_DURATION,
    LANE_NAMES,
    WAITING_WEIGHT,
    YELLOW_DURATION,
)


@dataclass
class LaneState:
    """Mutable state for a single lane."""

    name: str
    vehicle_count: int = 0
    score: float = 0.0
    signal: str = "RED"          # RED | YELLOW | GREEN
    waiting_cycles: int = 0
    emergency_detected: bool = False


class SignalController:
    """
    Decision engine for the 4-way intersection.

    Call :meth:`update` once per detection cycle with new vehicle counts
    and emergency flags.  The controller handles timing, scoring, and
    state transitions internally.

    Attributes:
        lanes:            per-lane state objects
        active_green:     lane name currently (or last) granted GREEN
        emergency_active: whether an emergency override is in progress
        emergency_lane:   which lane triggered the override
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lanes: dict[str, LaneState] = {
            name: LaneState(name=name) for name in LANE_NAMES
        }
        self.active_green: str = LANE_NAMES[0]
        self.emergency_active: bool = False
        self.emergency_lane: Optional[str] = None

        # Internal timing state
        self._green_start: float = monotonic()
        self._yellow_active: bool = False
        self._yellow_start: float = 0.0
        self._emergency_start: float = 0.0

        # Start with the first lane green
        self.lanes[self.active_green].signal = "GREEN"

    # ── Public API ────────────────────────────────────────────────────
    def update(
        self,
        counts: dict[str, int],
        emergencies: dict[str, bool],
    ) -> None:
        """
        Feed new detection results and advance the signal state machine.

        Parameters:
            counts:      mapping of lane name → vehicle count.
            emergencies: mapping of lane name → emergency flag.
        """
        with self.lock:
            # 1. Refresh per-lane counts and emergency flags.
            for name in LANE_NAMES:
                ls = self.lanes[name]
                ls.vehicle_count = counts.get(name, 0)
                ls.emergency_detected = emergencies.get(name, False)

            # 2. Handle emergency override
            if self._check_emergency():
                return

            # 3. Handle yellow transition
            if self._yellow_active:
                if monotonic() - self._yellow_start >= YELLOW_DURATION:
                    self._finish_yellow()
                return  # do nothing else during yellow

            # 4. Check if current GREEN has expired
            if monotonic() - self._green_start >= GREEN_DURATION:
                self._begin_switch()

            # 5. Recompute scores
            self._update_scores()

    def get_status(self) -> dict:
        """
        Build the status dict consumed by the ``/status`` endpoint.
        """
        with self.lock:
            lanes_status = {}
            for name in LANE_NAMES:
                ls = self.lanes[name]
                lanes_status[name] = {
                    "count": ls.vehicle_count,
                    "score": round(ls.score, 1),
                    "signal": ls.signal,
                    "waiting": ls.waiting_cycles,
                }
            return {
                "lanes": lanes_status,
                "emergency_active": self.emergency_active,
                "emergency_lane": self.emergency_lane,
                "active_green": self.active_green,
            }

    # ── Internal helpers ──────────────────────────────────────────────
    def _update_scores(self) -> None:
        """Recompute the score for every lane."""
        for ls in self.lanes.values():
            ls.score = ls.vehicle_count + (ls.waiting_cycles * WAITING_WEIGHT)

    def _check_emergency(self) -> bool:
        """
        If any lane reports an emergency, override the current signal.

        Returns ``True`` if emergency handling consumed this tick.
        """
        # Check if current emergency has expired
        if self.emergency_active:
            if monotonic() - self._emergency_start >= EMERGENCY_DURATION:
                self.emergency_active = False
                self.emergency_lane = None
                self._set_green(self.active_green)
                self._green_start = monotonic()
            return True  # still in emergency — skip normal logic

        # Look for new emergency
        emergency_lanes = [
            name for name in LANE_NAMES
            if self.lanes[name].emergency_detected
        ]
        if not emergency_lanes:
            return False

        # If multiple emergencies, pick the one with the highest count
        chosen = max(emergency_lanes, key=lambda n: self.lanes[n].vehicle_count)
        self.emergency_active = True
        self.emergency_lane = chosen
        self._emergency_start = monotonic()
        self._set_green(chosen)
        self.active_green = chosen
        return True

    def _begin_switch(self) -> None:
        """Start a YELLOW transition before switching GREEN."""
        # Set all lanes to YELLOW briefly
        for ls in self.lanes.values():
            ls.signal = "YELLOW"
        self._yellow_active = True
        self._yellow_start = monotonic()

    def _finish_yellow(self) -> None:
        """End the YELLOW phase and select the next GREEN lane."""
        self._yellow_active = False

        # Increment waiting cycles for non-GREEN lanes
        for ls in self.lanes.values():
            if ls.name != self.active_green:
                ls.waiting_cycles += 1

        # Reset waiting for the lane that was GREEN
        self.lanes[self.active_green].waiting_cycles = 0

        # Recompute scores to pick next
        self._update_scores()

        # Select lane with highest score (excluding the one that just had GREEN)
        candidates = [
            ls for ls in self.lanes.values() if ls.name != self.active_green
        ]
        if candidates:
            best = max(candidates, key=lambda ls: ls.score)
        else:
            best = self.lanes[self.active_green]

        self.active_green = best.name
        self._set_green(best.name)
        self._green_start = monotonic()

    def _set_green(self, lane_name: str) -> None:
        """Set *lane_name* to GREEN, everything else to RED."""
        for ls in self.lanes.values():
            ls.signal = "GREEN" if ls.name == lane_name else "RED"
