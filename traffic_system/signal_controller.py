"""
signal_controller.py
--------------------
Manages traffic signal states (red / green) for all lanes at an intersection.

Key responsibilities:
- Grant a green phase to a single lane for a calculated duration.
- Support emergency override (instantly switch green to the emergency lane).
- Expose the current signal state to the rest of the system.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from config import (
    NUM_LANES,
    LANE_NAMES,
    DEFAULT_GREEN_TIME,
    YELLOW_TIME,
    ALL_RED_CLEARANCE,
    EMERGENCY_GREEN_TIME,
)

SIGNAL_GREEN = "green"
SIGNAL_YELLOW = "yellow"
SIGNAL_RED = "red"


class SignalController:
    """Controls traffic signals for a single intersection."""

    def __init__(self) -> None:
        self._states: List[str] = [SIGNAL_RED] * NUM_LANES
        self._states[0] = SIGNAL_GREEN          # lane 0 starts green
        self._current_green: int = 0
        self._green_duration: float = DEFAULT_GREEN_TIME
        self._green_start: float = time.monotonic()
        self.emergency_override_active: bool = False
        self._emergency_lane: Optional[int] = None

    # ------------------------------------------------------------------
    # Phase control
    # ------------------------------------------------------------------

    def set_green(self, lane_id: int, duration: float) -> None:
        """Grant a green phase to *lane_id* for *duration* seconds.

        All other lanes immediately return to red.
        """
        self._states = [SIGNAL_RED] * NUM_LANES
        self._states[lane_id] = SIGNAL_GREEN
        self._current_green = lane_id
        self._green_duration = duration
        self._green_start = time.monotonic()

    def emergency_override(self, lane_id: int) -> None:
        """Immediately grant green to an emergency lane, bypassing normal timing."""
        self.emergency_override_active = True
        self._emergency_lane = lane_id
        self.set_green(lane_id, EMERGENCY_GREEN_TIME)

    def clear_emergency_override(self) -> None:
        """Deactivate the emergency override flag."""
        self.emergency_override_active = False
        self._emergency_lane = None

    def next_phase(self, next_lane: int, duration: float) -> None:
        """Transition to the next green phase (with implicit yellow on current lane)."""
        # In a full hardware integration a yellow phase would be inserted here;
        # for the simulation we keep the transition instantaneous.
        self.set_green(next_lane, duration)

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    @property
    def current_green_lane(self) -> int:
        return self._current_green

    @property
    def time_remaining(self) -> float:
        """Seconds remaining in the current green phase (0 when expired)."""
        elapsed = time.monotonic() - self._green_start
        return max(0.0, self._green_duration - elapsed)

    @property
    def phase_expired(self) -> bool:
        """True when the current green phase has run its full duration."""
        return self.time_remaining == 0.0

    def get_state(self) -> Dict:
        """Return a snapshot of the current signal state."""
        return {
            "states": self._states.copy(),
            "lane_names": LANE_NAMES,
            "current_green_lane": self._current_green,
            "current_green_lane_name": LANE_NAMES[self._current_green],
            "green_duration": self._green_duration,
            "time_remaining": self.time_remaining,
            "emergency_override": self.emergency_override_active,
            "emergency_lane": self._emergency_lane,
        }

    def get_signal_for_lane(self, lane_id: int) -> str:
        """Return the current signal colour for a specific lane."""
        return self._states[lane_id]
