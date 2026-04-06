"""
emergency_handler.py
--------------------
Detects emergency vehicles across all lanes and determines which lane should
receive the highest-priority signal override.

Priority order: ambulance > fire_truck > police
"""

from __future__ import annotations

from typing import Dict, List, Optional

from config import EMERGENCY_VEHICLES

# Priority mapping (higher value = higher urgency)
EMERGENCY_PRIORITY: Dict[str, int] = {
    "ambulance": 3,
    "fire_truck": 2,
    "police": 1,
}


class EmergencyHandler:
    """Tracks active emergency vehicles and recommends signal overrides."""

    def __init__(self) -> None:
        # lane_id -> list of emergency vehicle types present
        self._active: Dict[int, List[str]] = {}

    # ------------------------------------------------------------------
    # Detection & tracking
    # ------------------------------------------------------------------

    def update(self, lane_id: int, vehicles: List[Dict]) -> bool:
        """Update the emergency status for *lane_id*.

        Args:
            lane_id: The lane being reported.
            vehicles: The vehicle list returned by VehicleDetector.

        Returns:
            True if at least one emergency vehicle is present in this lane.
        """
        emergency_types = [
            v["type"] for v in vehicles if v.get("is_emergency", False)
        ]
        if emergency_types:
            self._active[lane_id] = emergency_types
        else:
            self._active.pop(lane_id, None)
        return bool(emergency_types)

    def clear_lane(self, lane_id: int) -> None:
        """Manually clear emergency status for a lane (e.g. after vehicle passed)."""
        self._active.pop(lane_id, None)

    def clear_all(self) -> None:
        """Remove all active emergencies."""
        self._active.clear()

    # ------------------------------------------------------------------
    # Priority resolution
    # ------------------------------------------------------------------

    @property
    def has_emergency(self) -> bool:
        """True if any lane currently has an emergency vehicle."""
        return bool(self._active)

    def get_priority_lane(self) -> Optional[int]:
        """Return the lane ID that should receive signal priority, or None.

        When multiple lanes have emergency vehicles the one with the
        highest-priority vehicle type wins.  Ties are broken by lane ID
        (lower wins).
        """
        if not self._active:
            return None
        best_lane: Optional[int] = None
        best_score = 0
        for lane_id, types in self._active.items():
            score = max(EMERGENCY_PRIORITY.get(t, 0) for t in types)
            if score > best_score or (
                score == best_score and (best_lane is None or lane_id < best_lane)
            ):
                best_score = score
                best_lane = lane_id
        return best_lane

    def get_active_emergencies(self) -> Dict[int, List[str]]:
        """Return a copy of the active emergencies mapping."""
        return dict(self._active)
