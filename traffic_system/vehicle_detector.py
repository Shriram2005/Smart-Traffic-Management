"""
vehicle_detector.py
-------------------
Detects and classifies vehicles in a traffic lane.

In simulation mode (default) a Poisson-distributed random vehicle list is
produced for each call, which is useful for testing and demos without camera
hardware.

In live mode the module expects a raw BGR frame (NumPy array) from a camera
and uses a background-subtraction + contour-detection pipeline to identify
moving vehicles.  Emergency vehicles are flagged based on aspect ratio / size
heuristics; a real deployment would replace this with a trained classifier
(e.g. YOLO).
"""

from __future__ import annotations

import random
from typing import List, Dict, Any

import numpy as np

from config import (
    VEHICLE_TYPES,
    VEHICLE_TYPE_WEIGHTS,
    EMERGENCY_VEHICLES,
)


class VehicleDetector:
    """Detects vehicles in a single lane."""

    def __init__(self, lane_id: int, simulation_mode: bool = True) -> None:
        self.lane_id = lane_id
        self.simulation_mode = simulation_mode
        # Background subtractor used in live mode
        self._bg_subtractor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_vehicles(self, frame: np.ndarray | None = None) -> List[Dict[str, Any]]:
        """Return a list of detected vehicle dicts for the current cycle.

        Each dict has the keys:
          - ``type``         (str)  – vehicle category
          - ``is_emergency`` (bool) – True for ambulance / fire_truck / police
        """
        if self.simulation_mode:
            return self._simulate_detection()
        return self._opencv_detection(frame)

    @staticmethod
    def has_emergency_vehicle(vehicles: List[Dict[str, Any]]) -> bool:
        """Return True if *any* vehicle in the list is an emergency vehicle."""
        return any(v["is_emergency"] for v in vehicles)

    @staticmethod
    def count_by_type(vehicles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Return a mapping of vehicle type → count."""
        counts: Dict[str, int] = {vtype: 0 for vtype in VEHICLE_TYPES}
        for v in vehicles:
            counts[v["type"]] = counts.get(v["type"], 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate_detection(self) -> List[Dict[str, Any]]:
        """Generate a realistic random vehicle list using a Poisson arrival model."""
        num_vehicles = int(np.random.poisson(lam=10))
        vehicles: List[Dict[str, Any]] = []
        for _ in range(num_vehicles):
            vtype = random.choices(VEHICLE_TYPES, weights=VEHICLE_TYPE_WEIGHTS, k=1)[0]
            vehicles.append(
                {
                    "type": vtype,
                    "is_emergency": vtype in EMERGENCY_VEHICLES,
                }
            )
        return vehicles

    # ------------------------------------------------------------------
    # Live (OpenCV) detection
    # ------------------------------------------------------------------

    def _opencv_detection(self, frame: np.ndarray | None) -> List[Dict[str, Any]]:
        """Detect moving vehicles in a camera frame using background subtraction."""
        if frame is None:
            return []

        try:
            import cv2  # noqa: PLC0415 – imported here to keep it optional
        except ImportError:
            return []

        # Lazy-init background subtractor
        if self._bg_subtractor is None:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=50, detectShadows=False
            )

        # Pre-process
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = self._bg_subtractor.apply(gray)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        vehicles: List[Dict[str, Any]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue  # skip noise
            x, y, w, h = cv2.boundingRect(cnt)
            vtype = self._classify_by_size(w, h, area)
            vehicles.append(
                {
                    "type": vtype,
                    "is_emergency": vtype in EMERGENCY_VEHICLES,
                    "bbox": (x, y, w, h),
                }
            )
        return vehicles

    @staticmethod
    def _classify_by_size(w: int, h: int, area: float) -> str:
        """Heuristic size-based vehicle classification (placeholder)."""
        if area > 15_000:
            return "truck"
        if area > 8_000:
            return "bus"
        aspect = w / max(h, 1)
        if aspect < 0.7:
            return "bike"
        return "car"
