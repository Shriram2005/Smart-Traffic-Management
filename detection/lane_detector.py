"""
Lane-level vehicle detection using YOLOv8.

Each LaneDetector is an independent instance that processes one lane's
image feed. Instances are designed to run in their own thread.
"""

from __future__ import annotations

import os
import random
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from utils.config import (
    CONFIDENCE_THRESHOLD,
    get_lane_image,
    VEHICLE_CLASSES,
    YOLO_MODEL,
)


@dataclass
class DetectionResult:
    """Container for per-lane detection output."""

    lane_name: str
    vehicle_count: int = 0
    bounding_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    emergency_detected: bool = False
    annotated_frame: Optional[np.ndarray] = None


class LaneDetector:
    """
    Detects vehicles in a single lane using YOLOv8.

    One instance per lane. Thread-safe — each instance loads its own
    model reference (ultralytics handles model caching internally).

    Attributes:
        lane_name:  Identifier such as ``"north"`` or ``"east"``.
        model:      Shared YOLOv8 model instance.
        lock:       Per-instance lock guarding ``latest_result``.
    """

    # Class-level model — loaded once, shared across instances (thread-safe reads).
    _model: Optional[YOLO] = None
    _model_lock = threading.Lock()

    def __init__(self, lane_name: str) -> None:
        self.lane_name = lane_name
        self.lock = threading.Lock()
        self.latest_result = DetectionResult(lane_name=lane_name)
        self._ensure_model_loaded()

    # ── Model loading ─────────────────────────────────────────────────
    @classmethod
    def _ensure_model_loaded(cls) -> None:
        """Lazy-load the YOLOv8 model (once for all instances)."""
        if cls._model is None:
            with cls._model_lock:
                if cls._model is None:
                    cls._model = YOLO(YOLO_MODEL)

    # ── Frame acquisition ─────────────────────────────────────────────
    def _read_frame(self) -> tuple[np.ndarray, list[tuple[int, int, int, int]] | None]:
        """
        Read the input image for this lane.

        Falls back to a synthetic frame if the configured path is
        missing or ``None``.
        Returns a tuple of (frame, synthetic_boxes). If synthetic_boxes is None,
        the frame is real and should be passed to YOLO.
        """
        path = get_lane_image(self.lane_name)
        if path and os.path.isfile(path):
            frame = cv2.imread(path)
            if frame is not None:
                return frame, None
        # Fallback: generate synthetic test frame
        return self._generate_synthetic_frame()

    @staticmethod
    def _generate_synthetic_frame() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """
        Create a 640×480 frame with random coloured rectangles that
        loosely resemble vehicles.  Useful for demo when no real images
        are available.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Dark grey road
        frame[:] = (50, 50, 50)
        # Draw road markings
        for y in range(0, 480, 40):
            cv2.rectangle(frame, (318, y), (322, y + 20), (200, 200, 200), -1)

        num_vehicles = random.randint(2, 10)
        boxes = []
        for _ in range(num_vehicles):
            x1 = random.randint(50, 550)
            y1 = random.randint(50, 400)
            w = random.randint(40, 100)
            h = random.randint(30, 60)
            colour = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255),
            )
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), colour, -1)
            boxes.append((x1, y1, x1 + w, y1 + h))
        return frame, boxes

    # ── Detection ─────────────────────────────────────────────────────
    def detect(self) -> DetectionResult:
        """
        Run YOLOv8 on the current frame and return a ``DetectionResult``.

        NOTE on emergency (ambulance) detection:
          ``ambulance`` is NOT a standard COCO class. As a heuristic we
          check for large ``truck``-class detections with an aspect ratio
          ≥ 2.0 (wide and low), which loosely resembles ambulance /
          emergency-vehicle silhouettes.  For production use, a custom-
          trained model is strongly recommended.
        """
        frame, synthetic_boxes = self._read_frame()
        
        boxes: list[tuple[int, int, int, int]] = []
        classes: list[str] = []
        emergency = False

        if synthetic_boxes is not None:
            # Synthetic mode: YOLO will not detect our random rectangles,
            # so bypass YOLO and return the fake boxes.
            boxes = synthetic_boxes
            classes = ["car"] * len(boxes)
            # 10% chance to simulate an emergency in synthetic frames for demo purposes
            emergency = random.random() < 0.1
        else:
            # Real image: Run YOLO
            results = self._model.predict(  # type: ignore[union-attr]
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            for det in results:
                for box in det.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = det.names[cls_id]
                    if cls_name not in VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    boxes.append((x1, y1, x2, y2))
                    classes.append(cls_name)

                    # Heuristic emergency detection: large trucks with wide
                    # aspect ratio may be ambulances / fire engines.
                    w, h = x2 - x1, y2 - y1
                    if cls_name == "truck" and w > 0 and h > 0:
                        aspect = max(w, h) / min(w, h)
                        area = w * h
                        if aspect >= 2.0 and area >= 8000:
                            emergency = True

        annotated = self._annotate_frame(frame, boxes, classes, emergency)

        result = DetectionResult(
            lane_name=self.lane_name,
            vehicle_count=len(boxes),
            bounding_boxes=boxes,
            class_names=classes,
            emergency_detected=emergency,
            annotated_frame=annotated,
        )

        with self.lock:
            self.latest_result = result

        return result

    # ── Annotation ────────────────────────────────────────────────────
    def _annotate_frame(
        self,
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        classes: list[str],
        emergency: bool,
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and overlays on a copy of *frame*."""
        img = frame.copy()

        # Bounding boxes
        for (x1, y1, x2, y2), cls_name in zip(boxes, classes):
            colour = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            label = cls_name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(
                img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

        # Top-left overlay
        overlay_text = f"{self.lane_name.capitalize()}: {len(boxes)} vehicles"
        (tw, th), _ = cv2.getTextSize(
            overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
        )
        cv2.rectangle(img, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
        cv2.putText(
            img, overlay_text, (8, th + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        # Emergency border
        if emergency:
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
            cv2.putText(
                img, "EMERGENCY", (w // 2 - 80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
            )

        return img
