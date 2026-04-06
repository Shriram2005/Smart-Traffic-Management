"""
Configuration constants for the Smart Traffic Management System.

All tunable parameters are defined here so they can be adjusted
without modifying any logic files.
"""

import os

# ── Lane Configuration ────────────────────────────────────────────────
LANE_NAMES: list[str] = ["north", "south", "east", "west"]

# ── YOLO Model ────────────────────────────────────────────────────────
YOLO_MODEL: str = "yolov8n.pt"

# COCO class names to count as vehicles
VEHICLE_CLASSES: list[str] = ["car", "motorcycle", "bus", "truck"]

# ── Signal Timing (seconds) ──────────────────────────────────────────
GREEN_DURATION: int = 10        # how long a lane stays GREEN
YELLOW_DURATION: int = 3        # transition YELLOW before switching
EMERGENCY_DURATION: int = 15    # GREEN duration for emergency override

# ── Scoring ───────────────────────────────────────────────────────────
WAITING_WEIGHT: float = 0.5     # multiplier for waiting_cycles in score

# ── Detection ─────────────────────────────────────────────────────────
DETECTION_INTERVAL: float = 1.0  # seconds between detection cycles
CONFIDENCE_THRESHOLD: float = 0.35  # minimum confidence for detections

# ── Flask ─────────────────────────────────────────────────────────────
FLASK_PORT: int = 5000

# ── Lane Image Paths ─────────────────────────────────────────────────
# Map each lane name to the path of its input image.
# Set a path to None or leave it missing to use synthetic test frames.
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR: str = os.path.join(BASE_DIR, "sample_inputs")

LANE_IMAGES: dict[str, str | None] = {
    "north": os.path.join(SAMPLE_DIR, "north.jpg"),
    "south": os.path.join(SAMPLE_DIR, "south.jpg"),
    "east":  os.path.join(SAMPLE_DIR, "east.jpg"),
    "west":  os.path.join(SAMPLE_DIR, "west.jpg"),
}
