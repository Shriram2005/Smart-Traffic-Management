"""
Flask dashboard for the Smart Traffic Management System.

Provides:
- MJPEG streaming endpoints for each lane's annotated video feed.
- A JSON ``/status`` endpoint polled by the frontend.
- The main ``index.html`` control-room interface.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import cv2
from flask import Flask, Response, jsonify, render_template

from utils.config import LANE_NAMES

if TYPE_CHECKING:
    from detection.lane_detector import LaneDetector
    from logic.signal_controller import SignalController

app = Flask(__name__)

# These are set by main.py before the server starts.
detectors: dict[str, LaneDetector] = {}
controller: SignalController | None = None


def set_shared_state(
    det: dict[str, LaneDetector],
    ctrl: SignalController,
) -> None:
    """Inject shared detectors and controller into the Flask module."""
    global detectors, controller
    detectors = det
    controller = ctrl


# ── MJPEG streaming ──────────────────────────────────────────────────

def _generate_mjpeg(lane_name: str):
    """Yield JPEG frames for MJPEG streaming of a single lane."""
    while True:
        det = detectors.get(lane_name)
        if det is None:
            time.sleep(0.5)
            continue
        with det.lock:
            frame = det.latest_result.annotated_frame
        if frame is None:
            time.sleep(0.1)
            continue
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
        time.sleep(0.05)  # ~20 fps cap to save CPU


@app.route("/feed/<lane_name>")
def video_feed(lane_name: str):
    """Stream the annotated frame for *lane_name* via MJPEG."""
    if lane_name not in LANE_NAMES:
        return "Lane not found", 404
    return Response(
        _generate_mjpeg(lane_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Status endpoint ──────────────────────────────────────────────────

@app.route("/status")
def status():
    """Return current signal state as JSON."""
    if controller is None:
        return jsonify({"error": "Controller not initialised"}), 503
    return jsonify(controller.get_status())


# ── Main page ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")
