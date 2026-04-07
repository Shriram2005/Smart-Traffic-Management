"""
Flask dashboard for the Smart Traffic Management System.

Provides:
- MJPEG streaming endpoints for each lane's annotated video feed.
- A JSON ``/status`` endpoint polled by the frontend.
- The main ``index.html`` control-room interface.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

import cv2
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from utils.config import LANE_NAMES, SAMPLE_DIR, set_lane_image

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


# ── Image Upload ──────────────────────────────────────────────────────

@app.route("/upload/<lane_name>", methods=["POST"])
def upload_image(lane_name: str):
    """Handle image upload for a specific lane from the dashboard."""
    if lane_name not in LANE_NAMES:
        return jsonify({"error": "Lane not found"}), 404
        
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        # Use timestamp to avoid caching and overwriting issues
        safe_name = f"{lane_name}_{int(time.time())}_{filename}"
        filepath = os.path.join(SAMPLE_DIR, safe_name)
        
        # Ensure sample dir exists
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        file.save(filepath)
        
        # Update the dictionary in memory so detectors instantly pick it up
        set_lane_image(lane_name, filepath)
        return jsonify({"success": True})
    return jsonify({"error": "Unknown error"}), 500


# ── Main page ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")
