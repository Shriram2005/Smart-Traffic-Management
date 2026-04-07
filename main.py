"""
Smart Traffic Management System — Entry Point.

Orchestrates:
1. Initialisation of per-lane detectors.
2. A background detection loop (one thread per lane).
3. The centralized signal controller.
4. The Flask dashboard server.

Usage::

    python main.py
    # Then open http://localhost:5000
"""

from __future__ import annotations

import sys
import os
import threading
import time

# Ensure the project root is on sys.path so relative imports work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.lane_detector import LaneDetector
from logic.signal_controller import SignalController
from dashboard.app import app, set_shared_state
from utils.config import DETECTION_INTERVAL, FLASK_PORT, LANE_NAMES


def detection_loop(
    detector: LaneDetector,
    stop_event: threading.Event,
) -> None:
    """
    Continuously run detection for a single lane, feeding results
    into the shared *controller*.

    Each lane gets its own thread running this function.
    """
    while not stop_event.is_set():
        try:
            detector.detect()
        except Exception as exc:
            print(f"[{detector.lane_name}] detection error: {exc}")
        time.sleep(DETECTION_INTERVAL)


def control_loop(
    controller: SignalController,
    detectors: dict[str, LaneDetector],
    stop_event: threading.Event,
) -> None:
    """
    Periodically collect counts from detectors and call
    ``controller.update()`` to advance the signal state machine.
    """
    while not stop_event.is_set():
        counts: dict[str, int] = {}
        emergencies: dict[str, bool] = {}
        for name, det in detectors.items():
            with det.lock:
                counts[name] = det.latest_result.vehicle_count
                emergencies[name] = det.latest_result.emergency_detected
        controller.update(counts, emergencies)
        time.sleep(DETECTION_INTERVAL)


def main() -> None:
    """Start the entire traffic management system."""
    print("=" * 60)
    print("  Smart Traffic Management System")
    print("=" * 60)

    # 1. Create detectors
    detectors: dict[str, LaneDetector] = {}
    for name in LANE_NAMES:
        print(f"  ► Initialising detector for lane: {name}")
        detectors[name] = LaneDetector(name)
    print()

    # 2. Create controller
    controller = SignalController()

    # 3. Share state with Flask
    set_shared_state(detectors, controller)

    # 4. Start detection threads
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    for name, det in detectors.items():
        t = threading.Thread(
            target=detection_loop,
            args=(det, stop_event),
            daemon=True,
            name=f"detect-{name}",
        )
        t.start()
        threads.append(t)
        print(f"  ✓ Detection thread started: {name}")

    # 5. Start signal controller thread
    sig_thread = threading.Thread(
        target=control_loop,
        args=(controller, detectors, stop_event),
        daemon=True,
        name="signal-controller",
    )
    sig_thread.start()
    print("  ✓ Signal controller thread started")

    # 6. Start Flask (blocks until interrupted)
    print(f"\n  Dashboard: http://localhost:{FLASK_PORT}\n")
    try:
        app.run(
            host="0.0.0.0",
            port=FLASK_PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()
        for t in threads:
            t.join(timeout=2)
        sig_thread.join(timeout=2)
        print("Done.")


if __name__ == "__main__":
    main()
