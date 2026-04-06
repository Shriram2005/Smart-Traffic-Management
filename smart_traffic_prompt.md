# Smart Traffic Management System — AI Prompt

Build a complete Smart Traffic Management System using Python with a modular, production-ready architecture. The system must use 4 separate image inputs, each representing one incoming lane at a traffic intersection (North, South, East, West). The system should process each lane independently for vehicle detection and make a centralized decision for traffic signal control.

---

## Project Structure (strictly follow this)

```
traffic_system/
├── detection/
│   ├── __init__.py
│   └── lane_detector.py       # YOLOv8 detection per lane
├── logic/
│   ├── __init__.py
│   └── signal_controller.py   # centralized decision engine
├── dashboard/
│   ├── __init__.py
│   ├── app.py                 # Flask app
│   └── templates/
│       └── index.html
├── utils/
│   ├── __init__.py
│   └── config.py              # all constants and thresholds
├── sample_inputs/             # include 4 sample images (one per lane) for demo
├── main.py
├── requirements.txt
└── README.md
```

---

## Dependencies (`requirements.txt` must include these)

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
flask>=3.0.0
Pillow>=10.0.0
```

---

## Input Handling

- Accept 4 image file paths as inputs (one per lane: North, South, East, West), configurable in `config.py`.
- **Primary mode: static images.** Process each image independently on each detection cycle (simulating a real-time feed by re-reading and optionally augmenting the image).
- **Fallback:** If no image is provided for a lane, generate a synthetic test frame with a random number of colored rectangles to simulate vehicles. This ensures the system is demo-ready without real traffic images.
- **Optional extension (implement only if images are working):** Support video file inputs per lane using OpenCV's `VideoCapture`.

---

## Vehicle Detection (`detection/lane_detector.py`)

- Use `ultralytics` YOLOv8 with model `yolov8n.pt` (nano — fastest; it will auto-download on first run).
- Detect the following COCO classes only: `car`, `motorcycle`, `bus`, `truck`.
- For emergency detection, also flag: `ambulance` — since `ambulance` is not a COCO class, detect it by identifying vehicles with an aspect ratio and size characteristic of emergency vehicles, OR simply use class name matching if a custom model is provided. Document this limitation clearly in a comment.
- Each `LaneDetector` instance must be independent (one per lane) and run in its own thread.
- Return per-lane: list of bounding boxes, vehicle count, and an `emergency_detected` boolean.

---

## Threading and Synchronization

- Use Python `threading.Thread` with one thread per lane detector (4 threads total).
- Use `threading.Lock` to protect shared state (lane counts, signal state).
- Main loop collects results from all 4 threads before passing to the decision engine.
- Do **not** use multiprocessing — keep it single-process for Flask compatibility.

---

## Signal Logic (`logic/signal_controller.py`)

**Scoring and lane selection:**
- Score formula: `score = vehicle_count + (waiting_cycles * WAITING_WEIGHT)`
- `WAITING_WEIGHT = 0.5` (configurable in `config.py`)
- Select the lane with the highest score for GREEN.
- `waiting_cycles` increments by 1 each decision cycle for lanes that are not GREEN, and resets to 0 when a lane receives GREEN.

**Signal timing:**
- A lane stays GREEN for a fixed `GREEN_DURATION = 10` seconds (configurable).
- Before switching to a new GREEN lane, set all lanes to YELLOW for `YELLOW_DURATION = 3` seconds, then switch.
- Signal states: `RED`, `YELLOW`, `GREEN`.

**Emergency override:**
- If any lane reports `emergency_detected = True`, immediately override and set that lane to GREEN regardless of score.
- If multiple lanes report emergency, prioritize the one with the highest vehicle count.
- Maintain an `emergency_active` flag and `emergency_lane` string for the dashboard.
- Emergency green lasts for `EMERGENCY_DURATION = 15` seconds (configurable), then resumes normal logic.

---

## Dashboard (`dashboard/app.py` + `index.html`)

**Flask setup:**
- Use MJPEG streaming (`multipart/x-mixed-replace`) to stream each lane's annotated frame via individual endpoints: `/feed/north`, `/feed/south`, `/feed/east`, `/feed/west`.
- Expose a `/status` JSON endpoint polled every 1 second by the frontend via `fetch()` in JavaScript.

**`/status` response shape:**
```json
{
  "lanes": {
    "north": { "count": 8, "score": 10.5, "signal": "GREEN", "waiting": 5 },
    "south": { "count": 3, "score": 4.0,  "signal": "RED",   "waiting": 2 },
    "east":  { "count": 5, "score": 6.5,  "signal": "RED",   "waiting": 3 },
    "west":  { "count": 2, "score": 2.5,  "signal": "RED",   "waiting": 1 }
  },
  "emergency_active": false,
  "emergency_lane": null,
  "active_green": "north"
}
```

**Frontend (`index.html`) must include:**
1. 2×2 grid of `<img>` tags streaming the 4 MJPEG feeds with lane labels.
2. Signal panel: four traffic light indicators (red/yellow/green circles) — only one GREEN at a time.
3. Stats table: lane name, vehicle count, score, waiting cycles, signal state.
4. Emergency banner: shown only when `emergency_active = true`. Style it prominently in red with the text "EMERGENCY OVERRIDE ACTIVATED — [LANE NAME]".
5. **Live density chart:** Use Chart.js (CDN) to plot vehicle counts for all 4 lanes over time (last 30 data points). Update every second from `/status`.

**Styling:** Use plain HTML/CSS (no frameworks). Dark background preferred for a control-room aesthetic. All in a single `index.html` file.

---

## Visualization (annotated frames)

- Draw bounding boxes around detected vehicles using `cv2.rectangle()`.
- Label each box with the vehicle class name.
- Overlay text in the top-left corner: `"North: 8 vehicles"` (use white text with a dark background strip for readability).
- If `emergency_detected`, draw the entire frame border in red (4px) and add text `"EMERGENCY"` in red at the top.

---

## `config.py` (all tunable constants must live here)

```python
LANE_NAMES = ["north", "south", "east", "west"]
YOLO_MODEL = "yolov8n.pt"
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]
WAITING_WEIGHT = 0.5
GREEN_DURATION = 10       # seconds
YELLOW_DURATION = 3       # seconds
EMERGENCY_DURATION = 15   # seconds
FLASK_PORT = 5000
DETECTION_INTERVAL = 1.0  # seconds between detection cycles
```

---

## `README.md` must include

1. Project overview (2–3 sentences).
2. Installation: `pip install -r requirements.txt`.
3. How to run: `python main.py` and then open `http://localhost:5000`.
4. How to provide custom lane images (path config in `config.py`).
5. Known limitations (e.g., ambulance detection without a custom model).

---

## Final Requirements

- Every module must have a docstring.
- No logic should live in `main.py` beyond orchestration (starting threads, launching Flask).
- The system must be runnable end-to-end with just `python main.py` — no manual setup beyond `pip install -r requirements.txt`.
- Code must be clean, typed where practical, and commented at non-obvious logic points.
