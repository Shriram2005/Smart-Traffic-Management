# Smart Traffic Management System

An AI-powered, real-time traffic signal controller for a 4-way intersection. The system uses **YOLOv8** to detect vehicles in each lane, applies a score-based algorithm to decide which lane gets the green signal, and streams annotated video feeds to a live web dashboard.

## Screenshot
<img width="1919" height="1079" alt="project screenshot" src="https://github.com/user-attachments/assets/521fae0f-7f69-4c26-9732-50d18b3bce4f" />

## Features

- **Per-lane vehicle detection** — independent YOLOv8 (nano) inference on 4 concurrent threads.
- **Score-based signal control** — `score = vehicle_count + waiting_cycles × 0.5`.
- **Emergency override** — automatically grants green to lanes with detected emergency vehicles.
- **Live MJPEG dashboard** — 2×2 video grid, animated traffic lights, stats table, and Chart.js density graph.
- **Synthetic fallback** — if no real images are provided, the system generates random test frames.

## Installation

```bash
pip install -r requirements.txt
```

> The YOLOv8 nano model (`yolov8n.pt`) is downloaded automatically on first run.

## How to Run

```bash
python main.py
```

Then open **http://localhost:5000** in your browser.

## Custom Lane Images

Place your images in the `sample_inputs/` directory with the following names:

| Lane  | File                   |
|-------|------------------------|
| North | `sample_inputs/north.jpg` |
| South | `sample_inputs/south.jpg` |
| East  | `sample_inputs/east.jpg`  |
| West  | `sample_inputs/west.jpg`  |

Or edit `utils/config.py` → `LANE_IMAGES` to point to any file path.

## Configuration

All tuneable constants live in **`utils/config.py`**:

| Parameter            | Default | Purpose                              |
|----------------------|---------|--------------------------------------|
| `GREEN_DURATION`     | 10 s    | How long a lane stays green          |
| `YELLOW_DURATION`    | 3 s     | Transition time before switching     |
| `EMERGENCY_DURATION` | 15 s    | Green time for emergency override    |
| `WAITING_WEIGHT`     | 0.5     | Score multiplier for waiting cycles  |
| `DETECTION_INTERVAL` | 1.0 s   | Time between detection cycles        |
| `FLASK_PORT`         | 5000    | Dashboard server port                |

## Known Limitations

1. **Ambulance detection** — `ambulance` is not a standard COCO class. The system uses a heuristic (large `truck` detections with wide aspect ratio ≥ 2.0) as a proxy. For reliable emergency vehicle recognition, a custom-trained model is recommended.
2. **Static images** — the primary mode re-reads the same image each cycle. For real-time use, replace with a video stream or camera feed.
3. **Single-process** — the system runs in a single Python process (multi-threaded) for Flask compatibility.
