# Smart Traffic Management System

A machine learning-based smart traffic management system that automatically controls traffic signals based on real-time traffic density and provides priority routing for emergency vehicles.

## Features

| Requirement | Module |
|---|---|
| Real-time vehicle detection (cars, buses, trucks, bikes, ambulance, fire truck, police) | `traffic_system/vehicle_detector.py` |
| Traffic density calculation for each lane | `traffic_system/traffic_density.py` |
| Adaptive traffic signal timing based on congestion | `traffic_system/signal_controller.py` |
| Emergency vehicle detection and automatic signal override | `traffic_system/emergency_handler.py` |
| Traffic flow optimisation using Reinforcement Learning (Q-learning) | `traffic_system/rl_optimizer.py` |
| Live traffic monitoring dashboard (Matplotlib) | `traffic_system/dashboard.py` |
| Data storage for traffic logs and analytics (SQLite) | `traffic_system/database.py` |

## Project Structure

```
Smart-Traffic-Management/
├── main.py                         # Entry point / simulation runner
├── config.py                       # All tuneable parameters
├── requirements.txt
├── traffic_system/
│   ├── vehicle_detector.py         # Simulated & OpenCV-based detection
│   ├── traffic_density.py          # Density metrics & green-time calculation
│   ├── signal_controller.py        # Signal state machine
│   ├── emergency_handler.py        # Emergency priority resolution
│   ├── rl_optimizer.py             # Q-learning optimizer
│   ├── database.py                 # SQLite persistence
│   └── dashboard.py                # Matplotlib monitoring dashboard
└── tests/
    ├── test_vehicle_detector.py
    ├── test_traffic_density.py
    ├── test_signal_controller.py
    ├── test_emergency_handler.py
    ├── test_rl_optimizer.py
    └── test_database.py
```

## Technologies

- **Python 3.10+**
- **NumPy / Pandas** – numerical processing and analytics
- **Matplotlib / Seaborn** – live monitoring dashboard
- **SQLite** – traffic log storage (via Python `sqlite3`)
- **Scikit-learn** – density classification helpers
- **OpenCV** (`opencv-python-headless`) – live camera-based vehicle detection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run a 30-step simulation (default)
python main.py

# Run 100 steps, save a dashboard snapshot every 20 steps
python main.py --steps 100 --plot-interval 20

# Run without saving PNG snapshots
python main.py --steps 50 --no-save-plots

# Use a custom database path
python main.py --db my_traffic.db
```

## How It Works

### 1. Vehicle Detection
Each lane is served by a `VehicleDetector` instance.  In **simulation mode** (default) vehicles are drawn from a Poisson distribution with realistic type weights (60 % cars, 15 % bikes, 10 % buses/trucks, ~5 % emergency).  In **live mode** it uses an OpenCV background-subtraction pipeline on camera frames.

### 2. Traffic Density
`TrafficDensityCalculator` converts raw vehicle counts into density percentages and classifies them as *low / medium / high / critical*.  Green-light durations are scaled linearly with count (clamped to `[MIN_GREEN_TIME, MAX_GREEN_TIME]`).

### 3. Adaptive Signal Control
`SignalController` maintains a single-green-phase state machine.  Only one lane is green at a time; all others are red.  Emergency lanes bypass the normal phase cycle.

### 4. Emergency Vehicle Priority
`EmergencyHandler` monitors all lanes and assigns priorities: **ambulance (3) > fire truck (2) > police (1)**.  The highest-priority lane immediately receives a 45-second green override, pre-empting the RL scheduler.

### 5. Reinforcement Learning Optimiser
`TrafficRLOptimizer` implements tabular Q-learning:
- **State** – per-lane density level (0–3), encoded as a single integer
- **Action** – which lane receives the next green phase
- **Reward** – vehicles cleared from the chosen lane minus total vehicles waiting
- The Q-table is persisted to `q_table.npy` and reloaded on startup so the agent improves across runs.

### 6. Dashboard
`TrafficDashboard` generates a 2 × 2 Matplotlib figure:
- Current vehicle count per lane (colour-coded by signal / emergency)
- Signal state tiles
- Vehicle count trend over recent steps
- Average traffic density per lane (from database)

### 7. Database
`TrafficDatabase` writes to three SQLite tables:
- `traffic_logs` – per-lane per-step snapshot
- `vehicle_counts` – per-type vehicle counts
- `signal_events` – every phase change or emergency override

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All 64 unit tests cover every module.
