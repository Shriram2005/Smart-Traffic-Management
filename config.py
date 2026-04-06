"""
Configuration settings for the Smart Traffic Management System.
"""

# ─── Intersection layout ──────────────────────────────────────────────────────
NUM_LANES = 4
LANE_NAMES = ["North", "South", "East", "West"]
LANE_CAPACITY = 30  # max vehicles per lane at a time

# ─── Vehicle types ────────────────────────────────────────────────────────────
VEHICLE_TYPES = ["car", "bus", "truck", "bike", "ambulance", "fire_truck", "police"]
EMERGENCY_VEHICLES = ["ambulance", "fire_truck", "police"]

# Sampling weights used in simulation (must match VEHICLE_TYPES order)
VEHICLE_TYPE_WEIGHTS = [0.60, 0.10, 0.10, 0.15, 0.02, 0.02, 0.01]

# ─── Traffic density thresholds (vehicle count) ───────────────────────────────
DENSITY_LOW = 5
DENSITY_MEDIUM = 15
DENSITY_HIGH = 25

# ─── Signal timing (seconds) ─────────────────────────────────────────────────
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60
DEFAULT_GREEN_TIME = 30
YELLOW_TIME = 3
ALL_RED_CLEARANCE = 2
EMERGENCY_GREEN_TIME = 45  # green duration granted to emergency lane

# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH = "traffic_data.db"

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_HISTORY = 50    # data-points retained for trend plots
DASHBOARD_REFRESH = 1.0   # seconds between simulation steps

# ─── Reinforcement Learning ───────────────────────────────────────────────────
RL_LEARNING_RATE = 0.1
RL_DISCOUNT_FACTOR = 0.9
RL_EPSILON = 0.1
RL_NUM_DENSITY_LEVELS = 4  # 0=low, 1=medium, 2=high, 3=critical
