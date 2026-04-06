"""
main.py
-------
Entry point for the Smart Traffic Management System.

Runs a simulation loop that:
  1. Detects vehicles in each lane (simulated).
  2. Calculates traffic density.
  3. Checks for emergency vehicles and overrides signals if required.
  4. Uses the RL optimizer to choose the optimal green lane.
  5. Updates the signal controller.
  6. Persists data to SQLite.
  7. Renders and saves the monitoring dashboard.

Usage
-----
    python main.py                   # run with default settings
    python main.py --steps 50        # run 50 simulation steps
    python main.py --steps 20 --no-save-plots
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

from config import (
    NUM_LANES,
    LANE_NAMES,
    DASHBOARD_REFRESH,
    DEFAULT_GREEN_TIME,
)
from traffic_system.vehicle_detector import VehicleDetector
from traffic_system.traffic_density import TrafficDensityCalculator
from traffic_system.signal_controller import SignalController
from traffic_system.emergency_handler import EmergencyHandler
from traffic_system.rl_optimizer import TrafficRLOptimizer
from traffic_system.database import TrafficDatabase
from traffic_system.dashboard import TrafficDashboard


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20, max_val: float = 30.0) -> str:
    """Return a simple ASCII progress bar for console output."""
    filled = int(round(value / max_val * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _print_state(
    step: int,
    vehicle_counts: List[int],
    signal_states: List[str],
    has_emergencies: List[bool],
    rl_action: int,
    time_remaining: float,
) -> None:
    """Print a formatted status block to the console."""
    print(f"\n{'='*60}")
    print(f" Step {step:>4d}  |  RL chose lane: {LANE_NAMES[rl_action]}  |  "
          f"Time remaining: {time_remaining:.1f}s")
    print(f"{'─'*60}")
    for i in range(NUM_LANES):
        signal = signal_states[i].upper()
        emerg = " 🚨 EMERGENCY" if has_emergencies[i] else ""
        bar = _bar(vehicle_counts[i])
        print(f"  {LANE_NAMES[i]:6s}  [{signal:5s}]{emerg:<14s}  "
              f"{bar}  {vehicle_counts[i]:2d} vehicles")
    print(f"{'='*60}", flush=True)


# ─── main simulation loop ─────────────────────────────────────────────────────

def run_simulation(
    steps: int = 30,
    save_plots: bool = True,
    plot_interval: int = 10,
    db_path: str = "traffic_data.db",
) -> None:
    """Run the traffic management simulation for *steps* cycles."""

    print("Initialising Smart Traffic Management System …")

    # Initialise components
    detectors = [VehicleDetector(i, simulation_mode=True) for i in range(NUM_LANES)]
    density_calc = TrafficDensityCalculator()
    signal_ctrl = SignalController()
    emergency_handler = EmergencyHandler()
    rl_optimizer = TrafficRLOptimizer()
    database = TrafficDatabase(db_path=db_path)
    dashboard = TrafficDashboard()

    # Try to load a pre-trained Q-table
    rl_optimizer.load("q_table.npy")

    prev_counts: List[int] = [0] * NUM_LANES
    rl_action = 0

    print(f"Starting simulation — {steps} steps …\n")

    for step in range(1, steps + 1):

        # ── 1. Detect vehicles ────────────────────────────────────────
        vehicles_per_lane = [det.detect_vehicles() for det in detectors]
        vehicle_counts = [len(v) for v in vehicles_per_lane]

        # ── 2. Density calculation ────────────────────────────────────
        density_info = density_calc.get_all_densities(vehicle_counts)
        # density_info[i] = (density_pct, density_class, green_time)

        # ── 3. Emergency vehicle detection ────────────────────────────
        has_emergencies: List[bool] = []
        for i, vehicles in enumerate(vehicles_per_lane):
            has_em = emergency_handler.update(i, vehicles)
            has_emergencies.append(has_em)

        # ── 4. Signal control ─────────────────────────────────────────
        priority_lane = emergency_handler.get_priority_lane()

        if priority_lane is not None and not signal_ctrl.emergency_override_active:
            # Emergency override
            signal_ctrl.emergency_override(priority_lane)
            rl_action = priority_lane
            database.log_signal_event(priority_lane, "emergency_override", 45)
            print(f"  ⚠  Emergency override → {LANE_NAMES[priority_lane]}")

        elif signal_ctrl.emergency_override_active and priority_lane is None:
            # Emergency cleared
            signal_ctrl.clear_emergency_override()

        if not signal_ctrl.emergency_override_active:
            # ── 5. RL optimizer picks the green lane ──────────────────
            rl_action = rl_optimizer.choose_action(vehicle_counts)

            # Calculate reward and update Q-table
            reward = rl_optimizer.calculate_reward(vehicle_counts, rl_action)
            rl_optimizer.update(prev_counts, rl_action, reward, vehicle_counts)

            # Set signal with density-adaptive duration
            green_duration = density_info[rl_action][2]
            signal_ctrl.set_green(rl_action, green_duration)
            database.log_signal_event(rl_action, "green_phase", green_duration)

        # ── 6. Persist data ───────────────────────────────────────────
        signal_state = signal_ctrl.get_state()
        for i in range(NUM_LANES):
            dpct, dclass, gtime = density_info[i]
            database.log_traffic(
                lane_id=i,
                vehicle_count=vehicle_counts[i],
                density=dpct,
                density_class=dclass,
                green_duration=gtime,
                has_emergency=has_emergencies[i],
            )
            counts_by_type = VehicleDetector.count_by_type(vehicles_per_lane[i])
            database.log_vehicle_counts(i, counts_by_type)

        # ── 7. Dashboard update ───────────────────────────────────────
        dashboard.update(vehicle_counts, signal_state["states"], has_emergencies)

        # Console output
        _print_state(
            step,
            vehicle_counts,
            signal_state["states"],
            has_emergencies,
            rl_action,
            signal_state["time_remaining"],
        )

        if save_plots and step % plot_interval == 0:
            fig = dashboard.render(
                vehicle_counts, signal_state["states"], has_emergencies, db=database
            )
            plot_path = f"traffic_dashboard_step{step:04d}.png"
            fig.savefig(plot_path, bbox_inches="tight", dpi=100)
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  📊 Dashboard saved → {plot_path}")

        prev_counts = vehicle_counts
        time.sleep(DASHBOARD_REFRESH * 0.05)  # throttle for demo purposes

    # ── Save final artefacts ──────────────────────────────────────────────────
    rl_optimizer.save("q_table.npy")
    print("\nQ-table saved to q_table.npy")

    # Final dashboard snapshot
    if save_plots:
        final_vehicles = vehicle_counts  # last step's counts
        fig = dashboard.render(
            final_vehicles,
            signal_ctrl.get_state()["states"],
            has_emergencies,
            db=database,
        )
        fig.savefig("traffic_dashboard_final.png", bbox_inches="tight", dpi=100)
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("Final dashboard saved → traffic_dashboard_final.png")

    # Print analytics summary
    print("\n─── Traffic Summary (last hour) ───────────────────────────")
    summary = database.get_traffic_summary()
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("  (no data yet)")

    print("\nSimulation complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Traffic Management System")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of simulation steps (default: 30)")
    parser.add_argument("--no-save-plots", action="store_true",
                        help="Disable saving dashboard PNG files")
    parser.add_argument("--plot-interval", type=int, default=10,
                        help="Save a dashboard snapshot every N steps (default: 10)")
    parser.add_argument("--db", type=str, default="traffic_data.db",
                        help="Path to the SQLite database file (default: traffic_data.db)")
    args = parser.parse_args()

    run_simulation(
        steps=args.steps,
        save_plots=not args.no_save_plots,
        plot_interval=args.plot_interval,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
