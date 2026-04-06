"""
dashboard.py
------------
Live monitoring dashboard powered by Matplotlib.

The dashboard is updated every simulation step and shows:
  1. Current vehicle count per lane (bar chart with colour-coded signals)
  2. Current signal states (coloured indicator tiles)
  3. Vehicle count trend over recent steps (line chart)
  4. Average traffic density per lane from the database (bar chart)
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Deque

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import LANE_NAMES, NUM_LANES, DASHBOARD_HISTORY


class TrafficDashboard:
    """Collects real-time data and renders the monitoring dashboard."""

    def __init__(self, max_history: int = DASHBOARD_HISTORY) -> None:
        self.max_history = max_history
        self._vehicle_history: List[Deque[int]] = [
            deque(maxlen=max_history) for _ in range(NUM_LANES)
        ]
        self._time_steps: Deque[int] = deque(maxlen=max_history)
        self._step = 0

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(
        self,
        vehicle_counts: List[int],
        signal_states: List[str],
        has_emergencies: List[bool],
    ) -> None:
        """Record the latest system snapshot."""
        self._step += 1
        self._time_steps.append(self._step)
        for i in range(NUM_LANES):
            self._vehicle_history[i].append(vehicle_counts[i])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        vehicle_counts: List[int],
        signal_states: List[str],
        has_emergencies: List[bool],
        db=None,
    ) -> plt.Figure:
        """Create and return a Matplotlib Figure with the current dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(
            "Smart Traffic Management System — Live Dashboard",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

        self._plot_vehicle_counts(axes[0, 0], vehicle_counts, signal_states, has_emergencies)
        self._plot_signal_tiles(axes[0, 1], signal_states, has_emergencies)
        self._plot_trend(axes[1, 0])
        self._plot_density(axes[1, 1], vehicle_counts, db)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    # ------------------------------------------------------------------
    # Individual sub-plots
    # ------------------------------------------------------------------

    def _plot_vehicle_counts(self, ax, vehicle_counts, signal_states, has_emergencies):
        ax.set_facecolor("#16213e")
        bar_colors = []
        for i in range(NUM_LANES):
            if has_emergencies[i]:
                bar_colors.append("#ff4757")
            elif signal_states[i] == "green":
                bar_colors.append("#2ed573")
            else:
                bar_colors.append("#ff6b81")

        bars = ax.bar(LANE_NAMES, vehicle_counts, color=bar_colors, edgecolor="white", linewidth=1.2)
        ax.set_title("Current Vehicle Count per Lane", color="white", fontsize=11)
        ax.set_xlabel("Lane", color="white")
        ax.set_ylabel("Vehicles", color="white")
        ax.set_ylim(0, 35)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        for bar, count in zip(bars, vehicle_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontweight="bold",
                color="white",
                fontsize=10,
            )

        # Legend
        legend_patches = [
            mpatches.Patch(color="#2ed573", label="Green"),
            mpatches.Patch(color="#ff6b81", label="Red"),
            mpatches.Patch(color="#ff4757", label="Emergency"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", facecolor="#16213e", labelcolor="white")

    def _plot_signal_tiles(self, ax, signal_states, has_emergencies):
        ax.set_facecolor("#16213e")
        ax.set_xlim(-0.5, NUM_LANES - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(NUM_LANES))
        ax.set_xticklabels(LANE_NAMES, color="white")
        ax.set_yticks([])
        ax.set_title("Signal States", color="white", fontsize=11)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        for i, (state, lane) in enumerate(zip(signal_states, LANE_NAMES)):
            if has_emergencies[i]:
                color, label = "#ff4757", "!! EMERGENCY !!"
            elif state == "green":
                color, label = "#2ed573", "GREEN"
            else:
                color, label = "#ff6b81", "RED"
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (i - 0.4, 0.1), 0.8, 0.8,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor="white", linewidth=1.5,
                )
            )
            ax.text(i, 0.5, label, ha="center", va="center",
                    fontweight="bold", fontsize=9, color="white")

    def _plot_trend(self, ax):
        ax.set_facecolor("#16213e")
        ax.set_title("Vehicle Count Trend", color="white", fontsize=11)
        ax.set_xlabel("Step", color="white")
        ax.set_ylabel("Vehicles", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.grid(True, alpha=0.2, color="gray")

        colors = ["#2ed573", "#ffa502", "#ff6b81", "#1e90ff"]
        steps = list(self._time_steps)
        for i in range(NUM_LANES):
            history = list(self._vehicle_history[i])
            if history:
                ax.plot(
                    steps[-len(history):],
                    history,
                    label=LANE_NAMES[i],
                    color=colors[i % len(colors)],
                    linewidth=2,
                )
        if any(list(self._vehicle_history[i]) for i in range(NUM_LANES)):
            ax.legend(facecolor="#16213e", labelcolor="white")
        else:
            ax.text(0.5, 0.5, "Collecting data…", ha="center", va="center",
                    transform=ax.transAxes, color="white")

    def _plot_density(self, ax, vehicle_counts, db=None):
        from config import LANE_CAPACITY
        ax.set_facecolor("#16213e")
        ax.set_title("Traffic Density (%)", color="white", fontsize=11)
        ax.set_xlabel("Lane", color="white")
        ax.set_ylabel("Density %", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_ylim(0, 105)

        if db is not None:
            try:
                summary = db.get_traffic_summary()
                if not summary.empty:
                    ax.bar(
                        summary["lane_name"],
                        summary["avg_density"],
                        color="#1e90ff",
                        edgecolor="white",
                    )
                    ax.set_title("Avg Traffic Density (%) — last hour", color="white", fontsize=11)
                    return
            except Exception:
                pass

        # Fall back to current-step density
        densities = [min(vc / LANE_CAPACITY, 1.0) * 100 for vc in vehicle_counts]
        ax.bar(LANE_NAMES, densities, color="#1e90ff", edgecolor="white")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, path: str = "traffic_dashboard.png") -> None:
        """Save the most recent rendered figure to *path*."""
        plt.savefig(path, bbox_inches="tight", dpi=100)
        plt.close("all")
