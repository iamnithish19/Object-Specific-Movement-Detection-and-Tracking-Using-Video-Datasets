"""
Tasks 9, 10, 11: Motion Analysis, Displacement, Direction, Speed & Trajectory
=============================================================================
Calculates:
- Task 9:  X and Y pixel displacement (frame-by-frame & net cumulative)
- Task 10: Movement direction (8-compass) and orientation angle (degrees)
- Task 11: Pixel-based speed (instantaneous & smoothed) and trajectory generation
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
import numpy as np


@dataclass
class MotionState:
    """Telemetry state of the tracked object at a specific frame."""
    frame_idx: int
    timestamp_sec: float
    x: float
    y: float
    dx: float                    # Task 9: instantaneous X displacement in px
    dy: float                    # Task 9: instantaneous Y displacement in px
    step_distance: float         # Euclidean displacement in current frame
    net_dx: float                # Task 9: total X displacement from start
    net_dy: float                # Task 9: total Y displacement from start
    cumulative_distance: float   # Total accumulated path length
    speed_px_sec: float          # Task 11: instantaneous speed in px/s
    speed_smoothed: float        # Task 11: rolling smoothed speed
    orientation_deg: float       # Task 10: heading angle [0..360) deg
    direction: str               # Task 10: 8-compass direction label
    speed_kmh: Optional[float] = None  # Optional real-world speed if calibrated


COMPASS_DIRECTIONS = [
    ("East", 0.0),
    ("North-East", 45.0),
    ("North", 90.0),
    ("North-West", 135.0),
    ("West", 180.0),
    ("South-West", 225.0),
    ("South", 270.0),
    ("South-East", 315.0),
]


class MotionAnalyzer:
    """Calculates motion telemetry, vectors, speed, and trajectory for a selected object."""

    def __init__(
        self,
        fps: float = 30.0,
        smoothing_window: int = 5,
        stationary_threshold_px: float = 1.5,
        pixels_per_meter: Optional[float] = None,
    ):
        self.fps = fps if fps > 0 else 30.0
        self.smoothing_window = max(1, smoothing_window)
        self.stationary_threshold_px = stationary_threshold_px
        self.pixels_per_meter = pixels_per_meter

        # State storage
        self.history: List[MotionState] = []
        self.trajectory_points: List[Tuple[int, int]] = []
        self._recent_dx: Deque[float] = deque(maxlen=self.smoothing_window)
        self._recent_dy: Deque[float] = deque(maxlen=self.smoothing_window)
        self._recent_speed: Deque[float] = deque(maxlen=self.smoothing_window)

        self.initial_pos: Optional[Tuple[float, float]] = None
        self.last_pos: Optional[Tuple[float, float]] = None
        self.cumulative_dist: float = 0.0
        self.last_heading_deg: float = 0.0

    def reset(self):
        """Reset motion analytics."""
        self.history.clear()
        self.trajectory_points.clear()
        self._recent_dx.clear()
        self._recent_dy.clear()
        self._recent_speed.clear()
        self.initial_pos = None
        self.last_pos = None
        self.cumulative_dist = 0.0
        self.last_heading_deg = 0.0

    @staticmethod
    def calculate_angle_and_direction(dx: float, dy: float, threshold: float = 1.5) -> Tuple[float, str]:
        """
        Calculates mathematical orientation angle in degrees and 8-compass direction.
        Note: Image Y increases downwards, so Cartesian vy = -dy.
        """
        magnitude = np.hypot(dx, dy)
        if magnitude < threshold:
            return 0.0, "Stationary"

        # Cartesian angle: 0 deg = East, 90 deg = North, 180 deg = West, 270 deg = South
        angle_rad = np.arctan2(-dy, dx)
        angle_deg = np.degrees(angle_rad) % 360.0

        # Compass sector resolution (each sector is 45 degrees, offset by 22.5 degrees)
        sector_idx = int(np.floor((angle_deg + 22.5) / 45.0)) % 8
        direction_label = COMPASS_DIRECTIONS[sector_idx][0]

        return float(angle_deg), direction_label

    def update(
        self,
        frame_idx: int,
        timestamp_sec: float,
        centroid: Tuple[float, float],
    ) -> MotionState:
        """
        Ingests the current frame object position and computes all motion metrics.
        """
        curr_x, curr_y = centroid

        if self.initial_pos is None:
            self.initial_pos = (curr_x, curr_y)
            self.last_pos = (curr_x, curr_y)
            dx = 0.0
            dy = 0.0
            step_dist = 0.0
            speed_px_sec = 0.0
            speed_smoothed = 0.0
            orientation_deg = 0.0
            direction = "Stationary"
        else:
            prev_x, prev_y = self.last_pos
            # Task 9: Calculate instantaneous displacement
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            step_dist = float(np.hypot(dx, dy))

            # Task 11: Calculate instantaneous speed (px/sec)
            speed_px_sec = step_dist * self.fps

            # Update sliding buffers
            self._recent_dx.append(dx)
            self._recent_dy.append(dy)
            self._recent_speed.append(speed_px_sec)

            # Smoothed vector and speed
            smooth_dx = float(np.mean(self._recent_dx))
            smooth_dy = float(np.mean(self._recent_dy))
            speed_smoothed = float(np.mean(self._recent_speed))

            # Task 10: Movement direction and orientation
            orientation_deg, direction = self.calculate_angle_and_direction(
                smooth_dx, smooth_dy, threshold=self.stationary_threshold_px
            )
            if direction != "Stationary":
                self.last_heading_deg = orientation_deg
            else:
                orientation_deg = self.last_heading_deg

            # Cumulative distance
            self.cumulative_dist += step_dist
            self.last_pos = (curr_x, curr_y)

        # Net displacement from starting coordinate
        net_dx = curr_x - self.initial_pos[0]
        net_dy = curr_y - self.initial_pos[1]

        # Optional metric conversion
        speed_kmh = None
        if self.pixels_per_meter is not None and self.pixels_per_meter > 0:
            speed_mps = speed_smoothed / self.pixels_per_meter
            speed_kmh = speed_mps * 3.6

        state = MotionState(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            x=round(curr_x, 2),
            y=round(curr_y, 2),
            dx=round(dx, 2),
            dy=round(dy, 2),
            step_distance=round(step_dist, 2),
            net_dx=round(net_dx, 2),
            net_dy=round(net_dy, 2),
            cumulative_distance=round(self.cumulative_dist, 2),
            speed_px_sec=round(speed_px_sec, 2),
            speed_smoothed=round(speed_smoothed, 2),
            orientation_deg=round(orientation_deg, 2),
            direction=direction,
            speed_kmh=round(speed_kmh, 2) if speed_kmh is not None else None,
        )

        self.history.append(state)
        self.trajectory_points.append((int(round(curr_x)), int(round(curr_y))))

        return state

    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Return the sequence of all trajectory coordinates."""
        return self.trajectory_points

    def get_latest_state(self) -> Optional[MotionState]:
        """Return the most recent motion telemetry state."""
        return self.history[-1] if self.history else None

    def get_summary_statistics(self) -> dict:
        """Compute aggregate motion metrics over the whole track."""
        if not self.history:
            return {}

        speeds = [s.speed_smoothed for s in self.history]
        steps = [s.step_distance for s in self.history]
        dur = self.history[-1].timestamp_sec - self.history[0].timestamp_sec

        net_dist = np.hypot(self.history[-1].net_dx, self.history[-1].net_dy)
        path_efficiency = (net_dist / self.cumulative_dist) if self.cumulative_dist > 0 else 1.0

        return {
            "total_frames_tracked": len(self.history),
            "tracking_duration_sec": round(dur, 2),
            "total_distance_px": round(self.cumulative_dist, 2),
            "net_displacement_px": round(float(net_dist), 2),
            "path_efficiency_ratio": round(float(path_efficiency), 3),
            "mean_speed_px_s": round(float(np.mean(speeds)), 2),
            "max_speed_px_s": round(float(np.max(speeds)), 2),
            "median_speed_px_s": round(float(np.median(speeds)), 2),
            "std_speed_px_s": round(float(np.std(speeds)), 2),
        }


if __name__ == "__main__":
    analyzer = MotionAnalyzer(fps=30.0)
    print("Testing MotionAnalyzer on synthetic trajectory...")
    for f in range(10):
        # Move diagonally right-down: +10 px X, +5 px Y per frame
        cx = 100 + f * 10
        cy = 100 + f * 5
        st = analyzer.update(frame_idx=f, timestamp_sec=f / 30.0, centroid=(cx, cy))
        print(f"F{f:02d}: pos=({st.x}, {st.y}) | dx={st.dx}, dy={st.dy} | "
              f"spd={st.speed_smoothed:.1f}px/s | dir={st.direction} ({st.orientation_deg:.1f} deg)")
    print("\nSummary:", analyzer.get_summary_statistics())
