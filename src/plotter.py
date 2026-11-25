"""
Task 12: Analytical Motion Graphs & Visual Dashboard
===================================================
Generates high-resolution Matplotlib analytical figures visualizing:
1. 2D Spatial Trajectory (image coordinate plane with start/end markers)
2. Speed vs. Time (instantaneous, smoothed, mean & peak markers)
3. Instantaneous X & Y Pixel Displacements over Time
4. Heading Orientation Angle & Compass Sector Distribution
"""

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np

from motion_analyzer import MotionState


class MotionPlotter:
    """Generates analytical multi-panel motion and trajectory charts."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_dashboard(
        self,
        history: List[MotionState],
        target_id: int,
        category: str,
        video_name: str,
        frame_width: int,
        frame_height: int,
        save_filename: Optional[str] = None,
    ) -> str:
        """
        Creates and saves a comprehensive 4-panel analytical chart.
        """
        if not history:
            print("[Warning] No motion history to plot.")
            return ""

        timestamps = [s.timestamp_sec for s in history]
        frames = [s.frame_idx for s in history]
        x_coords = [s.x for s in history]
        y_coords = [s.y for s in history]
        dx_vals = [s.dx for s in history]
        dy_vals = [s.dy for s in history]
        speeds = [s.speed_px_sec for s in history]
        smooth_speeds = [s.speed_smoothed for s in history]
        headings = [s.orientation_deg for s in history]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Object-Specific Movement & Trajectory Analysis\nTarget ID #{target_id} ({category.upper()}) | Video: {video_name}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # ----------------------------------------------------
        # Panel 1: 2D Spatial Trajectory Map (Image Coordinates)
        # ----------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_title("1. 2D Trajectory Map on Frame Plane", fontsize=12, fontweight="bold")
        # Invert Y to match image coordinates (0,0 is top-left)
        ax1.invert_yaxis()
        ax1.set_xlim(0, frame_width)
        ax1.set_ylim(frame_height, 0)
        ax1.set_xlabel("X Pixel Position")
        ax1.set_ylabel("Y Pixel Position")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Plot trajectory line
        ax1.plot(x_coords, y_coords, color="#0284c7", linewidth=2.5, label="Trajectory Path", zorder=3)

        # Mark Start and End points
        ax1.scatter(x_coords[0], y_coords[0], color="#16a34a", s=130, marker="o", edgecolors="black", label="Start", zorder=4)
        ax1.scatter(x_coords[-1], y_coords[-1], color="#dc2626", s=130, marker="X", edgecolors="black", label="End", zorder=4)

        # Subsample quiver arrows along path
        arrow_step = max(1, len(history) // 10)
        for i in range(0, len(history) - 1, arrow_step):
            ax1.annotate(
                "",
                xy=(x_coords[min(i + arrow_step, len(history) - 1)], y_coords[min(i + arrow_step, len(history) - 1)]),
                xytext=(x_coords[i], y_coords[i]),
                arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.5),
            )
        ax1.legend(loc="upper right")

        # ----------------------------------------------------
        # Panel 2: Speed Profile vs. Time
        # ----------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_title("2. Speed Profile vs. Time", fontsize=12, fontweight="bold")
        ax2.plot(timestamps, speeds, color="#93c5fd", linewidth=1.2, alpha=0.7, label="Instantaneous Speed (px/s)")
        ax2.plot(timestamps, smooth_speeds, color="#1e3a8a", linewidth=2.2, label="Smoothed Speed (px/s)")

        mean_speed = float(np.mean(smooth_speeds))
        max_speed = float(np.max(smooth_speeds))
        ax2.axhline(mean_speed, color="#059669", linestyle="--", linewidth=1.5, label=f"Mean Speed ({mean_speed:.1f} px/s)")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Speed (pixels / second)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")

        # ----------------------------------------------------
        # Panel 3: X & Y Pixel Displacements over Time
        # ----------------------------------------------------
        ax3 = axes[1, 0]
        ax3.set_title("3. Frame-to-Frame Displacement (dX, dY)", fontsize=12, fontweight="bold")
        ax3.plot(frames, dx_vals, color="#0284c7", linewidth=1.5, label="dX (horizontal px)")
        ax3.plot(frames, dy_vals, color="#c026d3", linewidth=1.5, label="dY (vertical px)")
        ax3.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)
        ax3.set_xlabel("Frame Number")
        ax3.set_ylabel("Displacement per Frame (pixels)")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper right")

        # ----------------------------------------------------
        # Panel 4: Heading Orientation Angle over Time
        # ----------------------------------------------------
        ax4 = axes[1, 1]
        ax4.set_title("4. Movement Orientation & Heading Angle", fontsize=12, fontweight="bold")
        ax4.plot(timestamps, headings, color="#d97706", linewidth=1.8, label="Orientation Angle (deg)")
        ax4.set_ylim(-10, 370)
        ax4.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax4.set_yticklabels(["0° E", "45° NE", "90° N", "135° NW", "180° W", "225° SW", "270° S", "315° SE", "360° E"])
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel("Heading Angle")
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_filename is None:
            save_filename = f"tracking_dashboard_id_{target_id}_{os.path.splitext(video_name)[0]}.png"
        output_filepath = os.path.join(self.output_dir, save_filename)
        fig.savefig(output_filepath, dpi=200)
        plt.close(fig)

        print(f"[OK] Analytical dashboard generated and saved to: {output_filepath}")
        return output_filepath


if __name__ == "__main__":
    from motion_analyzer import MotionAnalyzer

    analyzer = MotionAnalyzer(fps=30.0)
    for f in range(60):
        t = f / 30.0
        cx = 200 + f * 5 + 10 * np.sin(f * 0.2)
        cy = 150 + f * 3 + 8 * np.cos(f * 0.2)
        analyzer.update(f, t, (cx, cy))

    plotter = MotionPlotter()
    out_img = plotter.generate_dashboard(
        analyzer.history,
        target_id=1,
        category="car",
        video_name="synthetic_test.mp4",
        frame_width=640,
        frame_height=480,
    )
    print("Plotter verification complete:", out_img)
