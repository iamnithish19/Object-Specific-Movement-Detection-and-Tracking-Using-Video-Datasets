"""
Task 14: Tracking Accuracy Evaluation & Result Analysis
======================================================
Analyzes tracking performance, track stability, detection confidence,
path tortuosity, and kinematic metrics. Generates JSON, CSV, and Markdown reports.
"""

import csv
import json
import os
from collections import Counter
from typing import Dict, List, Optional
import numpy as np

from motion_analyzer import MotionState


class TrackingEvaluator:
    """Evaluates tracking performance and outputs comprehensive diagnostic reports."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(
        self,
        history: List[MotionState],
        target_id: int,
        category: str,
        video_metadata: dict,
        confidence_history: List[float],
        lost_frames_count: int = 0,
    ) -> dict:
        """Computes comprehensive evaluation metrics."""
        if not history:
            return {
                "target_id": target_id,
                "category": category,
                "video_file": video_metadata.get("filename", "unknown"),
                "video_resolution": f"{video_metadata.get('width', 0)}x{video_metadata.get('height', 0)}",
                "video_fps": video_metadata.get("fps", 30.0),
                "tracking_coverage": {
                    "total_video_frames": video_metadata.get("total_frames", 0),
                    "tracked_frames": 0,
                    "lost_or_untracked_frames": lost_frames_count,
                    "coverage_percentage": 0.0,
                    "duration_seconds": 0.0,
                },
                "detection_quality": {
                    "mean_confidence": 0.0,
                    "min_confidence": 0.0,
                    "max_confidence": 0.0,
                    "std_confidence": 0.0,
                },
                "motion_kinematics": {
                    "total_path_distance_px": 0.0,
                    "net_displacement_px": 0.0,
                    "path_efficiency_ratio": 0.0,
                    "mean_speed_px_s": 0.0,
                    "peak_speed_px_s": 0.0,
                    "median_speed_px_s": 0.0,
                    "speed_std_dev_px_s": 0.0,
                    "acceleration_smoothness_index": 0.0,
                },
                "directional_distribution_pct": {"Stationary": 100.0},
            }

        total_vid_frames = video_metadata.get("total_frames", len(history))
        fps = video_metadata.get("fps", 30.0)
        tracked_frames = len(history)
        coverage_pct = (tracked_frames / total_vid_frames * 100.0) if total_vid_frames > 0 else 0.0

        speeds = [s.speed_smoothed for s in history]
        inst_speeds = [s.speed_px_sec for s in history]
        total_dist = history[-1].cumulative_distance
        net_dist = float(np.hypot(history[-1].net_dx, history[-1].net_dy))
        efficiency = (net_dist / total_dist) if total_dist > 0 else 1.0

        # Directions breakdown
        directions = [s.direction for s in history]
        dir_counts = Counter(directions)
        dir_pct = {k: round(v / len(directions) * 100.0, 1) for k, v in dir_counts.items()}

        # Confidence statistics
        conf_stats = {
            "mean_confidence": round(float(np.mean(confidence_history)), 3) if confidence_history else 0.0,
            "min_confidence": round(float(np.min(confidence_history)), 3) if confidence_history else 0.0,
            "max_confidence": round(float(np.max(confidence_history)), 3) if confidence_history else 0.0,
            "std_confidence": round(float(np.std(confidence_history)), 3) if confidence_history else 0.0,
        }

        # Track continuity & smoothness
        # Smoothness: variance of second derivative (acceleration jitter)
        accelerations = [abs(speeds[i] - speeds[i - 1]) for i in range(1, len(speeds))] if len(speeds) > 1 else [0.0]
        smoothness_index = round(float(np.mean(accelerations)), 2)

        metrics = {
            "target_id": target_id,
            "category": category,
            "video_file": video_metadata.get("filename", "unknown"),
            "video_resolution": f"{video_metadata.get('width', 0)}x{video_metadata.get('height', 0)}",
            "video_fps": fps,
            "tracking_coverage": {
                "total_video_frames": total_vid_frames,
                "tracked_frames": tracked_frames,
                "lost_or_untracked_frames": lost_frames_count,
                "coverage_percentage": round(coverage_pct, 2),
                "duration_seconds": round(tracked_frames / fps, 2),
            },
            "detection_quality": conf_stats,
            "motion_kinematics": {
                "total_path_distance_px": round(total_dist, 2),
                "net_displacement_px": round(net_dist, 2),
                "path_efficiency_ratio": round(efficiency, 3),
                "mean_speed_px_s": round(float(np.mean(speeds)), 2),
                "peak_speed_px_s": round(float(np.max(inst_speeds)), 2),
                "median_speed_px_s": round(float(np.median(speeds)), 2),
                "speed_std_dev_px_s": round(float(np.std(speeds)), 2),
                "acceleration_smoothness_index": smoothness_index,
            },
            "directional_distribution_pct": dir_pct,
        }

        return metrics

    def export_csv_telemetry(
        self,
        history: List[MotionState],
        target_id: int,
        video_name: str,
    ) -> str:
        """Exports frame-by-frame motion telemetry to a CSV file."""
        base_name = os.path.splitext(video_name)[0]
        csv_filename = f"telemetry_id_{target_id}_{base_name}.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_idx",
                "timestamp_sec",
                "x_px",
                "y_px",
                "dx_px",
                "dy_px",
                "step_distance_px",
                "net_dx_px",
                "net_dy_px",
                "cumulative_distance_px",
                "speed_instant_px_s",
                "speed_smoothed_px_s",
                "orientation_deg",
                "direction",
                "speed_kmh",
            ])
            for s in history:
                writer.writerow([
                    s.frame_idx,
                    round(s.timestamp_sec, 3),
                    s.x,
                    s.y,
                    s.dx,
                    s.dy,
                    s.step_distance,
                    s.net_dx,
                    s.net_dy,
                    s.cumulative_distance,
                    s.speed_px_sec,
                    s.speed_smoothed,
                    s.orientation_deg,
                    s.direction,
                    s.speed_kmh if s.speed_kmh is not None else "",
                ])

        print(f"[OK] Telemetry CSV exported: {csv_path}")
        return csv_path

    def export_markdown_report(
        self,
        metrics: dict,
        target_id: int,
        video_name: str,
    ) -> str:
        """Exports a formatted Markdown evaluation report."""
        base_name = os.path.splitext(video_name)[0]
        md_filename = f"evaluation_report_id_{target_id}_{base_name}.md"
        md_path = os.path.join(self.output_dir, md_filename)

        cov = metrics.get("tracking_coverage", {})
        kin = metrics.get("motion_kinematics", {})
        det = metrics.get("detection_quality", {})
        dir_dist = metrics.get("directional_distribution_pct", {})

        md_content = f"""# Object-Specific Tracking Evaluation & Analysis Report

## Executive Summary
- **Target Object**: ID `#{metrics.get('target_id')}` ({metrics.get('category', '').upper()})
- **Video Source**: `{metrics.get('video_file')}` ({metrics.get('video_resolution')}, {metrics.get('video_fps')} FPS)
- **Tracking Continuity**: {cov.get('coverage_percentage')}% ({cov.get('tracked_frames')}/{cov.get('total_video_frames')} frames)
- **Mean Detection Confidence**: {det.get('mean_confidence', 0) * 100:.1f}%

---

## 1. Tracking Continuity & Stability Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Video Frames** | `{cov.get('total_video_frames')}` | Total duration frames of the input stream |
| **Tracked Frames** | `{cov.get('tracked_frames')}` | Frames where target ID was actively tracked |
| **Lost / Untracked Frames** | `{cov.get('lost_or_untracked_frames')}` | Frames target was occluded or absent |
| **Coverage Ratio** | **{cov.get('coverage_percentage')}%** | Persistence ratio across video timeline |
| **Track Duration** | `{cov.get('duration_seconds')} s` | Active tracked time in seconds |

---

## 2. Detection Quality & Confidence
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Confidence** | `{det.get('mean_confidence')}` | Average YOLO classification certainty |
| **Min Confidence** | `{det.get('min_confidence')}` | Lowest confidence observed |
| **Max Confidence** | `{det.get('max_confidence')}` | Peak confidence observed |
| **Confidence Std Dev**| `{det.get('std_confidence')}` | Stability of detection certainty |

---

## 3. Kinematic & Motion Analysis (Tasks 9, 10, 11)
| Kinematic Metric | Value | Units | Interpretation |
| :--- | :--- | :--- | :--- |
| **Total Path Distance** | `{kin.get('total_path_distance_px')}` | px | Cumulative integrated trajectory distance |
| **Net Displacement** | `{kin.get('net_displacement_px')}` | px | Euclidean distance from origin to end position |
| **Path Efficiency (Tortuosity)** | **{kin.get('path_efficiency_ratio')}** | ratio | `1.0` = straight line, `<0.7` = winding/curved |
| **Mean Smoothed Speed** | `{kin.get('mean_speed_px_s')}` | px/s | Average velocity across track |
| **Peak Instantaneous Speed** | `{kin.get('peak_speed_px_s')}` | px/s | Maximum velocity spike detected |
| **Median Speed** | `{kin.get('median_speed_px_s')}` | px/s | Central tendency of object velocity |
| **Acceleration Jitter Index** | `{kin.get('acceleration_smoothness_index')}` | px/s² | Trajectory smoothness measure |

---

## 4. Directional Distribution
Breakdown of heading sectors during motion:
"""
        for d, pct in dir_dist.items():
            md_content += f"- **{d}**: {pct}%\n"

        with open(md_path, mode="w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"[OK] Evaluation Markdown report exported: {md_path}")
        return md_path
