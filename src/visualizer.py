"""
Task 12: Video Visualizer
=========================
Renders bounding boxes, IDs, trajectory path, velocity heading arrows,
and a real-time HUD telemetry dashboard directly onto video frames.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from motion_analyzer import MotionState
from tracker import TrackedObject


class VideoVisualizer:
    """Renders comprehensive tracking visualizations on image frames."""

    def __init__(
        self,
        trail_length: int = 80,
        show_hud: bool = False,
        dim_background_objects: bool = True,
        arrow_scale: float = 0.25,
        target_color: Tuple[int, int, int] = (0, 255, 0),        # Vibrant Green
        trajectory_color: Tuple[int, int, int] = (0, 215, 255),  # Gold/Amber
        background_color: Tuple[int, int, int] = (80, 80, 80),   # Dim Gray
    ):
        self.trail_length = trail_length
        self.show_hud = show_hud
        self.dim_background_objects = dim_background_objects
        self.arrow_scale = arrow_scale
        self.target_color = target_color
        self.trajectory_color = trajectory_color
        self.background_color = background_color

    def draw_corner_brackets(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        color: Tuple[int, int, int],
        thickness: int = 2,
        length: int = 15,
    ):
        """Draws tactical corner brackets around a bounding box for sleek targeting visual."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Clamp length
        w = x2 - x1
        h = y2 - y1
        l = min(length, w // 3, h // 3)

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + l, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + l), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - l, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y2 if y1 + l > y2 else y1 + l), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + l, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - l), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - l, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - l), color, thickness)

    def draw_trajectory(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
    ):
        """Draws trailing breadcrumb polyline with progressive thickness/opacity."""
        if len(points) < 2:
            return

        visible_points = points[-self.trail_length :]
        num_pts = len(visible_points)

        for i in range(1, num_pts):
            p1 = visible_points[i - 1]
            p2 = visible_points[i]

            # Progression ratio: 0.1 at tail, 1.0 at head
            ratio = i / float(num_pts)
            thickness = max(1, int(1 + 2 * ratio))

            # Color fade towards brighter amber
            b = int(self.trajectory_color[0] * ratio)
            g = int(self.trajectory_color[1] * ratio)
            r = int(self.trajectory_color[2] * ratio)
            color = (b, g, r)

            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    def draw_velocity_arrow(
        self,
        frame: np.ndarray,
        centroid: Tuple[float, float],
        speed: float,
        orientation_deg: float,
        is_stationary: bool,
    ):
        """Draws heading arrow originating from centroid pointing in direction of movement."""
        if is_stationary:
            return

        cx, cy = int(centroid[0]), int(centroid[1])
        # Length clamped between 25px and 80px based on speed
        arrow_len = np.clip(speed * self.arrow_scale, 25.0, 85.0)

        # Convert orientation angle to screen vector (math vy = -screen_dy)
        rad = np.radians(orientation_deg)
        end_x = int(cx + arrow_len * np.cos(rad))
        end_y = int(cy - arrow_len * np.sin(rad))

        cv2.arrowedLine(
            frame,
            (cx, cy),
            (end_x, end_y),
            color=(0, 255, 255),  # Yellow arrow
            thickness=3,
            tipLength=0.30,
            line_type=cv2.LINE_AA,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        selected_id: int,
        category: str,
        motion_state: Optional[MotionState],
        frame_idx: int,
        total_frames: int,
        fps: float,
    ):
        """Draws a semi-transparent dark HUD telemetry panel."""
        h, w = frame.shape[:2]
        panel_w = 330
        panel_h = 190
        panel_x = 16
        panel_y = 16

        # Draw dark frosted overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 24, 28),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (0, 215, 255),
            1,
        )

        # Header
        cv2.putText(
            frame,
            f"TARGET LOCKED [ID #{selected_id}] {category.upper()}",
            (panel_x + 12, panel_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (panel_x + 10, panel_y + 32),
            (panel_x + panel_w - 10, panel_y + 32),
            (80, 90, 100),
            1,
        )

        if motion_state is not None:
            speed_txt = f"{motion_state.speed_smoothed:.1f} px/s"
            if motion_state.speed_kmh is not None:
                speed_txt += f" ({motion_state.speed_kmh:.1f} km/h)"

            lines = [
                f"Position (X, Y): ({motion_state.x:.0f}, {motion_state.y:.0f}) px",
                f"Displacement: dX={motion_state.dx:+.1f}, dY={motion_state.dy:+.1f} px",
                f"Speed: {speed_txt}",
                f"Heading: {motion_state.direction} ({motion_state.orientation_deg:.1f} deg)",
                f"Net Dist: {np.hypot(motion_state.net_dx, motion_state.net_dy):.1f} px | Total: {motion_state.cumulative_distance:.1f} px",
                f"Frame: {frame_idx + 1}/{total_frames} | Stream FPS: {fps:.1f}",
            ]
        else:
            lines = [
                "Status: Searching / Object out of view",
                f"Frame: {frame_idx + 1}/{total_frames} | Stream FPS: {fps:.1f}",
            ]

        y_offset = panel_y + 54
        for line in lines:
            cv2.putText(
                frame,
                line,
                (panel_x + 12, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (230, 240, 250),
                1,
                cv2.LINE_AA,
            )
            y_offset += 21

    def render(
        self,
        frame: np.ndarray,
        all_active_tracks: List[TrackedObject],
        selected_id: Optional[int],
        trajectory_points: List[Tuple[int, int]],
        motion_state: Optional[MotionState],
        frame_idx: int,
        total_frames: int,
        fps: float,
    ) -> np.ndarray:
        """Assembles all visualization layers onto the output video frame."""
        annotated = frame.copy()

        # 1. Render background objects
        selected_obj: Optional[TrackedObject] = None
        for obj in all_active_tracks:
            if obj.track_id == selected_id:
                selected_obj = obj
            else:
                # Dim background object
                x1, y1, x2, y2 = [int(v) for v in obj.bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), self.background_color, 1)
                cv2.putText(
                    annotated,
                    f"#{obj.track_id} {obj.category}",
                    (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    self.background_color,
                    1,
                    cv2.LINE_AA,
                )

        # 2. Render trajectory of selected object
        if trajectory_points:
            self.draw_trajectory(annotated, trajectory_points)

        # 3. Render selected object with vibrant target aesthetics
        if selected_obj is not None:
            x1, y1, x2, y2 = [int(v) for v in selected_obj.bbox]
            cx, cy = [int(v) for v in selected_obj.centroid]

            # Primary bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.target_color, 2)
            self.draw_corner_brackets(annotated, selected_obj.bbox, self.target_color, thickness=3)

            # Centroid bullseye
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(annotated, (cx, cy), 9, self.target_color, 2)

            # Target badge tag above box
            badge = f"TRACKED #{selected_obj.track_id} {selected_obj.category.upper()} ({int(selected_obj.confidence*100)}%)"
            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - th - 8)),
                (x1 + tw + 8, y1),
                self.target_color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                badge,
                (x1 + 4, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Heading arrow
            if motion_state is not None:
                is_stat = motion_state.direction == "Stationary"
                self.draw_velocity_arrow(
                    annotated,
                    selected_obj.centroid,
                    motion_state.speed_smoothed,
                    motion_state.orientation_deg,
                    is_stat,
                )

        # 4. Render HUD
        if self.show_hud and selected_id is not None:
            cat = selected_obj.category if selected_obj else "Unknown"
            self.draw_hud(
                annotated,
                selected_id,
                cat,
                motion_state,
                frame_idx,
                total_frames,
                fps,
            )

        return annotated
