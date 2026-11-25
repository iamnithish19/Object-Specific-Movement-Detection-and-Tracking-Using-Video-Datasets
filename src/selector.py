"""
Task 7: User Object Selection
=============================
Allows the user to select a specific object ID using:
1. Command-line argument (non-blocking)
2. Interactive terminal prompt with active object catalog
3. Interactive OpenCV mouse-click graphical selection
4. Automated smart selection (longest track or highest displacement)
"""

from typing import Dict, List, Optional
import cv2
import numpy as np


class ObjectSelector:
    """Manages object ID selection across interactive and automated modes."""

    def __init__(
        self,
        target_id: Optional[int] = None,
        target_class: Optional[str] = None,
        interactive_cli: bool = False,
        interactive_gui: bool = False,
    ):
        self.target_id: Optional[int] = target_id
        self.target_class: Optional[str] = target_class.lower() if target_class else None
        self.interactive_cli: bool = interactive_cli
        self.interactive_gui: bool = interactive_gui
        self.selected_id: Optional[int] = target_id

    def is_selected(self) -> bool:
        """Check if an object ID has been confirmed."""
        return self.selected_id is not None

    def prompt_cli_selection(self, seen_tracks: Dict[int, Dict]) -> int:
        """
        Displays available candidates in the console and prompts user to pick an ID.
        """
        print("\n" + "=" * 62)
        print(" TASK 7: ACTIVE OBJECT CANDIDATE SELECTION ")
        print("=" * 62)
        print(f"{'ID':<6} | {'Category':<10} | {'Detections':<12} | {'Avg Conf':<10} | {'Position (cx, cy)'}")
        print("-" * 62)

        valid_ids = []
        for t_id, info in seen_tracks.items():
            valid_ids.append(t_id)
            avg_conf = (info["conf_sum"] / info["detections_count"]) * 100 if info["detections_count"] > 0 else 0
            cx, cy = int(info["last_centroid"][0]), int(info["last_centroid"][1])
            print(f"{t_id:<6} | {info['category']:<10} | {info['detections_count']:<12} | {avg_conf:5.1f}%     | ({cx}, {cy})")

        print("-" * 62)
        while True:
            try:
                raw_input = input(f"Enter target object ID to track exclusively (options: {valid_ids}): ").strip()
                val = int(raw_input)
                if val in valid_ids:
                    self.selected_id = val
                    print(f"[OK] Selected target ID: {self.selected_id} ({seen_tracks[val]['category']})")
                    return self.selected_id
                else:
                    print(f"Invalid ID '{val}'. Please select from: {valid_ids}")
            except (ValueError, EOFError):
                # Fallback to first available if input stream closed (e.g. non-interactive runner)
                fallback = valid_ids[0] if valid_ids else 1
                self.selected_id = fallback
                print(f"[Non-interactive environment detected] Automatically selected target ID: {fallback}")
                return self.selected_id

    def select_via_mouse_click(self, frame: np.ndarray, active_tracks: List) -> Optional[int]:
        """
        Displays frame in an OpenCV window and lets user click on an object to select it.
        """
        if not active_tracks:
            return None

        preview = frame.copy()
        # Draw bounding boxes with candidate numbers
        for obj in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                preview,
                f"CLICK ID: {obj.track_id} ({obj.category})",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        instructions = "CLICK on an object box to select ID | Press ENTER/SPACE for default"
        cv2.putText(preview, instructions, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        clicked_id = [None]

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for obj in active_tracks:
                    x1, y1, x2, y2 = obj.bbox
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        clicked_id[0] = obj.track_id
                        print(f"Mouse selected object ID: {obj.track_id} ({obj.category}) at ({x}, {y})")
                        break

        window_title = "Task 7: Click Object to Select Track ID"
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_title, on_mouse)

            while clicked_id[0] is None:
                cv2.imshow(window_title, preview)
                key = cv2.waitKey(20) & 0xFF
                if key in [13, 32]:  # Enter or Space
                    clicked_id[0] = active_tracks[0].track_id
                    break
                elif key == 27:  # ESC
                    clicked_id[0] = active_tracks[0].track_id
                    break

            cv2.destroyWindow(window_title)
        except cv2.error:
            # Headless environment where cv2.imshow is unavailable
            clicked_id[0] = active_tracks[0].track_id

        self.selected_id = clicked_id[0]
        return self.selected_id

    def auto_select(self, seen_tracks: Dict[int, Dict]) -> int:
        """
        Picks the best target ID automatically:
        - If target_class is specified, filters by class first
        - Chooses track with highest detection count / longevity
        """
        candidates = list(seen_tracks.values())
        if self.target_class:
            filtered = [c for c in candidates if c["category"] == self.target_class]
            if filtered:
                candidates = filtered

        if not candidates:
            raise RuntimeError("No candidate objects detected to select from.")

        # Sort by detection count descending
        candidates.sort(key=lambda x: x["detections_count"], reverse=True)
        chosen = candidates[0]["id"]
        self.selected_id = chosen
        return chosen
