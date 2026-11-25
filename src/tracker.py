"""
Task 6 & Task 8: Object Tracking & Selected Object Isolation
============================================================
Integrates ByteTrack multi-object tracking to assign persistent IDs across frames.
Allows isolating and tracking only the user-selected object ID.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from classifier import ObjectClassifier


@dataclass
class TrackedObject:
    """Represents an active tracked object instance in the current frame."""
    track_id: int
    category: str
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]

    @property
    def centroid(self) -> Tuple[float, float]:
        """Calculates center of mass (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Calculates bottom center coordinate (ground contact)."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, y2

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


class ByteTrackTracker:
    """
    Multi-object tracker utilizing ByteTrack algorithm for persistent ID assignment.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        classifier: Optional[ObjectClassifier] = None,
    ):
        self.model_path = model_path
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classifier = classifier or ObjectClassifier()

        # Load YOLO model
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        # Keep record of all active track IDs observed so far
        self.all_seen_tracks: Dict[int, Dict] = {}

    def reset(self):
        """Reset tracking state for a new video."""
        self.all_seen_tracks.clear()
        self.model = YOLO(self.model_path)

    def update(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        Processes a frame, updates ByteTrack tracker state, and returns active TrackedObjects.
        """
        results = self.model.track(
            source=frame,
            tracker=self.tracker_type,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        active_tracks: List[TrackedObject] = []
        if not results or len(results) == 0:
            return active_tracks

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.boxes.id is None:
            return active_tracks

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        track_ids = r.boxes.id.cpu().numpy().astype(int)

        for bbox, conf, cls_idx, t_id in zip(boxes_xyxy, confs, classes, track_ids):
            raw_name = self.class_names.get(cls_idx, str(cls_idx))

            # Filter according to classifier taxonomy
            if not self.classifier.is_allowed(raw_name):
                continue

            category = self.classifier.map_category(raw_name)
            obj = TrackedObject(
                track_id=int(t_id),
                category=category,
                class_name=raw_name,
                confidence=float(conf),
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            )
            active_tracks.append(obj)

            # Record track profile
            if t_id not in self.all_seen_tracks:
                self.all_seen_tracks[t_id] = {
                    "id": t_id,
                    "category": category,
                    "class_name": raw_name,
                    "first_seen_frame": len(self.all_seen_tracks),
                    "detections_count": 0,
                    "conf_sum": 0.0,
                    "initial_centroid": obj.centroid,
                    "last_centroid": obj.centroid,
                }
            self.all_seen_tracks[t_id]["detections_count"] += 1
            self.all_seen_tracks[t_id]["conf_sum"] += obj.confidence
            self.all_seen_tracks[t_id]["last_centroid"] = obj.centroid

        return active_tracks


if __name__ == "__main__":
    import cv2
    import os
    from config import SAMPLE_VIDEOS_DIR

    tracker = ByteTrackTracker(conf_threshold=0.30)
    video_path = os.path.join(SAMPLE_VIDEOS_DIR, "pedestrians_walk.mp4")

    cap = cv2.VideoCapture(video_path)
    print("Testing ByteTrackTracker on pedestrian sample...")
    for f in range(25):
        ret, frame = cap.read()
        if not ret:
            break
        tracks = tracker.update(frame)
        if tracks:
            print(f"Frame {f:02d}: {len(tracks)} active tracks -> {[f'ID:{t.track_id} ({t.category})' for t in tracks]}")
    cap.release()
