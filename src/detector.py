"""
Task 4: Object Detection with YOLO
==================================
Performs deep-learning based object detection on video frames using YOLO.
Extracts bounding boxes, class labels, and detection confidence scores.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from classifier import ObjectClassifier


@dataclass
class Detection:
    """Represents a single detected object in a video frame."""
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float                        # 0.0 to 1.0
    class_id: int                            # Raw class index
    class_name: str                          # Raw class name (e.g. 'car', 'person')
    category: str                            # Mapped category (e.g. 'car', 'person', 'drone')

    @property
    def centroid(self) -> Tuple[float, float]:
        """Calculates center of mass (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Calculates bottom-center contact point (useful for ground plane tracking)."""
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


class YOLODetector:
    """YOLO-based object detector."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        classifier: Optional[ObjectClassifier] = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classifier = classifier or ObjectClassifier()

        # Load YOLO model
        print(f"Loading YOLO model from: {model_path} ...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # Dict {id: name}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs YOLO object detection on a single image/frame.
        Returns filtered list of Detection instances.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_idx in zip(boxes_xyxy, confs, classes):
            raw_name = self.class_names.get(cls_idx, str(cls_idx))

            # Filter according to classifier taxonomy
            if not self.classifier.is_allowed(raw_name):
                continue

            category = self.classifier.map_category(raw_name)
            detections.append(
                Detection(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=float(conf),
                    class_id=int(cls_idx),
                    class_name=raw_name,
                    category=category,
                )
            )

        return detections


if __name__ == "__main__":
    import cv2
    import os
    from config import SAMPLE_VIDEOS_DIR

    detector = YOLODetector(conf_threshold=0.35)
    sample_video = os.path.join(SAMPLE_VIDEOS_DIR, "traffic_cars.mp4")

    if os.path.exists(sample_video):
        cap = cv2.VideoCapture(sample_video)
        ret, frame = cap.read()
        cap.release()
        if ret:
            dets = detector.detect(frame)
            print(f"Detected {len(dets)} objects in first frame:")
            for d in dets:
                print(f" - {d.category} (raw: {d.class_name}) conf={d.confidence:.2f} bbox={[int(x) for x in d.bbox]}")
