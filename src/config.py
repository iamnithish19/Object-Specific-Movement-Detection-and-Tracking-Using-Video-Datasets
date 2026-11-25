"""
Configuration and Defaults for Object Detection & Tracking System
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_VIDEOS_DIR = os.path.join(DATA_DIR, "sample_videos")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")


@dataclass
class TrackerConfig:
    # Model settings
    model_name: str = "yolov8n.pt"
    tracker_type: str = "bytetrack.yaml"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.50

    # Target class filtering (None = track all classes)
    target_classes: List[str] = field(default_factory=lambda: ["person", "car", "truck", "bus", "motorcycle", "bicycle", "airplane"])

    # Motion analysis parameters
    smoothing_window: int = 5
    stationary_threshold_px: float = 1.5  # Below this displacement, object is considered stationary
    pixels_per_meter: float = None       # Optional calibration factor

    # Visualization settings
    trail_length: int = 80               # Number of historical points to render in trajectory
    show_hud: bool = True
    dim_background_objects: bool = True  # Visually subdue objects other than selected ID
    arrow_scale: float = 2.0             # Scale factor for velocity heading vector
    selected_box_color: Tuple[int, int, int] = (0, 255, 0)      # Bright Green (BGR)
    background_box_color: Tuple[int, int, int] = (80, 80, 80)   # Dim Gray (BGR)
    trajectory_color: Tuple[int, int, int] = (0, 215, 255)      # Vibrant Amber/Gold (BGR)
