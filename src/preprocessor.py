"""
Task 3: Video Frame Extraction & Preprocessing
==============================================
Handles video stream ingestion, frame extraction, resolution normalization,
contrast adjustment (CLAHE), and frame filtering.
"""

import os
from typing import Generator, Tuple, Dict, Any, Optional
import cv2
import numpy as np


class VideoPreprocessor:
    """Extracts frames from video files with customizable computer vision preprocessing."""

    def __init__(
        self,
        video_path: str,
        target_size: Optional[Tuple[int, int]] = None,
        apply_clahe: bool = False,
        apply_blur: bool = False,
        blur_kernel: int = 3,
    ):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        self.video_path = video_path
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.apply_blur = apply_blur
        self.blur_kernel = blur_kernel

        # Probe video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video stream: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0  # Safe default if container metadata missing

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0
        cap.release()

        # CLAHE instance (for contrast enhancement)
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def get_metadata(self) -> Dict[str, Any]:
        """Return video properties dictionary."""
        return {
            "path": self.video_path,
            "filename": os.path.basename(self.video_path),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
        }

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies preprocessing transformations to a raw video frame:
        1. Resizing (if target_size specified)
        2. Gaussian Blur (noise reduction)
        3. CLAHE (illumination normalization in LAB color space)
        """
        processed = frame.copy()

        # 1. Resize if required
        if self.target_size is not None:
            processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 2. Noise reduction
        if self.apply_blur and self.blur_kernel > 1:
            k = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
            processed = cv2.GaussianBlur(processed, (k, k), 0)

        # 3. Contrast adjustment (CLAHE on L-channel of LAB space to preserve colors)
        if self.apply_clahe:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            processed = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return processed

    def stream_frames(
        self, start_frame: int = 0, max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray, np.ndarray], None, None]:
        """
        Yields (frame_index, timestamp_sec, original_frame, preprocessed_frame).
        """
        cap = cv2.VideoCapture(self.video_path)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        yielded_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp_sec = frame_idx / self.fps
            preprocessed = self.preprocess_frame(frame)

            yield frame_idx, timestamp_sec, frame, preprocessed

            frame_idx += 1
            yielded_count += 1
            if max_frames is not None and yielded_count >= max_frames:
                break

        cap.release()


if __name__ == "__main__":
    print("Testing VideoPreprocessor...")
    from config import SAMPLE_VIDEOS_DIR

    sample_video = os.path.join(SAMPLE_VIDEOS_DIR, "traffic_cars.mp4")
    if os.path.exists(sample_video):
        prep = VideoPreprocessor(sample_video, apply_clahe=True)
        print("Metadata:", prep.get_metadata())
        for f_idx, t_sec, raw, proc in prep.stream_frames(max_frames=5):
            print(f"Frame {f_idx}: raw {raw.shape}, proc {proc.shape}, timestamp {t_sec:.3f}s")
        print("Preprocessor test passed!")
    else:
        print(f"Sample video not found at: {sample_video}")
