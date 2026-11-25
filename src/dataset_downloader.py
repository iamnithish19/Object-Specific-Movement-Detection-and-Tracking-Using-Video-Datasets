"""
Task 2: Dataset Collection & Downloader
=======================================
Downloads standardized real-world benchmark video datasets for object detection
and multi-object tracking (vehicles, pedestrians, multimodal traffic).
Also generates synthetic video streams for zero-dependency offline testing.
"""

import os
import sys
import urllib.request
import cv2
import numpy as np

# Standard benchmark datasets from public computer vision repositories
DATASET_REGISTRY = {
    "traffic": {
        "filename": "traffic_cars.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
        "description": "Highway / road traffic with multiple moving cars and vans.",
        "primary_classes": ["car", "truck", "bus"],
    },
    "pedestrians": {
        "filename": "pedestrians_walk.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
        "description": "Pedestrian walkway with people moving in multiple directions.",
        "primary_classes": ["person"],
    },
    "multiclass": {
        "filename": "multi_object_street.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
        "description": "Urban street scene with pedestrians, cyclists, and moving cars.",
        "primary_classes": ["person", "bicycle", "car"],
    },
}


def download_file_with_progress(url: str, output_path: str):
    """Download a file with console progress indicator."""
    print(f"Downloading from: {url}")
    print(f"Destination: {output_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, (downloaded / total_size) * 100.0)
            sys.stdout.write(
                f"\rProgress: [{percent:5.1f}%] - {downloaded / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB"
            )
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print("\nDownload finished.")


def generate_synthetic_drone_video(output_path: str, duration_sec: int = 6, fps: int = 30):
    """
    Generates a realistic synthetic aerial drone/surveillance video for Task 5/Task 13
    testing when offline or evaluating drone trajectories.
    """
    print(f"Generating synthetic drone test video at: {output_path}")
    width, height = 800, 600
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    np.random.seed(42)

    # Simulated drone state (starts at top left, moves diagonally, hovers, then flies eastward)
    x = 150.0
    y = 120.0
    vx = 2.5
    vy = 1.8

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Background: aerial terrain (subtle ground texture)
        frame[:] = (70, 85, 75)
        # Add road grid lines
        cv2.line(frame, (0, 300), (width, 300), (90, 90, 90), 40)
        cv2.line(frame, (0, 300), (width, 300), (200, 200, 200), 2)  # center divider
        cv2.line(frame, (400, 0), (400, height), (90, 90, 90), 40)

        # Background vehicle driving on road
        car_x = int((f * 4) % (width + 80) - 40)
        cv2.rectangle(frame, (car_x - 30, 285), (car_x + 30, 315), (30, 30, 180), -1)
        cv2.putText(frame, "Ground Target", (car_x - 30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Drone trajectory dynamics
        if f < 60:
            # Phase 1: Diagonal flight South-East
            x += vx
            y += vy
        elif f < 100:
            # Phase 2: Hover with small wind jitter
            x += np.random.uniform(-0.4, 0.4)
            y += np.random.uniform(-0.4, 0.4)
        else:
            # Phase 3: Acceleration East-North-East
            x += vx * 1.5
            y -= vy * 0.8

        ix, iy = int(x), int(y)

        # Render Drone / Quadcopter with 4 rotors and cross-frame
        # Central fuselage
        cv2.circle(frame, (ix, iy), 14, (40, 40, 40), -1)
        cv2.circle(frame, (ix, iy), 10, (0, 215, 255), -1)  # Core LED
        # Carbon arms
        arm_len = 28
        cv2.line(frame, (ix - arm_len, iy - arm_len), (ix + arm_len, iy + arm_len), (20, 20, 20), 3)
        cv2.line(frame, (ix - arm_len, iy + arm_len), (ix + arm_len, iy - arm_len), (20, 20, 20), 3)
        # 4 Rotors
        for dx, dy in [(-arm_len, -arm_len), (arm_len, -arm_len), (-arm_len, arm_len), (arm_len, arm_len)]:
            cv2.circle(frame, (ix + dx, iy + dy), 9, (180, 180, 180), 2)
            # Spinning blade blur
            blade_phase = (f * 45) % 180
            rad = np.radians(blade_phase)
            bx = int(9 * np.cos(rad))
            by = int(9 * np.sin(rad))
            cv2.line(frame, (ix + dx - bx, iy + dy - by), (ix + dx + bx, iy + dy + by), (240, 240, 240), 2)

        # Overlay video frame metadata
        cv2.putText(
            frame,
            f"Aerial Surveillance Test Feed | Frame {f+1}/{total_frames}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print("Synthetic drone video successfully created.")


def download_all_datasets(dest_dir: str = None) -> dict:
    """Download all benchmark videos into destination directory."""
    if dest_dir is None:
        dest_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_videos")
    os.makedirs(dest_dir, exist_ok=True)

    downloaded_files = {}

    for key, info in DATASET_REGISTRY.items():
        target_path = os.path.join(dest_dir, info["filename"])
        if not os.path.exists(target_path) or os.path.getsize(target_path) < 10000:
            try:
                print(f"\n--- Fetching dataset: {key} ({info['description']}) ---")
                download_file_with_progress(info["url"], target_path)
            except Exception as e:
                print(f"[Warning] Could not download {key} ({e}).")
        else:
            print(f"Dataset already present: {info['filename']}")
        downloaded_files[key] = target_path

    # Generate synthetic drone video as well
    drone_path = os.path.join(dest_dir, "drone_surveillance.mp4")
    if not os.path.exists(drone_path):
        generate_synthetic_drone_video(drone_path)
    downloaded_files["drone"] = drone_path

    return downloaded_files


if __name__ == "__main__":
    print("=" * 60)
    print("Task 2: Video Dataset Collection")
    print("=" * 60)
    files = download_all_datasets()
    print("\nDataset inventory:")
    for k, p in files.items():
        exists = os.path.exists(p)
        size_mb = os.path.getsize(p) / (1024 * 1024) if exists else 0
        print(f" - {k:<12}: {os.path.basename(p):<25} ({size_mb:.2f} MB)")
