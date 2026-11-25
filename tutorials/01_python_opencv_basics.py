"""
Task 1 Tutorial: Python & OpenCV Fundamentals for Video Object Tracking
========================================================================

This script covers essential concepts needed for video processing, object
detection, and visual trajectory tracking using Python and OpenCV:

1. Image representation as NumPy multi-dimensional arrays (H, W, C)
2. BGR vs. RGB vs. Grayscale color spaces
3. Drawing annotations: bounding boxes, centroids, heading arrows, HUD text
4. Video reading, frame extraction, property retrieval, and video writing
5. Image filtering, noise reduction, and contrast enhancement (CLAHE)
"""

import os
import cv2
import numpy as np


def tutorial_image_basics():
    """Demonstrate how OpenCV represents images as NumPy arrays."""
    print("\n--- 1. Image Representation & Color Spaces ---")

    # Create a synthetic canvas (480 height, 640 width, 3 color channels BGR)
    height, width = 480, 640
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    print(f"Canvas Shape (H, W, C): {canvas.shape}")
    print(f"Data type: {canvas.dtype} (values range from 0 to 255)")

    # Color space explanation: OpenCV uses BGR by default, not RGB
    # Fill background with dark navy blue: [Blue=40, Green=20, Red=15]
    canvas[:] = (40, 20, 15)

    # Convert to Grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale Shape (H, W): {gray.shape}")

    return canvas


def tutorial_drawing_primitives(frame):
    """Demonstrate drawing bounding boxes, labels, trajectory lines, and direction arrows."""
    print("\n--- 2. Drawing Primitives for Object Tracking ---")

    # 1. Bounding Box: (x1, y1) top-left, (x2, y2) bottom-right
    x1, y1, x2, y2 = 120, 150, 320, 360
    color_green = (0, 255, 128)  # BGR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_green, thickness=2)

    # 2. Object Centroid
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), radius=6, color=(0, 0, 255), thickness=-1)  # Red filled

    # 3. Label tag with background rectangle for contrast
    label = "ID: 4 | car (94%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw label background
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - 8),
        (x1 + text_w + 6, y1),
        (0, 200, 100),
        cv2.FILLED,
    )
    # Draw label text
    cv2.putText(
        frame,
        label,
        (x1 + 3, y1 - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    # 4. Trajectory Path (simulated historical positions)
    trajectory_points = np.array(
        [
            [160, 380],
            [180, 350],
            [200, 310],
            [215, 270],
            [cx, cy],
        ],
        dtype=np.int32,
    )
    cv2.polylines(
        frame,
        [trajectory_points],
        isClosed=False,
        color=(255, 200, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    # 5. Velocity / Movement Direction Arrow
    # Arrow starting at centroid and extending in direction of motion (dx=60, dy=-40)
    target_x = cx + 60
    target_y = cy - 40
    cv2.arrowedLine(
        frame,
        (cx, cy),
        (target_x, target_y),
        color=(0, 255, 255),
        thickness=3,
        tipLength=0.25,
    )

    # 6. Telemetry HUD
    hud_lines = [
        "TRACKING TELEMETRY",
        f"Position: ({cx}, {cy}) px",
        "Displacement: dx=+15, dy=-12 px",
        "Speed: 235.4 px/s | Heading: 38.7 deg (NE)",
    ]
    hud_y = 40
    for line in hud_lines:
        cv2.putText(
            frame,
            line,
            (20, hud_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 240, 255),
            1,
            cv2.LINE_AA,
        )
        hud_y += 24

    return frame


def tutorial_video_io():
    """Demonstrate how to write and read a video file using OpenCV."""
    print("\n--- 3. Video Creation and Reading with cv2.VideoWriter & cv2.VideoCapture ---")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "tutorial_demo.mp4")

    # Define video properties
    width, height = 640, 480
    fps = 30
    total_frames = 60  # 2-second demo clip

    # Create VideoWriter: FourCC codec 'mp4v' for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print(f"Generating sample video with animated object at: {video_path}")
    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (35, 30, 25)  # dark background

        # Object moves across the frame
        cx = int(80 + f * 7.5)
        cy = int(240 - 50 * np.sin(f * 0.1))
        radius = 24

        # Draw circle object
        cv2.circle(frame, (cx, cy), radius, (0, 165, 255), -1)
        # Draw label
        cv2.putText(
            frame,
            f"Object ID: 1 | Frame {f+1}/{total_frames}",
            (cx - 50, cy - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()
    print("Video encoding complete.")

    # Now verify reading back the video with cv2.VideoCapture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    retrieved_fps = cap.get(cv2.CAP_PROP_FPS)
    retrieved_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    retrieved_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    retrieved_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Read Verification -> Resolution: {retrieved_w}x{retrieved_h}, FPS: {retrieved_fps}, Frames: {retrieved_frames}")
    cap.release()


def main():
    print("=================================================================")
    print("Task 1: Python Basics & OpenCV for Video Object Movement Tracking")
    print("=================================================================")

    # 1. Image array basics
    canvas = tutorial_image_basics()

    # 2. Draw tracking visualization components
    annotated = tutorial_drawing_primitives(canvas)

    # 3. Save demonstration image
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    demo_image_path = os.path.join(output_dir, "tutorial_annotated_frame.png")
    cv2.imwrite(demo_image_path, annotated)
    print(f"Saved annotated frame to: {demo_image_path}")

    # 4. Video Read/Write
    tutorial_video_io()

    print("\n[SUCCESS] Task 1 tutorial completed without errors!")


if __name__ == "__main__":
    main()
