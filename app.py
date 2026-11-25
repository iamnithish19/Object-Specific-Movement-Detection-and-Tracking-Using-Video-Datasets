"""
Object-Specific Movement Detection & Tracking Web Application
=============================================================
Features:
- Landing / Home Page (/) describing project features with "Start Tracking" button
- Live Interactive Tracking Workspace (/tracker)
- Device Video Upload with immediate live stream switching
- Video Frame Splitting and disk export
- Consecutive Frame-by-Frame Comparison: Frame (t-1) vs Frame (t)
- Exact mathematical distance: dX, dY, Step Distance = sqrt(dX^2 + dY^2)
- Step controls: -1 Frame, +1 Frame, Timeline Slider
- Smart Auto-Lock when target leaves the frame
"""

import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Ensure src modules are discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from classifier import ObjectClassifier
from config import DATA_DIR, OUTPUTS_DIR, SAMPLE_VIDEOS_DIR
from motion_analyzer import MotionAnalyzer, MotionState
from preprocessor import VideoPreprocessor
from tracker import ByteTrackTracker
from visualizer import VideoVisualizer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250 MB max upload

UPLOADS_DIR = os.path.join(SAMPLE_VIDEOS_DIR, "uploads")
FRAMES_DIR = os.path.join(OUTPUTS_DIR, "extracted_frames")
GRAPHICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Graphics")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


class StreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.video_list = [
            {"id": "pedestrians", "name": "Pedestrians Walkway", "path": os.path.join(SAMPLE_VIDEOS_DIR, "pedestrians_walk.mp4")},
            {"id": "traffic", "name": "Highway Traffic Cars", "path": os.path.join(SAMPLE_VIDEOS_DIR, "traffic_cars.mp4")},
            {"id": "multiclass", "name": "Urban Street Scene", "path": os.path.join(SAMPLE_VIDEOS_DIR, "multi_object_street.mp4")},
            {"id": "drone", "name": "Aerial Drone Surveillance", "path": os.path.join(SAMPLE_VIDEOS_DIR, "drone_surveillance.mp4")},
        ]
        self._populate_uploaded_videos()

        # Check if an uploaded video exists, prefer it as default
        default_vid = self.video_list[0]
        for v in self.video_list:
            if "SAMPLE_CCTV_1" in v["name"] or "Uploaded" in v["name"]:
                default_vid = v
                break

        self.current_video_path = default_vid["path"]
        self.current_video_name = os.path.basename(default_vid["path"])
        self.video_generation_id = 0  # Incremented whenever video changes to break frame loops

        self.selected_id: Optional[int] = None
        self.auto_switch_on_loss: bool = True
        self.is_paused: bool = False
        self.current_frame_idx: int = 0
        self.total_video_frames: int = 0
        self.video_fps: float = 30.0

        # Step inspection mode
        self.step_mode: bool = False
        self.seek_frame_target: Optional[int] = None

        # Deep learning models
        self.classifier = ObjectClassifier()
        self.tracker = ByteTrackTracker(
            model_path="yolov8n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            classifier=self.classifier,
        )
        self.analyzer = MotionAnalyzer(fps=15.0)
        self.visualizer = VideoVisualizer(trail_length=80, show_hud=False)

        self.latest_telemetry: Dict = {}
        self.active_candidates: List[Dict] = []
        self.trajectory_history: List[Dict] = []
        self.lost_frames_counter: int = 0

    def _populate_uploaded_videos(self):
        """Scans uploads directory and registers existing uploaded videos."""
        if os.path.exists(UPLOADS_DIR):
            for fname in os.listdir(UPLOADS_DIR):
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    vpath = os.path.join(UPLOADS_DIR, fname)
                    if not any(v["path"] == vpath for v in self.video_list):
                        self.video_list.insert(0, {
                            "id": f"upload_{fname}",
                            "name": f"Uploaded Video: {fname}",
                            "path": vpath,
                        })

    def set_video(self, video_path: str, video_name: str):
        with self.lock:
            if os.path.exists(video_path):
                self.current_video_path = video_path
                self.current_video_name = video_name
                self.video_generation_id += 1  # Triggers stream generator to reload video
                self.selected_id = None
                self.current_frame_idx = 0
                self.seek_frame_target = 0
                self.tracker.reset()
                self.analyzer.reset()
                self.trajectory_history.clear()
                self.active_candidates.clear()
                self.latest_telemetry.clear()
                self.lost_frames_counter = 0
                print(f"[StreamManager] Switched active video to: {video_name} ({video_path}) [gen_id={self.video_generation_id}]")

    def set_target_id(self, target_id: Optional[int]):
        with self.lock:
            self.selected_id = target_id
            self.analyzer.reset()
            self.trajectory_history.clear()
            self.lost_frames_counter = 0

    def handle_click(self, x: float, y: float) -> Optional[int]:
        """Matches click coordinates to active candidate bounding boxes."""
        with self.lock:
            for cand in self.active_candidates:
                x1, y1, x2, y2 = cand["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_id = cand["id"]
                    self.analyzer.reset()
                    self.trajectory_history.clear()
                    self.lost_frames_counter = 0
                    print(f"[Live Server] Target ID switched via mouse click to: #{self.selected_id}")
                    return self.selected_id
        return None


manager = StreamManager()


def generate_frames():
    """Generator streaming annotated frames with frame-to-frame movement estimation."""
    while True:
        with manager.lock:
            video_path = manager.current_video_path
            current_gen = manager.video_generation_id

        if not video_path or not os.path.exists(video_path):
            time.sleep(0.3)
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        manager.video_fps = fps if fps > 0 else 30.0
        manager.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        manager.analyzer.fps = manager.video_fps

        frame_idx = 0
        prev_centroid: Optional[Tuple[float, float]] = None

        while cap.isOpened():
            # Crucial: Check if active video was changed by user
            if manager.video_generation_id != current_gen:
                break

            # Check seek requests from UI slider
            if manager.seek_frame_target is not None:
                frame_idx = max(0, min(manager.seek_frame_target, manager.total_video_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                manager.seek_frame_target = None

            # Handle step mode or pause
            if manager.is_paused and not manager.step_mode:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                # Video loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                prev_centroid = None
                continue

            manager.current_frame_idx = frame_idx
            timestamp_sec = frame_idx / manager.video_fps

            with manager.lock:
                active_tracks = manager.tracker.update(frame)

                # Candidate list for UI
                manager.active_candidates = [
                    {
                        "id": t.track_id,
                        "category": t.category,
                        "class_name": t.class_name,
                        "confidence": round(t.confidence * 100, 1),
                        "bbox": [round(v, 1) for v in t.bbox],
                        "centroid": [round(v, 1) for v in t.centroid],
                    }
                    for t in active_tracks
                ]

                # Default selection: if none selected, pick first active
                if manager.selected_id is None and active_tracks:
                    manager.selected_id = active_tracks[0].track_id

                # Find selected object
                selected_obj = next((t for t in active_tracks if t.track_id == manager.selected_id), None)
                motion_state = None

                # Smart recovery: if selected object left and auto-switch is on, lock onto active object
                if selected_obj is None:
                    manager.lost_frames_counter += 1
                    if manager.auto_switch_on_loss and manager.lost_frames_counter > 20 and active_tracks:
                        # Auto switch to active target
                        manager.selected_id = active_tracks[0].track_id
                        selected_obj = active_tracks[0]
                        manager.lost_frames_counter = 0
                        prev_centroid = None
                        print(f"[Smart Auto-Lock] Previous target left frame. Auto-switched to active object ID #{manager.selected_id}")
                else:
                    manager.lost_frames_counter = 0

                # Frame-by-Frame Comparison Computation
                frame_comparison_data = {}
                if selected_obj is not None:
                    curr_c = selected_obj.centroid
                    if prev_centroid is not None:
                        frame_dx = round(curr_c[0] - prev_centroid[0], 2)
                        frame_dy = round(curr_c[1] - prev_centroid[1], 2)
                        frame_step_dist = round(float(np.hypot(frame_dx, frame_dy)), 2)
                    else:
                        frame_dx = 0.0
                        frame_dy = 0.0
                        frame_step_dist = 0.0

                    motion_state = manager.analyzer.update(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        centroid=curr_c,
                    )

                    frame_comparison_data = {
                        "frame_prev": max(0, frame_idx - 1),
                        "frame_curr": frame_idx,
                        "prev_pos": [round(prev_centroid[0], 1), round(prev_centroid[1], 1)] if prev_centroid else [round(curr_c[0], 1), round(curr_c[1], 1)],
                        "curr_pos": [round(curr_c[0], 1), round(curr_c[1], 1)],
                        "dx": frame_dx,
                        "dy": frame_dy,
                        "distance_between_frames": frame_step_dist,
                        "cumulative_distance": motion_state.cumulative_distance,
                    }

                    manager.latest_telemetry = {
                        "frame": frame_idx,
                        "total_frames": manager.total_video_frames,
                        "timestamp": round(timestamp_sec, 2),
                        "selected_id": manager.selected_id,
                        "category": selected_obj.category.upper(),
                        "confidence": round(selected_obj.confidence * 100, 1),
                        "x": motion_state.x,
                        "y": motion_state.y,
                        "dx": frame_dx,
                        "dy": frame_dy,
                        "step_distance": frame_step_dist,
                        "speed": motion_state.speed_smoothed,
                        "orientation": motion_state.orientation_deg,
                        "direction": motion_state.direction,
                        "cumulative_dist": motion_state.cumulative_distance,
                        "net_disp": round(float(np.hypot(motion_state.net_dx, motion_state.net_dy)), 1),
                        "fps": round(manager.video_fps, 1),
                        "status": "tracking",
                        "comparison": frame_comparison_data,
                    }

                    prev_centroid = curr_c

                    # Append history
                    if len(manager.trajectory_history) > 150:
                        manager.trajectory_history.pop(0)
                    manager.trajectory_history.append({
                        "frame": frame_idx,
                        "time": round(timestamp_sec, 2),
                        "speed": motion_state.speed_smoothed,
                        "dx": frame_dx,
                        "dy": frame_dy,
                        "step_dist": frame_step_dist,
                        "total_dist": motion_state.cumulative_distance,
                    })
                else:
                    prev_centroid = None
                    manager.latest_telemetry = {
                        "frame": frame_idx,
                        "total_frames": manager.total_video_frames,
                        "timestamp": round(timestamp_sec, 2),
                        "selected_id": manager.selected_id,
                        "category": "UNKNOWN",
                        "confidence": 0.0,
                        "x": None,
                        "y": None,
                        "dx": 0.0,
                        "dy": 0.0,
                        "step_distance": 0.0,
                        "speed": 0.0,
                        "orientation": 0.0,
                        "direction": "Searching...",
                        "cumulative_dist": 0.0,
                        "net_disp": 0.0,
                        "fps": round(manager.video_fps, 1),
                        "status": "searching",
                        "comparison": None,
                    }

                # Visual overlay rendering (HUD disabled on video stream, shown separately in dashboard)
                annotated = manager.visualizer.render(
                    frame=frame,
                    all_active_tracks=active_tracks,
                    selected_id=manager.selected_id,
                    trajectory_points=manager.analyzer.get_trajectory(),
                    motion_state=motion_state,
                    frame_idx=frame_idx,
                    total_frames=manager.total_video_frames,
                    fps=manager.video_fps,
                )

                # Visual Movement Distance Comparison - subtle centroid markers
                if selected_obj is not None and prev_centroid is not None and "prev_pos" in frame_comparison_data:
                    p_prev = (int(frame_comparison_data["prev_pos"][0]), int(frame_comparison_data["prev_pos"][1]))
                    p_curr = (int(frame_comparison_data["curr_pos"][0]), int(frame_comparison_data["curr_pos"][1]))
                    cv2.line(annotated, p_prev, p_curr, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(annotated, p_prev, 4, (0, 165, 255), -1)  # orange dot for prev frame
                    cv2.circle(annotated, p_curr, 4, (0, 255, 0), -1)    # green dot for curr frame

            # If step mode was triggered, pause immediately after processing one frame
            if manager.step_mode:
                manager.step_mode = False
                manager.is_paused = True

            ret_enc, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret_enc:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            frame_idx += 1
            if not manager.is_paused:
                time.sleep(1.0 / (manager.video_fps * 1.15))

        cap.release()


# -------------------------------------------------------------
# Web Routes
# -------------------------------------------------------------

@app.route("/")
def home():
    """Renders the comprehensive Project Home / Landing Page."""
    return render_template_string(HOME_PAGE_HTML, video_list=manager.video_list, current_video=manager.current_video_name)


@app.route("/tracker")
def tracker():
    """Renders the Live Object Movement Detection & Tracking Workspace."""
    return render_template_string(TRACKER_PAGE_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/telemetry")
def api_telemetry():
    with manager.lock:
        return jsonify({
            "telemetry": manager.latest_telemetry,
            "candidates": manager.active_candidates,
            "selected_id": manager.selected_id,
            "current_video": manager.current_video_name,
            "current_video_path": manager.current_video_path,
            "videos": manager.video_list,
            "history": manager.trajectory_history[-30:],
            "is_paused": manager.is_paused,
            "current_frame": manager.current_frame_idx,
            "total_frames": manager.total_video_frames,
            "auto_switch": manager.auto_switch_on_loss,
            "video_gen": manager.video_generation_id,
        })


@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    """Upload video from device and switch live tracking to it immediately."""
    if "video_file" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400

    file = request.files["video_file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return jsonify({"status": "error", "message": "Supported formats: .mp4, .avi, .mov, .mkv"}), 400

    target_path = os.path.join(UPLOADS_DIR, filename)
    file.save(target_path)

    # Register in video list if not already present
    vid_entry = {
        "id": f"upload_{filename}",
        "name": f"Uploaded Video: {filename}",
        "path": target_path,
    }
    if not any(v["path"] == target_path for v in manager.video_list):
        manager.video_list.insert(0, vid_entry)

    # Switch stream to uploaded video immediately
    manager.set_video(target_path, filename)
    return jsonify({
        "status": "success",
        "filename": filename,
        "path": target_path,
        "video_gen": manager.video_generation_id,
    })


@app.route("/api/split_frames", methods=["POST"])
def api_split_frames():
    """Splits current video into individual frame images saved in data/outputs/extracted_frames/<name>/."""
    video_path = manager.current_video_path
    if not os.path.exists(video_path):
        return jsonify({"status": "error", "message": "Video not found"}), 400

    base_name = os.path.splitext(manager.current_video_name)[0]
    out_dir = os.path.join(FRAMES_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    max_extract = 250  # Extract up to 250 frames
    while cap.isOpened() and count < max_extract:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"frame_{count:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, frame_name), frame)
        count += 1
    cap.release()

    return jsonify({
        "status": "success",
        "extracted_count": count,
        "output_directory": out_dir,
        "message": f"Successfully extracted {count} frames to {out_dir}",
    })


@app.route("/api/step_frame", methods=["POST"])
def api_step_frame():
    """Steps forward (+1) or backward (-1) by one frame for detailed comparison."""
    data = request.json or {}
    direction = int(data.get("step", 1))
    target = max(0, min(manager.current_frame_idx + direction, manager.total_video_frames - 1))
    manager.seek_frame_target = target
    manager.step_mode = True
    manager.is_paused = False
    return jsonify({"status": "success", "target_frame": target})


@app.route("/api/seek_frame", methods=["POST"])
def api_seek_frame():
    """Seeks directly to specific frame index."""
    data = request.json or {}
    target = int(data.get("frame", 0))
    manager.seek_frame_target = max(0, min(target, manager.total_video_frames - 1))
    return jsonify({"status": "success", "seek_frame": manager.seek_frame_target})


@app.route("/api/toggle_pause", methods=["POST"])
def api_toggle_pause():
    manager.is_paused = not manager.is_paused
    return jsonify({"is_paused": manager.is_paused})


@app.route("/api/toggle_auto_switch", methods=["POST"])
def api_toggle_auto_switch():
    manager.auto_switch_on_loss = not manager.auto_switch_on_loss
    return jsonify({"auto_switch": manager.auto_switch_on_loss})


@app.route("/api/select_id", methods=["POST"])
def api_select_id():
    data = request.json or {}
    t_id = data.get("id")
    if t_id is not None:
        manager.set_target_id(int(t_id))
        return jsonify({"status": "success", "selected_id": manager.selected_id})
    return jsonify({"status": "error", "message": "Missing id"}), 400


@app.route("/api/click", methods=["POST"])
def api_click():
    data = request.json or {}
    x = float(data.get("x", 0))
    y = float(data.get("y", 0))
    matched_id = manager.handle_click(x, y)
    return jsonify({"status": "success", "selected_id": matched_id})


@app.route("/api/change_video", methods=["POST"])
def api_change_video():
    data = request.json or {}
    vpath = data.get("video_path")
    vname = data.get("video_name", os.path.basename(vpath))
    if vpath and os.path.exists(vpath):
        manager.set_video(vpath, vname)
        return jsonify({"status": "success", "video": vname, "video_gen": manager.video_generation_id})
    return jsonify({"status": "error", "message": "Invalid video path"}), 400


@app.route("/outputs/<path:filename>")
def serve_output_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/graphics/<path:filename>")
def serve_graphics_file(filename):
    return send_from_directory(GRAPHICS_DIR, filename)


# =============================================================
# 1. LANDING / HOME PAGE TEMPLATE
# =============================================================

HOME_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Object-Specific Movement Detection & Tracking</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {
      --bg-dark: #080c14;
      --card-bg: #111827;
      --card-border: rgba(255, 255, 255, 0.12);
      --accent-cyan: #06b6d4;
      --accent-blue: #38bdf8;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --text-main: #f9fafb;
      --text-muted: #e2e8f0;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body {
      background: var(--bg-dark);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      position: relative;
    }

    /* Full-Screen Muted Background Video */
    #bg-video {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      object-fit: cover;
      filter: brightness(1.18);
      z-index: -2;
      pointer-events: none;
    }
    .bg-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(8, 12, 20, 0.10);
      z-index: -1;
      pointer-events: none;
    }

    /* Top Navigation */
    nav {
      display: flex; justify-content: space-between; align-items: center;
      padding: 18px 48px; background: rgba(11, 15, 25, 0.55);
      backdrop-filter: blur(10px); border-bottom: 1px solid var(--card-border);
      position: sticky; top: 0; z-index: 50;
    }
    .logo {
      font-size: 1.25rem; font-weight: 800; color: var(--accent-blue);
      display: flex; align-items: center; gap: 10px; text-decoration: none;
    }
    .nav-links { display: flex; gap: 24px; align-items: center; }
    .nav-link { color: var(--text-muted); text-decoration: none; font-size: 0.9rem; font-weight: 500; transition: color 0.2s; }
    .nav-link:hover { color: var(--accent-blue); }

    .btn {
      padding: 10px 20px; border-radius: 8px; font-weight: 600; font-size: 0.9rem;
      cursor: pointer; text-decoration: none; display: inline-flex; align-items: center;
      gap: 8px; border: none; transition: all 0.2s;
    }
    .btn-primary {
      background: linear-gradient(135deg, #0284c7, #06b6d4); color: #fff;
      box-shadow: 0 4px 28px rgba(6, 182, 212, 0.45);
    }
    .btn-primary:hover {
      background: linear-gradient(135deg, #0369a1, #0891b2);
      transform: translateY(-2px); box-shadow: 0 8px 36px rgba(6, 182, 212, 0.65);
    }

    .center-stage {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      padding: 32px 24px;
      z-index: 1;
    }

    .features-line {
      font-size: 1.05rem;
      color: #f1f5f9;
      max-width: 700px;
      line-height: 1.7;
      margin-bottom: 32px;
      text-shadow: 0 2px 12px rgba(0, 0, 0, 0.75);
      background: rgba(8, 12, 20, 0.42);
      backdrop-filter: blur(6px);
      padding: 16px 24px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.12);
    }

    footer {
      border-top: 1px solid var(--card-border); padding: 18px; text-align: center;
      color: var(--text-muted); font-size: 0.85rem; background: rgba(11, 15, 25, 0.55);
      backdrop-filter: blur(8px); z-index: 1;
    }
  </style>
</head>
<body>

  <!-- Background Video (Strictly Muted / No Noise) -->
  <video id="bg-video" autoplay muted loop playsinline preload="auto">
    <source src="/graphics/VISTA%20HOME%20PAGE.mp4" type="video/mp4">
  </video>
  <div class="bg-overlay"></div>

  <!-- Navigation Bar -->
  <nav>
    <a href="/" class="logo">
      <i class="fa-solid fa-crosshairs"></i> VISTA Object Tracker
    </a>
    <div class="nav-links">
      <a href="/outputs/multi_dataset_benchmark_report.md" target="_blank" class="nav-link">Benchmark Report</a>
    </div>
  </nav>

  <!-- Centered Start Stage -->
  <main class="center-stage">
    <a href="/tracker" class="btn btn-primary" style="padding: 18px 48px; font-size: 1.2rem; border-radius: 12px;">
      <i class="fa-solid fa-play"></i> Start Tracking
    </a>
  </main>

  <footer>
    <p>Object-Specific Movement Detection and Tracking System • Built with Python, OpenCV, YOLOv8, ByteTrack, and Matplotlib</p>
  </footer>

  <script>
    (function() {
      const vid = document.getElementById('bg-video');
      if (vid) {
        vid.muted = true;
        vid.volume = 0;
        vid.play().catch(() => {});
      }
    })();
  </script>
</body>
</html>
"""


# =============================================================
# 2. LIVE TRACKING WORKSPACE TEMPLATE
# =============================================================

TRACKER_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Live Object Movement Detection & Tracking Workspace</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-dark: #090d16;
      --card-bg: #131b2e;
      --accent-cyan: #06b6d4;
      --accent-amber: #f59e0b;
      --accent-green: #10b981;
      --accent-rose: #f43f5e;
      --accent-blue: #38bdf8;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border-color: #22304a;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg-dark); color: var(--text-main); min-height: 100vh; padding: 16px; }

    header {
      display: flex; justify-content: space-between; align-items: center;
      background: var(--card-bg); padding: 14px 22px; border-radius: 12px;
      border: 1px solid var(--border-color); margin-bottom: 16px;
    }
    .header-title h1 { font-size: 1.35rem; font-weight: 700; color: var(--accent-blue); display: flex; align-items: center; gap: 10px; }
    .header-title p { font-size: 0.82rem; color: var(--text-muted); margin-top: 3px; }

    .grid-container {
      display: grid;
      grid-template-columns: 2.1fr 1fr;
      gap: 16px;
    }

    .card {
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border-color);
      padding: 16px;
      margin-bottom: 16px;
    }
    .card-title {
      font-size: 0.95rem; font-weight: 600; color: var(--text-main);
      display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
      border-bottom: 1px solid var(--border-color); padding-bottom: 8px;
    }

    /* Video Player */
    .video-wrapper {
      position: relative;
      background: #000;
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: crosshair;
      border: 2px solid var(--border-color);
    }
    #video-stream {
      width: 100%;
      height: auto;
      display: block;
    }
    .video-hint {
      position: absolute;
      top: 10px; right: 10px;
      background: rgba(9, 13, 22, 0.85);
      padding: 5px 10px;
      border-radius: 6px;
      font-size: 0.75rem;
      color: var(--accent-amber);
      pointer-events: none;
      border: 1px solid rgba(245, 158, 11, 0.4);
    }

    /* Frame Comparison Banner */
    .compare-banner {
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.12), rgba(16, 185, 129, 0.12));
      border: 1px solid rgba(6, 182, 212, 0.35);
      border-radius: 10px;
      padding: 12px 16px;
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }
    .comp-box {
      text-align: center;
      padding: 6px;
      background: rgba(15, 23, 42, 0.5);
      border-radius: 6px;
      border: 1px solid var(--border-color);
    }
    .comp-title { font-size: 0.70rem; color: var(--text-muted); text-transform: uppercase; }
    .comp-val { font-size: 1.15rem; font-weight: 700; color: #fff; margin-top: 2px; }

    /* Timeline & Step Controls */
    .timeline-bar {
      margin-top: 12px;
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(15, 23, 42, 0.4);
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid var(--border-color);
    }
    .timeline-slider {
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: #334155;
      outline: none;
      cursor: pointer;
    }

    .btn {
      background: #1e293b;
      color: #fff;
      border: 1px solid var(--border-color);
      padding: 7px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
      text-decoration: none;
    }
    .btn:hover { background: #334155; border-color: var(--accent-cyan); }
    .btn-primary { background: #0284c7; border-color: #0284c7; }
    .btn-primary:hover { background: #0369a1; }
    .btn-green { background: #059669; border-color: #059669; }
    .btn-green:hover { background: #047857; }

    /* Telemetry HUD Grid */
    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .tele-stat {
      background: rgba(15, 23, 42, 0.65);
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      transition: all 0.2s;
    }
    .tele-stat:hover {
      border-color: rgba(6, 182, 212, 0.4);
      background: rgba(15, 23, 42, 0.85);
    }
    .tele-label {
      font-size: 0.68rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.4px;
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .tele-value { font-size: 1.15rem; font-weight: 700; color: #fff; margin-top: 3px; }
    .tele-unit { font-size: 0.72rem; color: var(--text-muted); font-weight: normal; margin-left: 2px; }

    .hud-status-pill {
      font-size: 0.70rem;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .status-tracking {
      background: rgba(16, 185, 129, 0.2);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.5);
    }
    .status-searching {
      background: rgba(245, 158, 11, 0.2);
      color: #f59e0b;
      border: 1px solid rgba(245, 158, 11, 0.5);
    }

    /* Candidate list */
    .candidate-list { max-height: 200px; overflow-y: auto; margin-top: 8px; }
    .candidate-item {
      display: flex; justify-content: space-between; align-items: center;
      padding: 8px 10px; border-radius: 6px; margin-bottom: 6px;
      background: rgba(15, 23, 42, 0.4); border: 1px solid transparent; cursor: pointer;
    }
    .candidate-item:hover { border-color: var(--accent-cyan); background: rgba(6, 182, 212, 0.08); }
    .candidate-item.active { border-color: var(--accent-green); background: rgba(16, 185, 129, 0.12); }

    .badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.70rem; font-weight: 700; }
    .badge-target { background: #10b981; color: #000; }
    .badge-cat { background: #38bdf8; color: #000; }

    .upload-zone {
      border: 2px dashed #334155;
      border-radius: 8px;
      padding: 14px;
      text-align: center;
      cursor: pointer;
      background: rgba(15, 23, 42, 0.3);
      transition: all 0.2s;
    }
    .upload-zone:hover { border-color: var(--accent-cyan); background: rgba(6, 182, 212, 0.05); }

    .chart-box { height: 160px; margin-top: 10px; }
  </style>
</head>
<body>

  <header>
    <div class="header-title">
      <h1><i class="fa-solid fa-crosshairs"></i> Object Movement Detection & Frame-by-Frame Comparison</h1>
      <p>Compare Frame (t-1) vs (t) • Exact Movement Distance • YOLOv8 + ByteTrack</p>
    </div>
    <div style="display: flex; gap: 8px;">
      <a href="/" class="btn"><i class="fa-solid fa-house"></i> Home</a>
      <button class="btn btn-green" onclick="extractAllFrames()"><i class="fa-solid fa-layer-group"></i> Split & Save Frames</button>
      <a href="/outputs/multi_dataset_benchmark_report.md" target="_blank" class="btn"><i class="fa-solid fa-file-lines"></i> Benchmark Report</a>
    </div>
  </header>

  <div class="grid-container">

    <!-- Left Column: Video Stream, Frame Comparison & Timeline -->
    <div>
      <div class="card">
        <div class="card-title">
          <i class="fa-solid fa-video" style="color: var(--accent-cyan);"></i>
          <span>Live Tracking Stream</span>
          <span id="active-video-badge" style="margin-left: auto; font-size: 0.8rem; color: var(--accent-blue);"></span>
        </div>

        <div class="video-wrapper" id="video-box" onclick="handleVideoClick(event)">
          <img id="video-stream" src="/video_feed" alt="Live Video Stream">
          <div class="video-hint"><i class="fa-solid fa-crosshairs"></i> Click any object to lock tracking</div>
        </div>

        <!-- Frame-to-Frame Comparison & Exact Distance Banner -->
        <div class="compare-banner">
          <div class="comp-box">
            <div class="comp-title">Frame Comparison</div>
            <div class="comp-val" id="comp-frames">F[--] vs F[--]</div>
          </div>
          <div class="comp-box">
            <div class="comp-title">Frame Displacement</div>
            <div class="comp-val" id="comp-disp" style="color: var(--accent-cyan);">dX: 0, dY: 0</div>
          </div>
          <div class="comp-box">
            <div class="comp-title">Distance Between Frames</div>
            <div class="comp-val" id="comp-step-dist" style="color: var(--accent-amber);">0.0 <span style="font-size: 0.75rem;">px</span></div>
          </div>
          <div class="comp-box">
            <div class="comp-title">Total Cumulative Distance</div>
            <div class="comp-val" id="comp-total-dist" style="color: var(--accent-green);">0.0 <span style="font-size: 0.75rem;">px</span></div>
          </div>
        </div>

        <!-- Timeline Scrubber & Step Controls -->
        <div class="timeline-bar">
          <button class="btn" onclick="stepFrame(-1)"><i class="fa-solid fa-backward-step"></i> -1 Frame</button>
          <button class="btn btn-primary" onclick="togglePause()"><i class="fa-solid fa-pause" id="pause-btn-icon"></i> <span id="pause-btn-text">Pause</span></button>
          <button class="btn" onclick="stepFrame(1)">+1 Frame <i class="fa-solid fa-forward-step"></i></button>

          <input type="range" min="0" max="100" value="0" class="timeline-slider" id="frame-slider" oninput="seekFrame(this.value)">
          <span style="font-size: 0.8rem; color: var(--text-muted); min-width: 85px;" id="frame-counter-label">0 / 0</span>
        </div>
      </div>

      <!-- Real-Time Kinematics Chart -->
      <div class="card">
        <div class="card-title">
          <i class="fa-solid fa-chart-line" style="color: var(--accent-amber);"></i>
          <span>Frame Movement Distance & Speed Profile</span>
        </div>
        <div class="chart-box">
          <canvas id="motionChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Right Column: Upload Video, Telemetry, Candidate Picker -->
    <div>

      <!-- Upload Video From Device -->
      <div class="card">
        <div class="card-title">
          <i class="fa-solid fa-cloud-arrow-up" style="color: var(--accent-cyan);"></i>
          <span>Upload Video from This Device</span>
        </div>
        <div class="upload-zone" onclick="document.getElementById('file-input').click()">
          <i class="fa-solid fa-file-video" style="font-size: 1.8rem; color: #38bdf8; margin-bottom: 6px;"></i>
          <p style="font-size: 0.82rem; color: var(--text-main);">Click to Select Video File (.mp4, .avi, .mov)</p>
          <p style="font-size: 0.72rem; color: var(--text-muted); margin-top: 3px;" id="upload-status">Directly analyze videos from your device</p>
          <input type="file" id="file-input" style="display: none;" accept="video/*" onchange="uploadVideo(this)">
        </div>

        <div style="margin-top: 12px;">
          <label style="font-size: 0.75rem; color: var(--text-muted); display: block; margin-bottom: 4px;">Active Video Dataset:</label>
          <select id="video-dropdown" class="btn" style="width: 100%; text-align: left;" onchange="switchVideo(this.value)">
            <!-- Options populated dynamically -->
          </select>
        </div>
      </div>

      <!-- Target Telemetry HUD Dashboard (Separate from Video Space) -->
      <div class="card" style="border: 1px solid rgba(6, 182, 212, 0.4); background: rgba(19, 27, 46, 0.95); box-shadow: 0 4px 24px rgba(0,0,0,0.45);">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; margin-bottom: 12px;">
          <div id="hud-target-title" style="font-size: 0.95rem; font-weight: 700; color: #10b981; display: flex; align-items: center; gap: 8px;">
            <i class="fa-solid fa-crosshairs"></i>
            <span>TARGET: SEARCHING</span>
          </div>
          <div id="hud-status-badge" class="hud-status-pill status-searching">
            <i class="fa-solid fa-circle" style="font-size: 0.55rem;"></i> Searching
          </div>
        </div>

        <div class="telemetry-grid">
          <div class="tele-stat">
            <div class="tele-label"><i class="fa-solid fa-location-dot" style="color: #38bdf8;"></i> Position (X, Y)</div>
            <div class="tele-value" id="hud-pos" style="font-size: 1.05rem;">-- <span class="tele-unit">px</span></div>
          </div>
          <div class="tele-stat">
            <div class="tele-label"><i class="fa-solid fa-arrows-up-down-left-right" style="color: #06b6d4;"></i> Displacement</div>
            <div class="tele-value" id="hud-disp" style="font-size: 1.05rem; color: var(--accent-cyan);">dX: 0.0, dY: 0.0</div>
          </div>
          <div class="tele-stat">
            <div class="tele-label"><i class="fa-solid fa-gauge-high" style="color: #10b981;"></i> Speed</div>
            <div class="tele-value" id="hud-speed" style="font-size: 1.05rem;">0.0 <span class="tele-unit">px/s</span></div>
          </div>
          <div class="tele-stat">
            <div class="tele-label"><i class="fa-solid fa-compass" style="color: #a78bfa;"></i> Heading & Angle</div>
            <div class="tele-value" id="hud-heading" style="font-size: 1.0rem; color: #a78bfa;">--</div>
          </div>
          <div class="tele-stat">
            <div class="tele-label"><i class="fa-solid fa-ruler" style="color: #f59e0b;"></i> Net Displacement</div>
            <div class="tele-value" id="hud-net-dist" style="font-size: 1.05rem; color: var(--accent-amber);">0.0 <span class="tele-unit">px</span></div>
          </div>
          <div class="tele-stat">
            <div class="tele-label"><i class="fa-solid fa-route" style="color: #ec4899;"></i> Total Distance</div>
            <div class="tele-value" id="hud-total-dist" style="font-size: 1.05rem; color: #ec4899;">0.0 <span class="tele-unit">px</span></div>
          </div>
          <div class="tele-stat">
            <div class="tele-label"><i class="fa-solid fa-film" style="color: #38bdf8;"></i> Frame Progress</div>
            <div class="tele-value" id="hud-frame-progress" style="font-size: 1.05rem; color: #38bdf8;">0 / 0</div>
          </div>
          <div class="tele-stat">
            <div class="tele-label"><i class="fa-solid fa-bolt" style="color: #eab308;"></i> Stream Rate</div>
            <div class="tele-value" id="hud-stream-fps" style="font-size: 1.05rem; color: #eab308;">0.0 <span class="tele-unit">FPS</span></div>
          </div>
        </div>
      </div>

      <!-- Candidate Selector -->
      <div class="card">
        <div class="card-title">
          <i class="fa-solid fa-users-viewfinder" style="color: #a78bfa;"></i>
          <span>Active Objects in Frame</span>
          <button class="btn" style="margin-left: auto; padding: 3px 8px; font-size: 0.72rem;" onclick="toggleAutoSwitch()" id="auto-switch-btn">
            Auto-Lock: ON
          </button>
        </div>
        <div class="candidate-list" id="candidates-container">
          <div style="font-size: 0.8rem; color: var(--text-muted); text-align: center; padding: 15px;">
            Scanning frame objects...
          </div>
        </div>
      </div>

      <!-- Export & Split Status Card -->
      <div class="card" id="frame-split-card" style="display: none;">
        <div class="card-title">
          <i class="fa-solid fa-folder-open" style="color: var(--accent-green);"></i>
          <span>Extracted Frames Info</span>
        </div>
        <p id="split-msg" style="font-size: 0.8rem; color: var(--text-muted);"></p>
      </div>

    </div>

  </div>

  <script>
    // Initialize Kinematics Chart
    const ctx = document.getElementById('motionChart').getContext('2d');
    const motionChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Frame Distance Moved (px)',
            data: [],
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.15)',
            borderWidth: 2,
            tension: 0.2,
            fill: true,
          },
          {
            label: 'Smoothed Speed (px/s)',
            data: [],
            borderColor: '#06b6d4',
            borderWidth: 1.5,
            tension: 0.3,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: '#22304a' }, ticks: { color: '#94a3b8', maxTicksLimit: 8 } },
          y: { grid: { color: '#22304a' }, ticks: { color: '#94a3b8' } }
        },
        plugins: {
          legend: { labels: { color: '#f8fafc', font: { size: 10 } } }
        }
      }
    });

    let currentVideoList = [];
    let isUserSeeking = false;
    let currentVideoGen = -1;

    async function fetchTelemetry() {
      try {
        const resp = await fetch('/api/telemetry');
        const data = await resp.json();

        const t = data.telemetry;
        document.getElementById('active-video-badge').innerText = data.current_video;

        // If backend changed video generation, force reload image src
        if (currentVideoGen !== -1 && data.video_gen !== currentVideoGen) {
          console.log("Video source changed on server! Refreshing stream...");
          document.getElementById('video-stream').src = '/video_feed?t=' + Date.now();
        }
        currentVideoGen = data.video_gen;

        // Frame Comparison Banner
        if (t && t.comparison) {
          const c = t.comparison;
          document.getElementById('comp-frames').innerText = `F[${c.frame_prev}] ➜ F[${c.frame_curr}]`;
          document.getElementById('comp-disp').innerText = `dX: ${c.dx > 0 ? '+' : ''}${c.dx.toFixed(1)}, dY: ${c.dy > 0 ? '+' : ''}${c.dy.toFixed(1)}`;
          document.getElementById('comp-step-dist').innerHTML = `${c.distance_between_frames.toFixed(2)} <span style="font-size:0.75rem;">px</span>`;
          document.getElementById('comp-total-dist').innerHTML = `${c.cumulative_distance.toFixed(1)} <span style="font-size:0.75rem;">px</span>`;
        }

        // Live Telemetry HUD Dashboard (Separate from Video Space)
        if (t && t.selected_id) {
          const isSearching = t.status === "searching" || t.x === null || t.x === undefined;

          if (!isSearching) {
            document.getElementById('hud-target-title').innerHTML = `
              <i class="fa-solid fa-crosshairs" style="color: #10b981;"></i>
              <span>TARGET LOCKED [ID #${t.selected_id}] ${t.category}</span>
              <span class="badge badge-target" style="margin-left: 6px;">${t.confidence}%</span>
            `;
            document.getElementById('hud-status-badge').className = "hud-status-pill status-tracking";
            document.getElementById('hud-status-badge').innerHTML = '<i class="fa-solid fa-circle" style="font-size: 0.55rem;"></i> Active Tracking';

            document.getElementById('hud-pos').innerHTML = `(${t.x.toFixed(0)}, ${t.y.toFixed(0)}) <span class="tele-unit">px</span>`;
            document.getElementById('hud-disp').innerHTML = `dX: ${t.dx > 0 ? '+' : ''}${t.dx.toFixed(1)}, dY: ${t.dy > 0 ? '+' : ''}${t.dy.toFixed(1)}`;
            document.getElementById('hud-speed').innerHTML = `${t.speed.toFixed(1)} <span class="tele-unit">px/s</span>`;
            document.getElementById('hud-heading').innerHTML = `${t.direction} <span class="tele-unit">(${t.orientation.toFixed(1)}°)</span>`;
            document.getElementById('hud-net-dist').innerHTML = `${t.net_disp.toFixed(1)} <span class="tele-unit">px</span>`;
            document.getElementById('hud-total-dist').innerHTML = `${t.cumulative_dist.toFixed(1)} <span class="tele-unit">px</span>`;
          } else {
            document.getElementById('hud-target-title').innerHTML = `
              <i class="fa-solid fa-magnifying-glass" style="color: #f59e0b;"></i>
              <span>TARGET LOCKED [ID #${t.selected_id}] UNKNOWN</span>
            `;
            document.getElementById('hud-status-badge').className = "hud-status-pill status-searching";
            document.getElementById('hud-status-badge').innerHTML = '<i class="fa-solid fa-circle" style="font-size: 0.55rem;"></i> Out of View';

            document.getElementById('hud-pos').innerHTML = `-- <span class="tele-unit">px</span>`;
            document.getElementById('hud-disp').innerHTML = `dX: 0.0, dY: 0.0`;
            document.getElementById('hud-speed').innerHTML = `0.0 <span class="tele-unit">px/s</span>`;
            document.getElementById('hud-heading').innerHTML = `Searching...`;
            document.getElementById('hud-net-dist').innerHTML = `0.0 <span class="tele-unit">px</span>`;
            document.getElementById('hud-total-dist').innerHTML = `0.0 <span class="tele-unit">px</span>`;
          }

          document.getElementById('hud-frame-progress').innerHTML = `${data.current_frame} / ${data.total_frames}`;
          document.getElementById('hud-stream-fps').innerHTML = `${t.fps || 25.0} <span class="tele-unit">FPS</span>`;
        } else if (data.selected_id) {
          document.getElementById('hud-target-title').innerHTML = `
            <i class="fa-solid fa-crosshairs"></i>
            <span>TARGET LOCKED [ID #${data.selected_id}]</span>
          `;
          document.getElementById('hud-status-badge').className = "hud-status-pill status-searching";
          document.getElementById('hud-status-badge').innerHTML = '<i class="fa-solid fa-circle" style="font-size: 0.55rem;"></i> Initializing';
        }

        // Timeline Slider
        if (!isUserSeeking && data.total_frames > 0) {
          const slider = document.getElementById('frame-slider');
          slider.max = data.total_frames - 1;
          slider.value = data.current_frame;
          document.getElementById('frame-counter-label').innerText = `${data.current_frame} / ${data.total_frames}`;
        }

        // Pause button
        document.getElementById('pause-btn-icon').className = data.is_paused ? "fa-solid fa-play" : "fa-solid fa-pause";
        document.getElementById('pause-btn-text').innerText = data.is_paused ? "Resume" : "Pause";
        document.getElementById('auto-switch-btn').innerText = data.auto_switch ? "Auto-Lock: ON" : "Auto-Lock: OFF";

        // Kinematics Chart
        if (data.history && data.history.length > 0) {
          motionChart.data.labels = data.history.map(h => `F${h.frame}`);
          motionChart.data.datasets[0].data = data.history.map(h => h.step_dist);
          motionChart.data.datasets[1].data = data.history.map(h => h.speed);
          motionChart.update('none');
        }

        // Update Candidate List
        const container = document.getElementById('candidates-container');
        if (data.candidates && data.candidates.length > 0) {
          container.innerHTML = data.candidates.map(cand => `
            <div class="candidate-item ${cand.id === data.selected_id ? 'active' : ''}" onclick="selectId(${cand.id})">
              <div>
                <strong style="color: #38bdf8;">ID #${cand.id}</strong>
                <span class="badge badge-cat" style="margin-left: 5px;">${cand.category.toUpperCase()}</span>
                <span style="font-size: 0.72rem; color: #94a3b8; margin-left: 4px;">(${cand.confidence}%)</span>
              </div>
              <div style="font-size: 0.75rem; color: #94a3b8;">
                Pos: (${cand.centroid[0].toFixed(0)}, ${cand.centroid[1].toFixed(0)})
              </div>
            </div>
          `).join('');
        }

        // Populate video list if updated
        if (JSON.stringify(data.videos) !== JSON.stringify(currentVideoList)) {
          currentVideoList = data.videos;
          const dropdown = document.getElementById('video-dropdown');
          dropdown.innerHTML = currentVideoList.map(v => `
            <option value="${v.path}" ${v.path === data.current_video_path ? 'selected' : ''}>${v.name}</option>
          `).join('');
        }

      } catch (e) {
        console.error("Telemetry fetch error", e);
      }
    }

    setInterval(fetchTelemetry, 350);

    // Upload Video From Device
    async function uploadVideo(input) {
      if (!input.files || input.files.length === 0) return;
      const file = input.files[0];
      const formData = new FormData();
      formData.append("video_file", file);

      const statusEl = document.getElementById('upload-status');
      statusEl.innerHTML = `<span style="color: #f59e0b;"><i class="fa-solid fa-spinner fa-spin"></i> Uploading ${file.name}...</span>`;

      try {
        const resp = await fetch('/api/upload_video', { method: 'POST', body: formData });
        const res = await resp.json();
        if (res.status === 'success') {
          statusEl.innerHTML = `<span style="color: #10b981;"><i class="fa-solid fa-check"></i> Uploaded: ${res.filename}</span>`;
          // Force instant reload of video stream
          document.getElementById('video-stream').src = '/video_feed?t=' + Date.now();
          fetchTelemetry();
        } else {
          statusEl.innerHTML = `<span style="color: #f43f5e;"><i class="fa-solid fa-triangle-exclamation"></i> ${res.message}</span>`;
        }
      } catch (e) {
        statusEl.innerHTML = `<span style="color: #f43f5e;">Upload failed</span>`;
      }
    }

    // Split & Extract all frames
    async function extractAllFrames() {
      const card = document.getElementById('frame-split-card');
      const msg = document.getElementById('split-msg');
      card.style.display = 'block';
      msg.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Splitting video into individual frame images...`;

      const resp = await fetch('/api/split_frames', { method: 'POST' });
      const res = await resp.json();
      if (res.status === 'success') {
        msg.innerHTML = `<strong>${res.extracted_count} frames</strong> extracted successfully!<br>Saved to folder: <code>${res.output_directory}</code>`;
      } else {
        msg.innerHTML = `Frame extraction failed: ${res.message}`;
      }
    }

    // Timeline Scrubber & Step Controls
    function seekFrame(val) {
      isUserSeeking = true;
      fetch('/api/seek_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: parseInt(val) })
      }).then(() => { isUserSeeking = false; });
    }

    async function stepFrame(step) {
      await fetch('/api/step_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step: step })
      });
    }

    async function togglePause() {
      await fetch('/api/toggle_pause', { method: 'POST' });
    }

    async function toggleAutoSwitch() {
      await fetch('/api/toggle_auto_switch', { method: 'POST' });
    }

    async function selectId(id) {
      await fetch('/api/select_id', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: id })
      });
    }

    async function switchVideo(path) {
      await fetch('/api/change_video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: path })
      });
      // Force video feed reload
      document.getElementById('video-stream').src = '/video_feed?t=' + Date.now();
      fetchTelemetry();
    }

    // Click on video to select object
    async function handleVideoClick(e) {
      const img = document.getElementById('video-stream');
      const rect = img.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      const realX = clickX * scaleX;
      const realY = clickY * scaleY;

      await fetch('/api/click', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: realX, y: realY })
      });
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    port = 5000
    print(f"Starting Object Movement Detection & Tracking Web App on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
