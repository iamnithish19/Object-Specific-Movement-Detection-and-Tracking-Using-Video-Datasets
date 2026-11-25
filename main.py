"""
Object-Specific Movement Detection and Tracking System
======================================================
End-to-End Pipeline fulfilling Tasks 1 through 14:
- Object detection with YOLO
- Object classification (person, car, drone, bicycle, etc.)
- Multi-object tracking with ByteTrack persistent IDs
- Interactive / Automated user selection of specific object ID
- Isolated tracking of the selected object across video frames
- Frame-to-frame & net X/Y pixel displacement calculation
- Motion direction & orientation estimation
- Pixel-based speed calculation & trajectory generation
- Video visualization with bounding boxes, badges, velocity arrows & HUD
- Analytical multi-panel Matplotlib dashboard generation
- Quantitative tracking evaluation and report generation
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from classifier import ObjectClassifier
from config import OUTPUTS_DIR, SAMPLE_VIDEOS_DIR, TrackerConfig
from evaluator import TrackingEvaluator
from motion_analyzer import MotionAnalyzer
from plotter import MotionPlotter
from preprocessor import VideoPreprocessor
from selector import ObjectSelector
from tracker import ByteTrackTracker
from visualizer import VideoVisualizer


def run_pipeline(
    video_path: str,
    target_id: int = None,
    target_class: str = None,
    interactive_cli: bool = False,
    interactive_gui: bool = False,
    auto_select: bool = False,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.28,
    save_video: bool = True,
    pixels_per_meter: float = None,
    apply_clahe: bool = False,
    output_dir: str = OUTPUTS_DIR,
):
    print("=" * 70)
    print(" OBJECT-SPECIFIC MOVEMENT DETECTION & TRACKING PIPELINE ")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]

    # Task 3: Video Preprocessing & Ingestion
    print(f"\n[Task 3] Initializing video preprocessor for: {video_path}")
    preprocessor = VideoPreprocessor(video_path, apply_clahe=apply_clahe)
    meta = preprocessor.get_metadata()
    print(f"Video Info: {meta['width']}x{meta['height']} @ {meta['fps']} FPS, {meta['total_frames']} frames ({meta['duration_sec']}s)")

    # Task 5: Object Taxonomy & Classification
    print(f"\n[Task 5] Initializing Object Classifier...")
    classifier = ObjectClassifier()

    # Task 4 & Task 6: YOLO Detector & ByteTrack Tracker
    print(f"\n[Task 4 & 6] Loading YOLO & ByteTrack Tracker ({model_name})...")
    tracker = ByteTrackTracker(
        model_path=model_name,
        tracker_type="bytetrack.yaml",
        conf_threshold=conf_threshold,
        classifier=classifier,
    )

    # Task 7: Object Selector
    selector = ObjectSelector(
        target_id=target_id,
        target_class=target_class,
        interactive_cli=interactive_cli,
        interactive_gui=interactive_gui,
    )

    # Phase 1: Candidate Discovery & ID Selection (if target ID not pre-specified)
    if not selector.is_selected():
        print(f"\n[Task 7] Scanning video to discover candidate objects for user selection...")
        scan_frames_limit = min(180, meta["total_frames"])
        cap_scan = cv2.VideoCapture(video_path)
        first_frame_with_tracks = None
        active_tracks_snapshot = []

        for f_idx in range(scan_frames_limit):
            ret, frame = cap_scan.read()
            if not ret:
                break
            tracks = tracker.update(frame)
            if tracks and first_frame_with_tracks is None:
                first_frame_with_tracks = frame.copy()
                active_tracks_snapshot = tracks

            # Stop scanning early if we accumulated sufficient stable candidate tracks
            if len(tracker.all_seen_tracks) >= 1 and any(
                t["detections_count"] >= 8 for t in tracker.all_seen_tracks.values()
            ):
                break
        cap_scan.release()

        if not tracker.all_seen_tracks:
            raise RuntimeError(
                f"No objects detected in the first {scan_frames_limit} frames of {video_name}. "
                "Try lowering the confidence threshold (--conf) or checking video lighting."
            )

        # Execute selected strategy
        if interactive_gui and first_frame_with_tracks is not None:
            chosen_id = selector.select_via_mouse_click(first_frame_with_tracks, active_tracks_snapshot)
        elif interactive_cli:
            chosen_id = selector.prompt_cli_selection(tracker.all_seen_tracks)
        else:
            chosen_id = selector.auto_select(tracker.all_seen_tracks)
            print(f"[Auto-Select] Selected target ID: #{chosen_id} ({tracker.all_seen_tracks[chosen_id]['category']})")
    else:
        chosen_id = selector.selected_id
        print(f"\n[Task 7] User pre-selected target ID: #{chosen_id}")

    # Reset tracker state for a clean pass across entire video
    tracker.reset()

    # Tasks 9, 10, 11: Motion Analyzer
    analyzer = MotionAnalyzer(
        fps=meta["fps"],
        smoothing_window=5,
        stationary_threshold_px=1.5,
        pixels_per_meter=pixels_per_meter,
    )

    # Task 12: Visualizer & Video Writer Setup
    visualizer = VideoVisualizer(trail_length=80, show_hud=True)
    out_video_path = os.path.join(output_dir, f"tracked_id_{chosen_id}_{base_name}.mp4")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, meta["fps"], (meta["width"], meta["height"]))

    # Execution tracking variables
    print(f"\n[Task 8] Tracking ONLY Target ID #{chosen_id} across entire video...")
    start_time = time.time()
    tracked_frames_count = 0
    lost_frames_count = 0
    confidence_history = []
    target_category = "unknown"

    # Stream frames through pipeline
    for frame_idx, t_sec, raw_frame, proc_frame in preprocessor.stream_frames():
        active_tracks = tracker.update(proc_frame)

        # Locate selected object
        selected_obj = next((t for t in active_tracks if t.track_id == chosen_id), None)

        motion_state = None
        if selected_obj is not None:
            tracked_frames_count += 1
            confidence_history.append(selected_obj.confidence)
            target_category = selected_obj.category

            # Tasks 9, 10, 11: Compute displacement, speed, direction, orientation
            motion_state = analyzer.update(
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
                centroid=selected_obj.centroid,
            )
        else:
            if tracked_frames_count > 0:
                lost_frames_count += 1

        # Task 12: Render frame annotations
        if save_video:
            annotated_frame = visualizer.render(
                frame=raw_frame,
                all_active_tracks=active_tracks,
                selected_id=chosen_id,
                trajectory_points=analyzer.get_trajectory(),
                motion_state=motion_state,
                frame_idx=frame_idx,
                total_frames=meta["total_frames"],
                fps=meta["fps"],
            )
            writer.write(annotated_frame)

        # Console progress indicator every 30 frames
        if (frame_idx + 1) % 30 == 0 or frame_idx == meta["total_frames"] - 1:
            spd = f"{motion_state.speed_smoothed:.1f} px/s" if motion_state else "N/A"
            heading = f"{motion_state.direction} ({motion_state.orientation_deg:.0f}°)" if motion_state else "N/A"
            sys.stdout.write(
                f"\rFrame [{frame_idx + 1}/{meta['total_frames']}] | Speed: {spd:<12} | Heading: {heading:<18}"
            )
            sys.stdout.flush()

    if writer is not None:
        writer.release()
    total_elapsed = time.time() - start_time
    proc_fps = meta["total_frames"] / total_elapsed if total_elapsed > 0 else 0
    print(f"\nProcessing complete in {total_elapsed:.2f}s ({proc_fps:.1f} FPS processing speed).")
    if save_video:
        print(f"[Task 12] Annotated Video saved to: {out_video_path}")

    # Task 12: Analytical Graphs Dashboard (Matplotlib)
    print("\n[Task 12] Generating multi-panel analytical graphs...")
    plotter = MotionPlotter(output_dir=output_dir)
    chart_path = plotter.generate_dashboard(
        history=analyzer.history,
        target_id=chosen_id,
        category=target_category,
        video_name=video_name,
        frame_width=meta["width"],
        frame_height=meta["height"],
    )

    # Task 14: Evaluation & Accuracy Analysis
    print("\n[Task 14] Evaluating tracking accuracy and compiling metrics...")
    evaluator = TrackingEvaluator(output_dir=output_dir)
    metrics = evaluator.evaluate(
        history=analyzer.history,
        target_id=chosen_id,
        category=target_category,
        video_metadata=meta,
        confidence_history=confidence_history,
        lost_frames_count=lost_frames_count,
    )

    # Export telemetry CSV & Markdown report
    csv_path = evaluator.export_csv_telemetry(analyzer.history, chosen_id, video_name)
    md_path = evaluator.export_markdown_report(metrics, chosen_id, video_name)

    # Print summary highlights
    print("\n" + "=" * 70)
    print(" TRACKING & MOVEMENT ESTIMATION SUMMARY ")
    print("=" * 70)
    cov = metrics["tracking_coverage"]
    kin = metrics["motion_kinematics"]
    print(f"Target ID               : #{chosen_id} ({target_category.upper()})")
    print(f"Tracking Coverage       : {cov['coverage_percentage']}% ({cov['tracked_frames']}/{cov['total_video_frames']} frames)")
    print(f"Total Distance Traveled : {kin['total_path_distance_px']} pixels")
    print(f"Net Displacement        : {kin['net_displacement_px']} pixels")
    print(f"Path Efficiency Ratio   : {kin['path_efficiency_ratio']} (1.0 = direct linear path)")
    print(f"Mean Smoothed Speed     : {kin['mean_speed_px_s']} px/s")
    print(f"Peak Speed              : {kin['peak_speed_px_s']} px/s")
    print(f"Directional Distribution: {metrics['directional_distribution_pct']}")
    print("=" * 70)
    print(f"Annotated Video Output  : {out_video_path}")
    print(f"Analytical Graph Chart  : {chart_path}")
    print(f"Telemetry Log (CSV)     : {csv_path}")
    print(f"Evaluation Report (MD)  : {md_path}")
    print("=" * 70)

    return {
        "metrics": metrics,
        "video_path": out_video_path,
        "chart_path": chart_path,
        "csv_path": csv_path,
        "report_path": md_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Object-Specific Movement Detection and Tracking Using Video Datasets"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=os.path.join(SAMPLE_VIDEOS_DIR, "pedestrians_walk.mp4"),
        help="Path to input video file (default: pedestrians_walk.mp4)",
    )
    parser.add_argument(
        "--select-id",
        type=int,
        default=None,
        help="Task 7: Specific object ID to track exclusively (e.g. 1, 2)",
    )
    parser.add_argument(
        "--select-class",
        type=str,
        default=None,
        help="Filter candidates by category (e.g. person, car, drone)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Task 7: Prompt interactive console ID selection",
    )
    parser.add_argument(
        "--gui-click",
        action="store_true",
        help="Task 7: Select target object ID via mouse click in OpenCV window",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Automatically select the primary object candidate (default)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.28,
        help="YOLO detection confidence threshold (default: 0.28)",
    )
    parser.add_argument(
        "--ppm",
        type=float,
        default=None,
        help="Calibration scale: pixels per real-world meter",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Enable CLAHE illumination enhancement in preprocessing",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip rendering annotated video file to save processing time",
    )

    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        target_id=args.select_id,
        target_class=args.select_class,
        interactive_cli=args.interactive,
        interactive_gui=args.gui_click,
        auto_select=args.auto,
        model_name=args.model,
        conf_threshold=args.conf,
        save_video=not args.no_video,
        pixels_per_meter=args.ppm,
        apply_clahe=args.clahe,
    )


if __name__ == "__main__":
    main()
