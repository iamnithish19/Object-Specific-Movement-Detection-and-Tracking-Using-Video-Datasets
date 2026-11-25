"""
Task 13 & 14: Multi-Dataset Benchmark Evaluation
================================================
Runs the tracking and motion analysis system across all sample video datasets:
1. traffic_cars.mp4 (Car / Vehicle dataset)
2. pedestrians_walk.mp4 (Pedestrian dataset)
3. multi_object_street.mp4 (Multi-class street dataset)
4. drone_surveillance.mp4 (Aerial drone dataset)

Produces consolidated benchmark comparison tables, telemetry, and graphs.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from config import DATA_DIR, OUTPUTS_DIR, SAMPLE_VIDEOS_DIR
from main import run_pipeline


def run_benchmark():
    print("=" * 75)
    print(" TASK 13 & 14: MULTI-DATASET TRACKING & MOVEMENT BENCHMARK ")
    print("=" * 75)

    datasets = [
        {
            "name": "Traffic Scene (Vehicles)",
            "file": os.path.join(SAMPLE_VIDEOS_DIR, "traffic_cars.mp4"),
            "target_class": "car",
            "conf": 0.25,
            "max_frames": None,
        },
        {
            "name": "Pedestrian Walkway (Person)",
            "file": os.path.join(SAMPLE_VIDEOS_DIR, "pedestrians_walk.mp4"),
            "target_class": "person",
            "conf": 0.28,
            "max_frames": None,
        },
        {
            "name": "Multi-Object Street Scene (Pedestrian / Car / Bike)",
            "file": os.path.join(SAMPLE_VIDEOS_DIR, "multi_object_street.mp4"),
            "target_class": None,
            "conf": 0.28,
            "max_frames": None,
        },
        {
            "name": "Aerial Drone Surveillance (Drone)",
            "file": os.path.join(SAMPLE_VIDEOS_DIR, "drone_surveillance.mp4"),
            "target_class": None,
            "conf": 0.15,
            "max_frames": None,
        },
    ]

    results = []

    for idx, d in enumerate(datasets, 1):
        if not os.path.exists(d["file"]):
            print(f"[Warning] Dataset file not found: {d['file']}. Skipping.")
            continue

        print(f"\n>>> Running Benchmark {idx}/{len(datasets)}: {d['name']} ({os.path.basename(d['file'])})")
        t0 = time.time()
        res = run_pipeline(
            video_path=d["file"],
            target_class=d["target_class"],
            auto_select=True,
            conf_threshold=d["conf"],
            save_video=True,
            output_dir=OUTPUTS_DIR,
        )
        elapsed = time.time() - t0
        res["benchmark_name"] = d["name"]
        res["elapsed_sec"] = elapsed
        results.append(res)

    # Compile Consolidated Benchmark Markdown Report
    summary_path = os.path.join(OUTPUTS_DIR, "multi_dataset_benchmark_report.md")
    with open(summary_path, mode="w", encoding="utf-8") as f:
        f.write("# Multi-Dataset Tracking & Movement Benchmark Report\n\n")
        f.write("Evaluation of object-specific detection, tracking, speed, and trajectory analysis across distinct video domains.\n\n")
        f.write("## Comparative Performance Summary\n\n")
        f.write("| Dataset | Target Object | Tracked Frames | Coverage % | Total Dist (px) | Net Disp (px) | Mean Speed (px/s) | Peak Speed (px/s) | Path Efficiency |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for r in results:
            m = r["metrics"]
            cov = m.get("tracking_coverage", {})
            kin = m.get("motion_kinematics", {})
            f.write(
                f"| **{r['benchmark_name']}** | #{m.get('target_id')} ({m.get('category','').upper()}) | "
                f"{cov.get('tracked_frames')} | {cov.get('coverage_percentage')}% | "
                f"{kin.get('total_path_distance_px')} px | {kin.get('net_displacement_px')} px | "
                f"{kin.get('mean_speed_px_s')} px/s | {kin.get('peak_speed_px_s')} px/s | "
                f"{kin.get('path_efficiency_ratio')} |\n"
            )

        f.write("\n\n## Artifacts Generated\n\n")
        for r in results:
            m = r["metrics"]
            f.write(f"### {r['benchmark_name']}\n")
            f.write(f"- **Annotated Video**: `{os.path.basename(r['video_path'])}`\n")
            f.write(f"- **Analytical Chart**: `{os.path.basename(r['chart_path'])}`\n")
            f.write(f"- **Telemetry Log**: `{os.path.basename(r['csv_path'])}`\n")
            f.write(f"- **Individual Report**: `{os.path.basename(r['report_path'])}`\n\n")

    print("\n" + "=" * 75)
    print(" ALL BENCHMARKS COMPLETED SUCCESSFULLY ")
    print(f" Master Summary Report: {summary_path}")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
