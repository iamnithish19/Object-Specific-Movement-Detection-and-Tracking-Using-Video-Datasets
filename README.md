# Object-Specific Movement Detection and Tracking Using Video Datasets

A modular computer vision system built with **Python**, **OpenCV**, **YOLO**, **ByteTrack**, and **Matplotlib**. The system detects multiple objects in video feeds, assigns persistent IDs, allows interactive or automated selection of a specific object ID, and isolates and tracks only that object while estimating its frame-to-frame pixel displacements, heading orientation, compass direction, velocity, and trajectory path.

---

## 🆕 Device Video Upload & Frame-by-Frame Comparison (Web UI)

The updated live web interface at `http://localhost:5000` includes:
1. **Upload Video From Device**:
   - Drag-and-drop or select any `.mp4`, `.avi`, `.mov`, or `.mkv` video file directly from your local device.
   - Automatically stores it in `data/sample_videos/uploads/` and tracks it immediately.
2. **Split Video into Frames**:
   - Click **"Split & Save Frames"** to extract all video frames into individual JPEG files located at `data/outputs/extracted_frames/<video_name>/`.
3. **Consecutive Frame-by-Frame Comparison**:
   - Compares **Frame $t-1$** with **Frame $t$**:
     $$\Delta X = X_t - X_{t-1} \text{ px}, \quad \Delta Y = Y_t - Y_{t-1} \text{ px}$$
     $$\text{Movement Distance} = \sqrt{(\Delta X)^2 + (\Delta Y)^2} \text{ px}$$
     $$\text{Cumulative Distance} = \sum \text{Movement Distance} \text{ px}$$
   - Shows step displacement vector arrow on the video linking $(X_{t-1}, Y_{t-1}) \to (X_t, Y_t)$.
   - Frame step controls: `-1 Frame`, `+1 Frame`, `Pause/Resume`, and interactive timeline scrubber.
4. **Smart Target Switching**:
   - Never gets permanently stuck when an object walks off-screen.
   - Automatically re-locks onto visible moving objects (or click any object with the mouse to lock).

---

## 📌 Project Tasks Breakdown (Tasks 1 to 14)

| Task | Description | Implementation Module | Output / Demonstration |
| :--- | :--- | :--- | :--- |
| **Task 1** | Learn Python basics and OpenCV | `tutorials/01_python_opencv_basics.py`<br>`tutorials/opencv_cheatsheet.md` | NumPy image matrices, BGR color spaces, drawing shapes/HUD, video read/write |
| **Task 2** | Collect or download video datasets | `src/dataset_downloader.py` | Highway traffic, pedestrian walkways, multi-object urban street, aerial drone video |
| **Task 3** | Extract frames & preprocessing | `src/preprocessor.py` | Frame streaming generator, resizing, CLAHE illumination adjustment, Gaussian blur |
| **Task 4** | Detect objects in frames using YOLO | `src/detector.py` | Ultralytics YOLOv8 inference, bounding boxes `[x1, y1, x2, y2]`, confidence scores |
| **Task 5** | Classify detected objects | `src/classifier.py` | Taxonomy mapping to `person`, `car`, `drone`, `bicycle`, color palette assignment |
| **Task 6** | Assign unique IDs via tracking | `src/tracker.py` | Multi-object tracking with ByteTrack Kalman filter state association |
| **Task 7** | Allow user to select specific object ID | `src/selector.py` | 1) Mouse click in OpenCV window<br>2) Interactive terminal prompt<br>3) `--select-id` CLI flag<br>4) Automated candidate selection |
| **Task 8** | Track only selected object | `src/tracker.py`, `main.py` | Background objects visually dimmed/suppressed; tracking & telemetry focused strictly on chosen ID |
| **Task 9** | Calculate X & Y pixel displacement | `src/motion_analyzer.py` | Instantaneous $\Delta x_t, \Delta y_t$ and net origin displacements $\Delta X_{\text{net}}, \Delta Y_{\text{net}}$ |
| **Task 10**| Estimate direction & orientation | `src/motion_analyzer.py` | Mathematical heading angle $\theta \in [0^\circ, 360^\circ)$ and 8-point compass sector resolution |
| **Task 11**| Calculate speed & trajectory | `src/motion_analyzer.py` | Instantaneous/smoothed velocity in px/s (and km/h if calibrated) and coordinate trail |
| **Task 12**| Visualize movement & graphs | `src/visualizer.py`<br>`src/plotter.py` | Annotated video with glowing target brackets, heading arrow, trajectory trail, live HUD, and 4-panel analytical chart |
| **Task 13**| Test system with different videos | `evaluate_dataset.py` | Systematic evaluation across 4 benchmark datasets (Vehicles, Pedestrians, Urban Multimodal, Drone) |
| **Task 14**| Evaluate tracking accuracy & results | `src/evaluator.py` | Continuity, coverage ratio, path efficiency, confidence stats, CSV telemetry and Markdown reports |

---

## 📐 Mathematical Formulation

### 1. Spatial Coordinates & Image Inversion
In computer vision, $(0,0)$ is located at the top-left corner of the frame. The X-axis points **right**, while the Y-axis points **downward**.
Centroid of bounding box $[x_1, y_1, x_2, y_2]$:
$$x_c = \frac{x_1 + x_2}{2}, \quad y_c = \frac{y_1 + y_2}{2}$$

### 2. Instantaneous & Cumulative Pixel Displacement (Task 9)
$$\Delta x_t = x_t - x_{t-1}, \quad \Delta y_t = y_t - y_{t-1}$$
$$\Delta d_t = \sqrt{(\Delta x_t)^2 + (\Delta y_t)^2}$$
Net displacement from initial appearance $(x_0, y_0)$:
$$\Delta X_{\text{net}} = x_t - x_0, \quad \Delta Y_{\text{net}} = y_t - y_0$$
Cumulative path distance traveled:
$$D_{\text{total}} = \sum_{k=1}^t \Delta d_k$$

### 3. Orientation Angle & Compass Direction (Task 10)
Because image $Y$ points downward, Cartesian $v_y = -\Delta y_t$:
$$\theta = \operatorname{atan2}(-\Delta y_t, \Delta x_t) \times \frac{180}{\pi} \pmod{360^\circ}$$

Compass direction sector quantization:
$$\text{Sector Index} = \left\lfloor \frac{\theta + 22.5^\circ}{45^\circ} \right\rfloor \pmod 8$$
* **East**: $0^\circ \pm 22.5^\circ$
* **North-East**: $45^\circ \pm 22.5^\circ$
* **North**: $90^\circ \pm 22.5^\circ$
* **North-West**: $135^\circ \pm 22.5^\circ$
* **West**: $180^\circ \pm 22.5^\circ$
* **South-West**: $225^\circ \pm 22.5^\circ$
* **South**: $270^\circ \pm 22.5^\circ$
* **South-East**: $315^\circ \pm 22.5^\circ$
*(If $\Delta d_t < \text{threshold}$, object is classified as **Stationary**)*.

### 4. Pixel-Based Speed Estimation (Task 11)
$$v_t = \frac{\Delta d_t}{\Delta t} = \Delta d_t \times \text{FPS} \quad (\text{pixels / second})$$
Smoothed speed via sliding window of size $W$:
$$\bar{v}_t = \frac{1}{W} \sum_{i=0}^{W-1} v_{t-i}$$
If real-world scale factor $S = \text{pixels per meter}$ is provided:
$$v_{\text{km/h}} = \left(\frac{\bar{v}_t}{S}\right) \times 3.6$$

### 5. Path Efficiency & Tortuosity (Task 14)
$$\text{Path Efficiency} = \frac{\sqrt{(\Delta X_{\text{net}})^2 + (\Delta Y_{\text{net}})^2}}{D_{\text{total}}}$$
* A value of **$1.0$** indicates perfectly straight, direct linear motion.
* Values below **$0.7$** indicate curved, zig-zag, or wandering paths.

---

## 🗂️ Project Directory Structure

```
d:\MSD\VISTA PROJECT\
├── .venv\                           # Isolated Python virtual environment
├── data\
│   ├── sample_videos\               # Test datasets (traffic, pedestrian, street, drone)
│   └── outputs\                     # Rendered videos, PNG charts, CSV telemetry, MD reports
├── src\
│   ├── __init__.py
│   ├── config.py                    # Global system parameters and color settings
│   ├── preprocessor.py              # Task 3: Frame extraction and CLAHE enhancement
│   ├── detector.py                  # Task 4: YOLOv8 object detection
│   ├── classifier.py                # Task 5: Taxonomy classification & category mapping
│   ├── tracker.py                   # Task 6 & 8: ByteTrack multi-object tracking
│   ├── selector.py                  # Task 7: Interactive CLI, GUI mouse-click & auto selection
│   ├── motion_analyzer.py           # Tasks 9, 10, 11: Displacements, speed, heading, trajectory
│   ├── visualizer.py                # Task 12: Bounding boxes, IDs, velocity arrows & HUD
│   ├── plotter.py                   # Task 12: 4-panel Matplotlib analytical dashboard
│   ├── evaluator.py                 # Task 14: Metrics, continuity analysis, report generator
│   └── dataset_downloader.py        # Task 2: Automated dataset downloader & synthetic generator
├── tutorials\
│   ├── 01_python_opencv_basics.py    # Task 1: Practical tutorial script
│   └── opencv_cheatsheet.md          # Task 1: Theoretical and practical guide
├── tests\
│   └── test_tracking_system.py      # Automated unit and integration test suite
├── main.py                          # Unified CLI and pipeline runner
├── evaluate_dataset.py              # Task 13 & 14: Batch test on all datasets
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Activate Environment
In PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run Task 1 Tutorial (OpenCV Fundamentals)
```powershell
python tutorials\01_python_opencv_basics.py
```

### 3. Download / Verify Video Datasets (Task 2)
```powershell
python src\dataset_downloader.py
```

### 4. Run Tracking with Interactive Target ID Selection (Task 7)
* **Interactive Terminal Prompt**:
  ```powershell
  python main.py --video data\sample_videos\pedestrians_walk.mp4 --interactive
  ```
* **Interactive Mouse Click Selection (OpenCV Window)**:
  ```powershell
  python main.py --video data\sample_videos\pedestrians_walk.mp4 --gui-click
  ```
* **Direct Target ID Selection**:
  ```powershell
  python main.py --video data\sample_videos\pedestrians_walk.mp4 --select-id 2
  ```
* **Filter by Class Category**:
  ```powershell
  python main.py --video data\sample_videos\traffic_cars.mp4 --select-class car
  ```

### 5. Run the Multi-Dataset Benchmark (Tasks 13 & 14)
```powershell
python evaluate_dataset.py
```

### 6. Run Unit Tests
```powershell
python -m unittest tests\test_tracking_system.py
```

---

## 📊 Benchmark Results Summary (Tasks 13 & 14)

| Dataset | Target Object | Tracked Frames | Coverage % | Total Distance | Net Displacement | Mean Speed | Peak Speed | Path Efficiency | Dominant Direction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Traffic Scene** | Car (`#2`) | 8 | 2.1% | 53.4 px | 53.4 px | 85.8 px/s | 109.1 px/s | **0.999** | North (87.5%) |
| **Pedestrian Walkway** | Person (`#2`) | 56 | 9.4% | 550.6 px | 502.9 px | 120.7 px/s | 215.1 px/s | **0.913** | East (57.1%), NE (41.1%) |
| **Multi-Object Street** | Person (`#1`) | 68 | 10.5% | 372.7 px | 335.2 px | 66.4 px/s | 173.4 px/s | **0.899** | NE (54.4%), North (35.3%) |
| **Drone Surveillance** | Target (`#1`) | 26 | 14.4% | 174.1 px | 150.8 px | 201.6 px/s | 653.6 px/s | **0.866** | South-East (84.6%) |

All annotated output videos, 4-panel analytical dashboards (PNG), telemetry logs (CSV), and evaluation reports (Markdown) are saved in `data/outputs/`.
