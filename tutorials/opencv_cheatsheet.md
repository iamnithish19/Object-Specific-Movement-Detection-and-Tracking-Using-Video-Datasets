# OpenCV & Python Fundamentals for Video Object Tracking

## 1. Computer Vision Coordinate System

OpenCV uses a standard raster graphics coordinate system:
* **Origin `(0, 0)`**: Top-left corner of the image.
* **X-axis**: Extends horizontally to the **right** ($0 \le x < \text{width}$).
* **Y-axis**: Extends vertically **downward** ($0 \le y < \text{height}$).

```
(0, 0) ───────────────────────────> +X (width)
  │
  │     Bounding Box: [x1, y1, x2, y2]
  │     (x1, y1) ┌──────────────┐
  │              │  Centroid    │
  │              │    (cx, cy)  │
  │              └──────────────┘ (x2, y2)
  │
  v
 +Y (height)
```

> [!IMPORTANT]
> Because the Y-axis points downward in image coordinates, upward motion produces a **negative** $\Delta y$ ($\Delta y < 0$), while downward motion produces a **positive** $\Delta y$ ($\Delta y > 0$).
> When calculating standard mathematical orientation angles $\theta$, we compute:
> $$\theta = \operatorname{atan2}(-\Delta y, \Delta x) \times \frac{180}{\pi}$$
> This aligns standard 2D Cartesian mathematics ($+Y$ up, $+X$ right) with screen space.

---

## 2. BGR vs. RGB Color Spaces
* OpenCV loads and processes images in **BGR** (Blue, Green, Red) channel order by default.
* Matplotlib and web tools expect **RGB** (Red, Green, Blue).
* Conversion in OpenCV:
  ```python
  rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
  gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
  ```

---

## 3. Video Processing Pipeline in OpenCV

### 3.1 Capturing Video
```python
import cv2

cap = cv2.VideoCapture("input_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video stream
    # Process frame...
cap.release()
```

### 3.2 Writing Video
```python
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
writer.write(annotated_frame)
writer.release()
```

---

## 4. Key OpenCV Primitives for Object Tracking

| Primitive | OpenCV Function | Usage in Project |
| :--- | :--- | :--- |
| **Bounding Box** | `cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)` | Highlights detected and tracked objects |
| **Centroid Marker**| `cv2.circle(img, (cx, cy), radius, color, -1)` | Marks center of mass $(cx, cy)$ |
| **Trajectory Trail**| `cv2.polylines(img, [points], isClosed, color, thickness)` | Visualizes the historical path |
| **Velocity Vector** | `cv2.arrowedLine(img, pt1, pt2, color, thickness, tipLength)` | Points in instantaneous direction of motion |
| **HUD / Text** | `cv2.putText(img, text, org, font, scale, color, thickness)` | Displays live telemetry (ID, Speed, Direction) |
