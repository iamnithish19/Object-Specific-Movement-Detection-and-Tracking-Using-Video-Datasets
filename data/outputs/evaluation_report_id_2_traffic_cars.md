# Object-Specific Tracking Evaluation & Analysis Report

## Executive Summary
- **Target Object**: ID `#2` (CAR)
- **Video Source**: `traffic_cars.mp4` (768x432, 12.5 FPS)
- **Tracking Continuity**: 2.12% (8/377 frames)
- **Mean Detection Confidence**: 50.6%

---

## 1. Tracking Continuity & Stability Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Video Frames** | `377` | Total duration frames of the input stream |
| **Tracked Frames** | `8` | Frames where target ID was actively tracked |
| **Lost / Untracked Frames** | `308` | Frames target was occluded or absent |
| **Coverage Ratio** | **2.12%** | Persistence ratio across video timeline |
| **Track Duration** | `0.64 s` | Active tracked time in seconds |

---

## 2. Detection Quality & Confidence
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Confidence** | `0.506` | Average YOLO classification certainty |
| **Min Confidence** | `0.386` | Lowest confidence observed |
| **Max Confidence** | `0.656` | Peak confidence observed |
| **Confidence Std Dev**| `0.099` | Stability of detection certainty |

---

## 3. Kinematic & Motion Analysis (Tasks 9, 10, 11)
| Kinematic Metric | Value | Units | Interpretation |
| :--- | :--- | :--- | :--- |
| **Total Path Distance** | `53.44` | px | Cumulative integrated trajectory distance |
| **Net Displacement** | `53.36` | px | Euclidean distance from origin to end position |
| **Path Efficiency (Tortuosity)** | **0.999** | ratio | `1.0` = straight line, `<0.7` = winding/curved |
| **Mean Smoothed Speed** | `85.76` | px/s | Average velocity across track |
| **Peak Instantaneous Speed** | `109.06` | px/s | Maximum velocity spike detected |
| **Median Speed** | `95.92` | px/s | Central tendency of object velocity |
| **Acceleration Jitter Index** | `17.9` | px/s² | Trajectory smoothness measure |

---

## 4. Directional Distribution
Breakdown of heading sectors during motion:
- **Stationary**: 12.5%
- **North**: 87.5%
