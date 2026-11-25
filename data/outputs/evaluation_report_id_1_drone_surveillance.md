# Object-Specific Tracking Evaluation & Analysis Report

## Executive Summary
- **Target Object**: ID `#1` (PERSON)
- **Video Source**: `drone_surveillance.mp4` (800x600, 30.0 FPS)
- **Tracking Continuity**: 14.44% (26/180 frames)
- **Mean Detection Confidence**: 38.6%

---

## 1. Tracking Continuity & Stability Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Video Frames** | `180` | Total duration frames of the input stream |
| **Tracked Frames** | `26` | Frames where target ID was actively tracked |
| **Lost / Untracked Frames** | `154` | Frames target was occluded or absent |
| **Coverage Ratio** | **14.44%** | Persistence ratio across video timeline |
| **Track Duration** | `0.87 s` | Active tracked time in seconds |

---

## 2. Detection Quality & Confidence
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Confidence** | `0.386` | Average YOLO classification certainty |
| **Min Confidence** | `0.155` | Lowest confidence observed |
| **Max Confidence** | `0.63` | Peak confidence observed |
| **Confidence Std Dev**| `0.136` | Stability of detection certainty |

---

## 3. Kinematic & Motion Analysis (Tasks 9, 10, 11)
| Kinematic Metric | Value | Units | Interpretation |
| :--- | :--- | :--- | :--- |
| **Total Path Distance** | `174.13` | px | Cumulative integrated trajectory distance |
| **Net Displacement** | `150.77` | px | Euclidean distance from origin to end position |
| **Path Efficiency (Tortuosity)** | **0.866** | ratio | `1.0` = straight line, `<0.7` = winding/curved |
| **Mean Smoothed Speed** | `201.59` | px/s | Average velocity across track |
| **Peak Instantaneous Speed** | `653.56` | px/s | Maximum velocity spike detected |
| **Median Speed** | `210.06` | px/s | Central tendency of object velocity |
| **Acceleration Jitter Index** | `45.98` | px/s² | Trajectory smoothness measure |

---

## 4. Directional Distribution
Breakdown of heading sectors during motion:
- **Stationary**: 3.8%
- **South**: 3.8%
- **South-East**: 84.6%
- **East**: 7.7%
