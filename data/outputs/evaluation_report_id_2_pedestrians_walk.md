# Object-Specific Tracking Evaluation & Analysis Report

## Executive Summary
- **Target Object**: ID `#2` (PERSON)
- **Video Source**: `pedestrians_walk.mp4` (768x432, 12.0 FPS)
- **Tracking Continuity**: 9.4% (56/596 frames)
- **Mean Detection Confidence**: 83.4%

---

## 1. Tracking Continuity & Stability Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Video Frames** | `596` | Total duration frames of the input stream |
| **Tracked Frames** | `56` | Frames where target ID was actively tracked |
| **Lost / Untracked Frames** | `520` | Frames target was occluded or absent |
| **Coverage Ratio** | **9.4%** | Persistence ratio across video timeline |
| **Track Duration** | `4.67 s` | Active tracked time in seconds |

---

## 2. Detection Quality & Confidence
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Confidence** | `0.834` | Average YOLO classification certainty |
| **Min Confidence** | `0.598` | Lowest confidence observed |
| **Max Confidence** | `0.924` | Peak confidence observed |
| **Confidence Std Dev**| `0.084` | Stability of detection certainty |

---

## 3. Kinematic & Motion Analysis (Tasks 9, 10, 11)
| Kinematic Metric | Value | Units | Interpretation |
| :--- | :--- | :--- | :--- |
| **Total Path Distance** | `550.61` | px | Cumulative integrated trajectory distance |
| **Net Displacement** | `502.95` | px | Euclidean distance from origin to end position |
| **Path Efficiency (Tortuosity)** | **0.913** | ratio | `1.0` = straight line, `<0.7` = winding/curved |
| **Mean Smoothed Speed** | `120.69` | px/s | Average velocity across track |
| **Peak Instantaneous Speed** | `215.08` | px/s | Maximum velocity spike detected |
| **Median Speed** | `125.56` | px/s | Central tendency of object velocity |
| **Acceleration Jitter Index** | `12.46` | px/s² | Trajectory smoothness measure |

---

## 4. Directional Distribution
Breakdown of heading sectors during motion:
- **Stationary**: 1.8%
- **North-East**: 41.1%
- **East**: 57.1%
