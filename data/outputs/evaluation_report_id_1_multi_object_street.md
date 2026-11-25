# Object-Specific Tracking Evaluation & Analysis Report

## Executive Summary
- **Target Object**: ID `#1` (PERSON)
- **Video Source**: `multi_object_street.mp4` (768x432, 12.0 FPS)
- **Tracking Continuity**: 10.51% (68/647 frames)
- **Mean Detection Confidence**: 81.5%

---

## 1. Tracking Continuity & Stability Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Video Frames** | `647` | Total duration frames of the input stream |
| **Tracked Frames** | `68` | Frames where target ID was actively tracked |
| **Lost / Untracked Frames** | `563` | Frames target was occluded or absent |
| **Coverage Ratio** | **10.51%** | Persistence ratio across video timeline |
| **Track Duration** | `5.67 s` | Active tracked time in seconds |

---

## 2. Detection Quality & Confidence
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Confidence** | `0.815` | Average YOLO classification certainty |
| **Min Confidence** | `0.564` | Lowest confidence observed |
| **Max Confidence** | `0.894` | Peak confidence observed |
| **Confidence Std Dev**| `0.072` | Stability of detection certainty |

---

## 3. Kinematic & Motion Analysis (Tasks 9, 10, 11)
| Kinematic Metric | Value | Units | Interpretation |
| :--- | :--- | :--- | :--- |
| **Total Path Distance** | `372.68` | px | Cumulative integrated trajectory distance |
| **Net Displacement** | `335.22` | px | Euclidean distance from origin to end position |
| **Path Efficiency (Tortuosity)** | **0.899** | ratio | `1.0` = straight line, `<0.7` = winding/curved |
| **Mean Smoothed Speed** | `66.43` | px/s | Average velocity across track |
| **Peak Instantaneous Speed** | `173.41` | px/s | Maximum velocity spike detected |
| **Median Speed** | `67.01` | px/s | Central tendency of object velocity |
| **Acceleration Jitter Index** | `5.91` | px/s² | Trajectory smoothness measure |

---

## 4. Directional Distribution
Breakdown of heading sectors during motion:
- **Stationary**: 1.5%
- **North**: 35.3%
- **North-East**: 54.4%
- **East**: 8.8%
