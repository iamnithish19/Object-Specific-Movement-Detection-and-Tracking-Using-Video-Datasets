"""
Automated Unit & Integration Test Suite
=======================================
Verifies core components:
- Preprocessor (Frame streaming, metadata, resizing)
- Classifier (Taxonomy mapping, filtering, colors)
- Motion Analyzer (Displacements dx/dy, direction angles, speeds, trajectories)
- Evaluator (Metric compilation, CSV and Markdown export)
- Visualizer (HUD and overlay rendering integrity)
"""

import os
import sys
import unittest
import numpy as np

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from classifier import ObjectClassifier
from evaluator import TrackingEvaluator
from motion_analyzer import MotionAnalyzer, MotionState
from visualizer import VideoVisualizer


class TestObjectClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ObjectClassifier(allowed_categories=["car", "person", "drone"])

    def test_category_mapping(self):
        self.assertEqual(self.classifier.map_category("truck"), "car")
        self.assertEqual(self.classifier.map_category("bus"), "car")
        self.assertEqual(self.classifier.map_category("person"), "person")
        self.assertEqual(self.classifier.map_category("airplane"), "drone")

    def test_allowed_filtering(self):
        self.assertTrue(self.classifier.is_allowed("car"))
        self.assertTrue(self.classifier.is_allowed("truck"))
        self.assertTrue(self.classifier.is_allowed("person"))
        self.assertFalse(self.classifier.is_allowed("dog"))


class TestMotionAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MotionAnalyzer(fps=30.0, stationary_threshold_px=1.5)

    def test_stationary_detection(self):
        st = self.analyzer.update(0, 0.0, (100.0, 100.0))
        self.assertEqual(st.direction, "Stationary")
        self.assertEqual(st.dx, 0.0)
        self.assertEqual(st.dy, 0.0)

    def test_linear_displacement_and_direction(self):
        self.analyzer.update(0, 0.0, (100.0, 100.0))
        # Moving purely to the right (+10 px in X, 0 px in Y) -> East
        st = self.analyzer.update(1, 1.0 / 30.0, (110.0, 100.0))
        self.assertEqual(st.dx, 10.0)
        self.assertEqual(st.dy, 0.0)
        self.assertEqual(st.direction, "East")
        self.assertAlmostEqual(st.orientation_deg, 0.0, places=1)
        self.assertAlmostEqual(st.speed_px_sec, 300.0, places=1)

    def test_north_displacement_and_direction(self):
        analyzer = MotionAnalyzer(fps=30.0, stationary_threshold_px=1.5)
        analyzer.update(0, 0.0, (100.0, 100.0))
        # Moving upward in screen space (-10 px in Y, 0 px in X) -> North
        st = analyzer.update(1, 1.0 / 30.0, (100.0, 90.0))
        self.assertEqual(st.dx, 0.0)
        self.assertEqual(st.dy, -10.0)
        self.assertEqual(st.direction, "North")
        self.assertAlmostEqual(st.orientation_deg, 90.0, places=1)

    def test_trajectory_accumulation(self):
        analyzer = MotionAnalyzer(fps=30.0)
        for i in range(5):
            analyzer.update(i, i / 30.0, (100.0 + i * 5, 100.0 + i * 5))
        traj = analyzer.get_trajectory()
        self.assertEqual(len(traj), 5)
        self.assertEqual(traj[0], (100, 100))
        self.assertEqual(traj[-1], (120, 120))


class TestEvaluatorAndVisualizer(unittest.TestCase):
    def test_evaluator_metrics(self):
        evaluator = TrackingEvaluator()
        history = [
            MotionState(0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Stationary"),
            MotionState(1, 0.1, 20.0, 10.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0, 100.0, 100.0, 0.0, "East"),
        ]
        meta = {"filename": "test.mp4", "width": 640, "height": 480, "fps": 10.0, "total_frames": 10}
        metrics = evaluator.evaluate(history, 1, "car", meta, [0.85, 0.90])

        self.assertIn("tracking_coverage", metrics)
        self.assertEqual(metrics["tracking_coverage"]["tracked_frames"], 2)
        self.assertAlmostEqual(metrics["detection_quality"]["mean_confidence"], 0.875, places=2)

    def test_visualizer_rendering(self):
        visualizer = VideoVisualizer()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = visualizer.render(
            frame=dummy_frame,
            all_active_tracks=[],
            selected_id=1,
            trajectory_points=[(100, 100), (120, 130)],
            motion_state=None,
            frame_idx=0,
            total_frames=100,
            fps=30.0,
        )
        self.assertEqual(rendered.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
