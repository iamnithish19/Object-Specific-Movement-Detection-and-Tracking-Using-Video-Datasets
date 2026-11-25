"""
Task 5: Object Classification & Category Mapping
================================================
Classifies detected objects into categories such as person, car, or drone.
Handles class aliases, taxonomy grouping, filtering, and color palette assignment.
"""

from typing import Dict, List, Optional, Tuple

# Mapping COCO dataset classes into higher-level target project categories
CATEGORY_MAPPINGS: Dict[str, str] = {
    # Person category
    "person": "person",
    # Vehicle / Car category
    "car": "car",
    "motorcycle": "car",
    "bus": "car",
    "truck": "car",
    "van": "car",
    # Bicycle
    "bicycle": "bicycle",
    # Aerial / Drone category
    # Standard COCO doesn't have explicit 'drone' class, but uses airplane/bird
    # or synthetic drone detector
    "airplane": "drone",
    "bird": "drone",
    "drone": "drone",
}

# Distinguishable BGR color palette for visualization
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person": (255, 128, 0),     # Vibrant Blue/Cyan (BGR)
    "car": (0, 200, 255),        # Amber / Orange-Yellow
    "drone": (255, 0, 255),      # Magenta / Violet
    "bicycle": (50, 220, 100),   # Green
    "default": (180, 180, 180),  # Light Gray
}


class ObjectClassifier:
    """Manages object category taxonomy, mapping, filtering, and visual aesthetics."""

    def __init__(self, allowed_categories: Optional[List[str]] = None):
        """
        :param allowed_categories: e.g. ["person", "car", "drone"]. If None, all are allowed.
        """
        self.allowed_categories = [c.lower() for c in allowed_categories] if allowed_categories else None

    def map_category(self, raw_class_name: str) -> str:
        """Map raw YOLO class name to target project taxonomy."""
        cleaned = raw_class_name.lower().strip()
        return CATEGORY_MAPPINGS.get(cleaned, cleaned)

    def is_allowed(self, raw_class_name: str) -> bool:
        """Check if detected object belongs to the enabled category set."""
        if self.allowed_categories is None:
            return True
        category = self.map_category(raw_class_name)
        return category in self.allowed_categories or raw_class_name.lower() in self.allowed_categories

    def get_color(self, category_name: str) -> Tuple[int, int, int]:
        """Returns BGR color tuple for a category."""
        return CATEGORY_COLORS.get(category_name.lower(), CATEGORY_COLORS["default"])

    def format_label(self, track_id: Optional[int], category: str, confidence: float) -> str:
        """Creates a standardized human-readable badge."""
        id_str = f"#{track_id} " if track_id is not None else ""
        return f"{id_str}{category.upper()} {int(confidence * 100)}%"


if __name__ == "__main__":
    classifier = ObjectClassifier(allowed_categories=["person", "car", "drone"])
    print("Test mapping:")
    for sample in ["bus", "truck", "person", "airplane", "cat", "car"]:
        cat = classifier.map_category(sample)
        allowed = classifier.is_allowed(sample)
        color = classifier.get_color(cat)
        print(f" - {sample:<10} -> Category: {cat:<8} | Allowed: {allowed!s:<5} | Color: {color}")
