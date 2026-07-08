import unittest

import numpy as np

from od_platform.visualization import (
    BeautifyVisualizer,
    Detection,
    DrawStyle,
    detections_from_yolo_result,
)


class TestVisualization(unittest.TestCase):
    def test_beautify_visualizer_draws_on_image_copy(self) -> None:
        image = np.full((120, 180, 3), 40, dtype=np.uint8)
        detection = Detection(box=(20, 25, 100, 90), confidence=0.875, label="person")
        visualizer = BeautifyVisualizer(
            labels=["person"],
            label_mapping={"person": "person_cn"},
            color_mapping={"person": (0, 255, 0)},
        )

        annotated = visualizer.draw(
            image,
            [detection],
            DrawStyle(font_size=14, line_width=2, padding_x=4, padding_y=4, radius=3),
            use_label_mapping=True,
        )

        self.assertEqual(annotated.shape, image.shape)
        self.assertFalse(np.array_equal(annotated, image))
        self.assertTrue(np.array_equal(image, np.full((120, 180, 3), 40, dtype=np.uint8)))

    def test_empty_detections_return_image_copy(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        visualizer = BeautifyVisualizer(labels=["person"])

        annotated = visualizer.draw(image, [])

        self.assertTrue(np.array_equal(annotated, image))
        self.assertIsNot(annotated, image)

    def test_detections_from_yolo_result(self) -> None:
        class FakeBoxes:
            data = np.array([[1, 2, 30, 40, 0.91, 0], [5, 6, 50, 60, 0.75, 1]], dtype=float)

        class FakeResult:
            boxes = FakeBoxes()
            names = {0: "person", 1: "helmet"}

        detections = detections_from_yolo_result(
            FakeResult(),
            color_mapping={"helmet": (255, 0, 0)},
        )

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].box, (1, 2, 30, 40))
        self.assertEqual(detections[0].label, "person")
        self.assertAlmostEqual(detections[0].confidence, 0.91)
        self.assertEqual(detections[1].color, (255, 0, 0))

    def test_from_yolo_results_converts_explicit_arrays(self) -> None:
        detections = BeautifyVisualizer.from_yolo_results(
            boxes=np.array([[1, 2, 30, 40]], dtype=float),
            confidences=np.array([0.8], dtype=float),
            labels=["scratch"],
            color_mapping={"scratch": (0, 0, 255)},
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].box, (1, 2, 30, 40))
        self.assertEqual(detections[0].label, "scratch")
        self.assertAlmostEqual(detections[0].confidence, 0.8)
        self.assertEqual(detections[0].color, (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
