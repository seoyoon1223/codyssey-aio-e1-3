"""Focused behavior checks for the Mini NPU simulator."""

import unittest

from mini_npu.analysis import analyze_dataset
from mini_npu.core import UNDECIDED, classify, classify_labels, mac, normalize_label, validate_matrix


class CoreTests(unittest.TestCase):
    def test_cross_and_x_mac_scores(self):
        cross = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        x_filter = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        self.assertEqual(mac(cross, cross), 5.0)
        self.assertEqual(mac(cross, x_filter), 1.0)

    def test_epsilon_tie(self):
        self.assertEqual(classify(0.9, 0.9 + 0.5e-9), UNDECIDED)
        self.assertEqual(classify_labels(1.0, 1.0), UNDECIDED)

    def test_label_aliases(self):
        self.assertEqual(normalize_label("+"), "Cross")
        self.assertEqual(normalize_label("cross"), "Cross")
        self.assertEqual(normalize_label("x"), "X")
        self.assertIsNone(normalize_label("circle"))

    def test_matrix_shape_validation(self):
        with self.assertRaisesRegex(ValueError, "row 2"):
            validate_matrix([[1, 2], [3]], 2)

    def test_dataset_cases_fail_independently_on_bad_shape(self):
        data = {
            "filters": {"size_2": {"cross": [[1, 0], [0, 1]], "x": [[0, 1], [1, 0]]}},
            "patterns": {
                "size_2_1": {"input": [[1, 0], [0, 1]], "expected": "cross"},
                "size_2_2": {"input": [[1, 0]], "expected": "+"},
            },
        }
        results = analyze_dataset(data)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["passed"])
        self.assertFalse(results[1]["passed"])


if __name__ == "__main__":
    unittest.main()
