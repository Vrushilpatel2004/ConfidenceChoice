import unittest

from src.abstention import summarize_abstention


class AbstentionTests(unittest.TestCase):
    def test_threshold_boundary_and_answered_accuracy(self):
        result = summarize_abstention(
            [0, 1, 1], [0, 0, 1], [[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]], 0.8
        )
        self.assertEqual(result, {
            "answered": 2, "skipped": 1, "coverage": 2 / 3, "accuracy": 1.0,
        })

    def test_no_answers_has_undefined_accuracy(self):
        result = summarize_abstention([0], [0], [[0.6, 0.4]], 0.8)
        self.assertEqual(result, {
            "answered": 0, "skipped": 1, "coverage": 0.0, "accuracy": None,
        })
