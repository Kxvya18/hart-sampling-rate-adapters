import unittest

import numpy as np

from data import change_rate, high_band_fraction, split_subjects, stable_window_indices


class TestRateProtocol(unittest.TestCase):
    def test_filter_precedes_uniform_cap(self):
        clean = np.zeros((120 * 6, 6), dtype=np.float32)
        labels = np.repeat([1, 2, 1, 2, 1, 2], 120)
        labels[:120] = np.tile([1, 2], 60)
        clean[120] = np.nan
        starts, targets = stable_window_indices(clean, labels, 3, np.random.default_rng(7))
        self.assertEqual(len(starts), 3)
        self.assertTrue(set(starts).issubset({240, 360, 480, 600}))
        np.testing.assert_array_equal(targets, labels[starts])

    def test_shapes_and_antialiasing(self):
        t = np.arange(120) / 30
        high = np.sin(2 * np.pi * 11 * t)
        x = np.tile(high[None, :, None], (2, 1, 6)).astype(np.float32)
        aa = change_rate(x, "aa10")
        naive = change_rate(x, "naive10")
        self.assertEqual(aa.shape, x.shape)
        self.assertLess(float(np.mean(aa ** 2)), float(np.mean(naive ** 2)))
        self.assertTrue(np.isfinite(high_band_fraction(aa)).all())

    def test_subject_disjointness(self):
        groups = np.repeat(np.arange(12).astype(str), 3)
        subjects, indices = split_subjects(groups, 7)
        for left in subjects:
            for right in subjects:
                if left != right:
                    self.assertFalse(set(groups[indices[left]]) & set(groups[indices[right]]))


if __name__ == "__main__":
    unittest.main()
