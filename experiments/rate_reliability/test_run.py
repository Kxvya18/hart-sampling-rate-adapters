import unittest

import numpy as np
import torch

from run import extract, gate_features, metrics, pool_features


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_encoding = torch.nn.Identity()
        self.fft_encoding = torch.nn.Identity()

    def forward(self, batch, **_kwargs):
        patches = batch[:, :4, :].permute(0, 2, 1).unsqueeze(2)
        time = self.time_encoding(patches)
        frequency = self.fft_encoding(2 * patches)
        return type("Output", (), {"last_hidden_state": time + frequency})()


class TestRun(unittest.TestCase):
    def test_one_pass_embeddings(self):
        x = np.ones((5, 120, 6), dtype=np.float32)
        result = extract(DummyBackbone(), x, torch.device("cpu"), 2)
        self.assertEqual(result["fused"].shape, (5, 48))
        self.assertTrue(np.all(result["frequency"] == 2 * result["time"]))

    def test_zero_mean_motion_preserves_variation(self):
        stationary = torch.zeros(1, 1, 4, 2)
        moving = torch.tensor([-1., 1., -1., 1.]).reshape(1, 1, 4, 1).repeat(1, 1, 1, 2)
        self.assertTrue(torch.equal(stationary.mean(dim=2), moving.mean(dim=2)))
        self.assertFalse(torch.equal(pool_features(stationary), pool_features(moving)))

    def test_gate_features_and_metrics(self):
        p = {"time": np.array([[0.8, 0.2], [0.3, 0.7]]),
             "frequency": np.array([[0.6, 0.4], [0.1, 0.9]])}
        self.assertEqual(gate_features(p, np.array([0.2, 0.3]), 15).shape, (2, 7))
        result = metrics(np.array([0, 1]), p["time"], np.array([0, 1]))
        self.assertEqual(result["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
