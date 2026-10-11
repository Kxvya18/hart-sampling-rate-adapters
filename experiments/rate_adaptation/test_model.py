import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from model import AdaptedHART, EncodingAdapter, alignment_loss
from views import rate_view, shared_band
from diagnostics import class_diagnostics, training_order


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=4)
        self.time_encoding = nn.Linear(3, 4)
        self.fft_encoding = nn.Linear(3, 4)
        self.mixer = nn.Linear(4, 4)

    def forward(self, x, **_kwargs):
        patches = x.reshape(len(x), 4, 3, 2).permute(0, 3, 1, 2)
        time = self.time_encoding(patches)
        frequency = self.fft_encoding(patches.flip(2))
        return SimpleNamespace(last_hidden_state=self.mixer(torch.cat((time, frequency), dim=2)))


class LongToyBackbone(ToyBackbone):
    def forward(self, x, **kwargs):
        return super().forward(x[:, :12], **kwargs)


class TestArchitecture(unittest.TestCase):
    def test_identity_initialization(self):
        adapter = EncodingAdapter(width=4, rank=2)
        tokens = torch.randn(3, 2, 4, 4)
        self.assertTrue(torch.equal(tokens, adapter(tokens, torch.tensor([30., 15., 10.]))))

    def test_gradient_reaches_adapter_but_not_backbone(self):
        model = AdaptedHART(ToyBackbone(), 2, 3, rank=2)
        model.train()
        logits, _ = model(torch.randn(4, 12, 2), torch.tensor([30., 15., 10., 30.]))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
        self.assertGreater(model.adapters['time'].up.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertFalse(model.backbone.training)
        model.close()
        self.assertFalse(model.backbone.time_encoding._forward_hooks)

    def test_common_band_removes_unshared_frequencies(self):
        t = np.arange(120) / 30
        low = np.sin(2 * np.pi * 2 * t)
        high = np.sin(2 * np.pi * 9 * t)
        values = (low + high)[None, :, None].astype(np.float32)
        filtered = shared_band(values, 10)
        np.testing.assert_allclose(filtered[0, :, 0], low, atol=1e-6)
        for rate in (15, 10, 12, 6):
            self.assertEqual(rate_view(values, rate).shape, values.shape)

    def test_alignment_identity(self):
        a = torch.randn(8, 10)
        self.assertAlmostEqual(alignment_loss(a, a).item(), 0., places=6)

    def test_blind_adapter_does_not_use_rate(self):
        adapter = EncodingAdapter(width=4, rank=2, conditioned=False)
        nn.init.normal_(adapter.up.weight)
        tokens = torch.randn(3, 2, 4, 4)
        self.assertTrue(torch.equal(adapter(tokens, torch.full((3,), 30.)),
                                    adapter(tokens, torch.full((3,), 6.))))

    def test_training_balancing_uses_training_indices_only(self):
        labels = np.concatenate((np.zeros(9900, dtype=int), np.ones(100, dtype=int), [99]))
        indices = np.arange(10000)
        sampled = training_order(indices, labels, np.random.default_rng(7), "sqrt-balanced")
        self.assertNotIn(10000, sampled)
        self.assertGreater(np.sum(labels[sampled] == 1), 300)
        uniform = training_order(indices, labels, np.random.default_rng(7), "uniform")
        np.testing.assert_array_equal(np.sort(uniform), indices)

    def test_metrics_expose_absent_classes(self):
        from run import score
        y = np.array([0, 0, 1])
        p = np.array([[.9, .05, .05], [.9, .05, .05], [.05, .9, .05]])
        metrics = score(y, p, np.arange(3))
        self.assertAlmostEqual(metrics['macro_f1'], 2 / 3)
        self.assertAlmostEqual(metrics['supported_macro_f1'], 1.)
        self.assertEqual(metrics['supported_classes'], 2)
        absent = class_diagnostics(y, p, np.arange(3))[-1]
        self.assertEqual(absent['support'], 0)
        self.assertEqual(absent['predicted_count'], 0)

    def test_cached_end_to_end_reports_and_output_protection(self):
        import run
        import pandas as pd
        rng = np.random.default_rng(4)
        x = rng.normal(size=(24, 120, 2)).astype(np.float32)
        labels = np.tile([1, 2, 7], 8)
        groups = np.repeat([f'S{i:03d}' for i in range(8)], 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            output = root / 'result'
            argv = ['run.py', '--data', str(data), '--cache', str(root / 'cache'),
                    '--output', str(output), '--epochs', '1', '--max-steps', '1',
                    '--batch-size', '2', '--device', 'cpu', '--train-sampling', 'sqrt-balanced',
                    '--checkpoint-audit', '--rate-audit',
                    '--methods', 'blind_adapter', 'rate_adapter', 'full_align', 'shared_align']

            def build(_args, channels, classes, method, device):
                return AdaptedHART(LongToyBackbone(), channels, classes, rank=2,
                                   conditioned=method != 'blind_adapter').to(device)

            with patch('sys.argv', argv), patch.object(run, 'load_harth', return_value=(x, labels, groups)), \
                    patch.object(run, 'build_model', side_effect=build):
                run.main()
            self.assertEqual(len(pd.read_csv(output / 'metrics.csv')), 28)
            self.assertEqual(len(pd.read_csv(output / 'per_class.csv')), 28 * 12)
            self.assertEqual(len(pd.read_csv(output / 'per_subject.csv')), 28 * 2)
            self.assertEqual(len(pd.read_csv(output / 'contrasts.csv')), 42)
            self.assertEqual(len(pd.read_csv(output / 'training_summary.csv')), 4)
            self.assertEqual(len(pd.read_csv(output / 'final_metrics.csv')), 28)
            audit = pd.read_csv(output / 'rate_metadata_audit.csv')
            self.assertEqual(len(audit), 28)
            self.assertTrue((audit[audit.method == 'blind_adapter'].mean_probability_l1_change == 0).all())
            args = SimpleNamespace(data=data, cache=root / 'cache', data_seed=7, windows_per_subject=40)
            with patch.object(run, 'load_harth', side_effect=AssertionError('cache not used')):
                cached = run.load_windows(args)
            np.testing.assert_array_equal(cached[0], x)
            with patch('sys.argv', argv), self.assertRaises(SystemExit):
                run.main()

    def test_shared_alignment_training_and_checkpoint(self):
        from run import fit
        rng = np.random.default_rng(4)
        x = rng.normal(size=(6, 12, 2)).astype(np.float32)
        arrays = {key: x.copy() for key in ('clean30', 'aa15', 'aa10',
                                            'common_clean15', 'common_low15',
                                            'common_clean10', 'common_low10')}
        labels = np.array([0, 1, 2, 0, 1, 2])
        model = AdaptedHART(ToyBackbone(), 2, 3, rank=2)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(lr=.001, seed=7, epochs=1, batch_size=2,
                                   max_steps=1, align_weight=.1, patience=2,
                                   output=Path(directory))
            logs = []
            summary = fit(model, args, 'shared_align', arrays,
                {'train': np.array([0, 1, 2, 3]), 'val': np.array([4, 5])},
                labels, labels, np.arange(3), torch.device('cpu'), logs)
            self.assertEqual(logs[0]['steps'], 1)
            self.assertTrue((args.output / 'shared_align.pt').is_file())
            self.assertTrue(np.isfinite(logs[0]['validation_nll']))
            self.assertTrue(np.isfinite(logs[0]['alignment_to_ce_gradient_ratio_first_batch']))
            self.assertEqual(summary['stop_reason'], 'max_steps')
            self.assertTrue(summary['best_at_final_epoch'])
        model.close()


if __name__ == '__main__':
    unittest.main()
