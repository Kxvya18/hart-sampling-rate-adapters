import unittest
from types import SimpleNamespace

import torch
from torch import nn

from support_model import SupportAdapter, SupportHART, fourier_patch_support


class SpectrumBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=4, context_length=12, patch_length=3,
                                      patch_stride=3, fft_remove_component="last", fuse_fft=True)
        self.time_encoding = nn.Linear(3, 4)
        self.fft_encoding = nn.Linear(3, 4)
        self.past_fft = None

    def forward(self, x, **kwargs):
        time = x.reshape(len(x), 4, 3, 2).permute(0, 3, 1, 2)
        spectrum = torch.fft.rfft(x, dim=1)[:, :-1]
        frequency = torch.cat((spectrum.real, spectrum.imag), dim=1)
        frequency = frequency.reshape(len(x), 4, 3, 2).permute(0, 3, 1, 2)
        output = self.fft_encoding(frequency)
        self.past_fft = output
        return SimpleNamespace(last_hidden_state=torch.cat((self.time_encoding(time), output), dim=2))


class TestSupportArchitecture(unittest.TestCase):
    def test_residual_support_preserves_pretrained_tokens(self):
        backbone = SpectrumBackbone()
        model = SupportHART(backbone, 2, 3, rank=2, project=False, residual_support=True)
        values = torch.randn(2, 12, 2)
        model(values, torch.tensor([6., 30.]))
        original = backbone.past_fft.detach().clone()
        nn.init.normal_(model.adapters['frequency'].up.weight)
        model(values, torch.tensor([6., 30.]))
        self.assertTrue(torch.equal(original[0, :, [1, 3]], backbone.past_fft[0, :, [1, 3]]))
        self.assertFalse(torch.equal(original[1], backbone.past_fft[1]))
        model.close()

    def test_layout_full_support_and_monotonicity(self):
        config = SimpleNamespace(context_length=120, patch_length=12, patch_stride=12,
                                 fft_remove_component="last")
        mask = fourier_patch_support(config, torch.tensor([6., 10., 30.]), 'cpu', torch.float32)
        self.assertEqual(mask.shape, (3, 1, 10, 12))
        self.assertTrue(torch.all(mask[2] == 1))
        self.assertTrue(torch.all(mask[0] <= mask[1]))
        self.assertTrue(torch.all(mask[1] <= mask[2]))
        self.assertTrue(torch.equal(mask[:, :, :5], mask[:, :, 5:]))
        config.fft_remove_component = 'dc'
        dc = fourier_patch_support(config, torch.tensor([6.]), 'cpu', torch.float32)
        self.assertEqual(dc.sum().item(), 24)

    def test_projected_patch_has_zero_gradient_outside_support(self):
        config = SimpleNamespace(context_length=12, patch_length=3, patch_stride=3,
                                 fft_remove_component="last")
        mask = fourier_patch_support(config, torch.tensor([6.]), 'cpu', torch.float32)
        values = torch.randn(1, 2, 4, 3, requires_grad=True)
        adapter = SupportAdapter(4, 2)
        nn.init.normal_(adapter.up.weight)
        encoder = nn.Linear(3, 4)
        adapter(encoder(values * mask), mask.mean(-1, keepdim=True)).square().sum().backward()
        self.assertTrue(torch.all(values.grad[mask.expand_as(values) == 0] == 0))

    def test_empty_tokens_zero_and_backbone_frozen(self):
        backbone = SpectrumBackbone()
        model = SupportHART(backbone, 2, 3, rank=2).train()
        logits, _ = model(torch.randn(2, 12, 2), torch.tensor([6., 30.]))
        logits.square().sum().backward()
        self.assertTrue(torch.all(backbone.past_fft[0, :, [1, 3]] == 0))
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertGreater(model.adapters['time'].up.weight.grad.abs().sum().item(), 0)
        self.assertFalse(backbone.training)
        model.close()
        self.assertFalse(backbone.fft_encoding._forward_pre_hooks)
        self.assertFalse(backbone.fft_encoding._forward_hooks)

    def test_ablation_capacity_and_identity_initialization(self):
        modules = [SupportAdapter(4, 2, contextual=c) for c in (False, True)]
        self.assertEqual(sum(p.numel() for p in modules[0].parameters()),
                         sum(p.numel() for p in modules[1].parameters()))
        tokens = torch.randn(2, 2, 4, 4)
        self.assertTrue(torch.equal(tokens, modules[1](tokens, torch.ones(2, 1, 4, 1))))


if __name__ == '__main__':
    unittest.main()
