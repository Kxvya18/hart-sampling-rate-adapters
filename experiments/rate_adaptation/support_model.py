"""Nyquist-support projection and bounded physical-context adapters for HART."""

import torch
from torch import nn
from torch.nn import functional as F

from model import AdaptedHART


def fourier_patch_support(config, rates, device, dtype):
    n = config.context_length
    frequencies = torch.arange(n // 2 + 1, device=device, dtype=dtype) * (30. / n)
    frequencies = frequencies[1:] if config.fft_remove_component == "dc" else frequencies[:-1]
    frequencies = torch.cat((frequencies, frequencies))
    length, stride = config.patch_length, config.patch_stride
    patches = (len(frequencies) - length) // stride + 1
    used = length + stride * (patches - 1)
    frequencies = frequencies[-used:].unfold(0, length, stride)
    return (frequencies[None] <= rates[:, None, None] / 2).to(dtype)[:, None]


class SupportAdapter(nn.Module):
    def __init__(self, width, rank, contextual=True):
        super().__init__()
        self.contextual = contextual
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank)
        self.rate = nn.Linear(2, rank, bias=False)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens, support):
        context = torch.cat((support, 1 - support), dim=-1)
        if not self.contextual:
            context = torch.zeros_like(context)
        return tokens + self.up(F.gelu(self.down(self.norm(tokens)) + self.rate(context)))


class SupportHART(AdaptedHART):
    """Project normalized FFT patch inputs, not the original physical waveform."""

    def __init__(self, backbone, channels, classes, rank=8, project=True, contextual=True, residual_support=False):
        config = backbone.config
        if not getattr(config, "fuse_fft", False) or config.context_length % 2:
            raise ValueError("SupportHART requires an even-context HART FFT backbone")
        super().__init__(backbone, channels, classes, rank)
        self.close()
        self.project = project
        self.residual_support = residual_support
        self.adapters = nn.ModuleDict({name: SupportAdapter(config.d_model, rank, contextual)
                                       for name in ("time", "frequency")})
        self._frequency_support = None
        self._handles = [backbone.fft_encoding.register_forward_pre_hook(self._project_fft),
                         backbone.time_encoding.register_forward_hook(self._support_hook("time")),
                         backbone.fft_encoding.register_forward_hook(self._support_hook("frequency"))]

    def _project_fft(self, _module, args):
        inputs = args[0]
        mask = fourier_patch_support(self.backbone.config, self._rates, inputs.device, inputs.dtype)
        if mask.shape[2:] != inputs.shape[2:]:
            raise RuntimeError("FFT patch layout does not match the physical support map")
        self._frequency_support = mask.mean(dim=-1, keepdim=True)
        return (inputs * mask, *args[1:]) if self.project else args

    def _support_hook(self, name):
        def apply(_module, _args, output):
            support = ((self._rates / 30)[:, None, None, None]
                       if name == "time" else self._frequency_support)
            adapted = self.adapters[name](output, support)
            if self.residual_support:
                # Preserve pretrained tokens; restrict only the learned correction.
                adapted = output + support * (adapted - output)
            # Empty patches carry no observed Fourier coordinates. This removes
            # their encoder bias as well, but does not mask mixed-support patches.
            if name == "frequency" and self.project:
                adapted = adapted * (support > 0).to(adapted.dtype)
            return adapted
        return apply

    def forward(self, values, rates):
        if not torch.all((rates > 0) & (rates <= 30)):
            raise ValueError("Sampling rates must lie in (0,30]")
        return super().forward(values, rates)
