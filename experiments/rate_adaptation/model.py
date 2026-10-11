"""Small trainable adapters inside a frozen HART time/FFT encoder."""

import torch
from torch import nn
from torch.nn import functional as F


class EncodingAdapter(nn.Module):
    def __init__(self, width=48, rank=8, conditioned=True):
        super().__init__()
        self.conditioned = conditioned
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank)
        self.rate = nn.Linear(2, rank, bias=False)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens, rates):
        rate_features = torch.stack((torch.log2(rates / 30), rates / 30), dim=-1)
        context = self.rate(rate_features) if self.conditioned else self.rate(torch.zeros_like(rate_features))
        delta = self.up(F.gelu(self.down(self.norm(tokens)) + context[:, None, None, :]))
        return tokens + delta


def summarize(tokens):
    return torch.cat((tokens.mean(dim=2), tokens.std(dim=2, unbiased=False)), dim=-1).flatten(1)


class AdaptedHART(nn.Module):
    def __init__(self, backbone, channels, classes, rank=8, use_adapters=True, conditioned=True):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.use_adapters = use_adapters
        width = backbone.config.d_model
        self.adapters = nn.ModuleDict({
            name: EncodingAdapter(width, rank, conditioned) for name in ("time", "frequency")
        })
        if not use_adapters:
            self.adapters.requires_grad_(False)
        self.head = nn.Sequential(nn.LayerNorm(2 * channels * width), nn.Linear(2 * channels * width, classes))
        self._rates = None
        self._handles = [backbone.time_encoding.register_forward_hook(self._hook("time")),
                         backbone.fft_encoding.register_forward_hook(self._hook("frequency"))]

    def _hook(self, name):
        def apply(_module, _args, output):
            if self.use_adapters:
                return self.adapters[name](output, self._rates)
            return output
        return apply

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, values, rates):
        self._rates = rates.to(device=values.device, dtype=values.dtype)
        output = self.backbone(values, enable_masking=False, return_dict=True)
        embedding = summarize(output.last_hidden_state)
        return self.head(embedding), embedding

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def alignment_loss(left, right):
    return (1 - F.cosine_similarity(left, right, dim=-1, eps=1e-8)).clamp_min(0).mean()
