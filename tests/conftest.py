"""Keep local modules importable from pytest's console entry point."""

import sys
from pathlib import Path

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TinySequentialCore(nn.Module):
    """Trainable smoke-only core with the real sequential core's call contract."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(4, 3, kernel_size=1)
        self.state = None

    def forward(self, frames, t0):
        if t0 == 0:
            self.state = None
        outputs = []
        for index in range(frames.shape[1]):
            logits = self.projection(frames[:, index])
            if self.state is not None:
                logits = logits + self.state * 0.05
            self.state = logits
            outputs.append(logits)
        return torch.stack(outputs, dim=2)

    def detach_states(self):
        if self.state is not None:
            self.state = self.state.detach()
