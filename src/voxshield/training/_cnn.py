"""The log-mel CNN network, isolated because it requires PyTorch.

This module imports ``torch`` at module scope and raises :class:`ImportError`
without it, exactly as :mod:`voxshield.data.torch_dataset` does. Keeping the
network here rather than nested inside :meth:`LogMelCNN._build_network` is what
lets the base class be a real ``nn.Module`` at class-creation time; a nested
definition whose base comes from a function-local import cannot be type-checked,
because the base class does not exist until the factory runs.

Nothing imports this module unless the ``logmel_cnn`` baseline is actually
requested, so the two scikit-learn baselines still run in an environment with no
PyTorch installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on the environment
    _MISSING = (
        "the logmel_cnn baseline needs PyTorch; install torch, or choose "
        "mfcc_logreg / mfcc_xgboost, which need only scikit-learn and XGBoost"
    )
    raise ImportError(_MISSING) from exc

__all__ = ["LogMelNet", "seed_everything"]


def seed_everything(seed: int) -> None:
    """Seed torch's global and per-device RNGs.

    Args:
        seed: Non-negative integer seed.
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():  # pragma: no cover - depends on the machine
        torch.cuda.manual_seed_all(int(seed))


class LogMelNet(nn.Module):
    """Two convolutional blocks, global average pooling, linear head.

    Global average pooling rather than flatten is the load-bearing choice. Pooling
    over time and frequency makes the score invariant to where a cue sits in the
    clip, so edge padding does not shift the answer, and it keeps the parameter
    count small enough to train a smoke run on CPU. A flattened first layer over
    300 frames is both larger and more sensitive to clip length than anything
    this baseline needs.

    Args:
        in_bands: Mel bands per frame.
        widths: Output channels of the convolutional blocks.
        dropout: Dropout probability before the head.
    """

    def __init__(self, in_bands: int, widths: Sequence[int], dropout: float) -> None:
        super().__init__()
        if in_bands < 1:
            msg = f"in_bands must be positive, got {in_bands}"
            raise ValueError(msg)
        layers: list[Any] = []
        current = 1
        for width in widths:
            if width < 1:
                msg = f"every conv width must be positive, got {width}"
                raise ValueError(msg)
            layers += [
                nn.Conv2d(current, width, kernel_size=3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
            ]
            current = width
        if current < 1:  # pragma: no cover - guarded by the width check above
            msg = "no convolutional widths were supplied"
            raise ValueError(msg)
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(current, 2)

    def forward(self, x: Any) -> Any:
        """Score a batch of log-mel matrices.

        Args:
            x: Float tensor of shape ``(batch, n_frames, n_bands)``.

        Returns:
            Logits of shape ``(batch, 2)``. Softmax is applied by the caller so
            that training can use cross entropy on logits, which is numerically
            better behaved than taking log of a softmax.
        """
        # (batch, time, bands) -> (batch, 1, bands, time)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)
        return self.head(x)
