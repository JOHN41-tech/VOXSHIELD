"""The log-mel CRNN network, isolated because it requires PyTorch.

This module imports ``torch`` at module scope and raises :class:`ImportError`
without it, exactly as :mod:`voxshield.training._cnn` does. Keeping the network
here rather than nested inside :meth:`LogMelCRNN._build_network` is what lets the
wrapper be a real ``nn.Module``-bearing object at class-creation time; a nested
definition whose base comes from a function-local import cannot be type-checked,
because the base class does not exist until the factory runs.

Nothing imports this module unless the ``logmel_crnn`` family is actually
requested, so the scikit-learn baselines still run in an environment with no
PyTorch installed.

Three ideas separate this network from :class:`~voxshield.training._cnn.LogMelNet`,
and each earns its place by removing a specific way the simpler model is wrong:

* :class:`SEBlock` reweights channels before pooling. Global average pooling
  treats every channel as equally informative, so a channel that fires on silence
  contributes as much to the score as one that fires on vocoder artefacts.
  Squeeze-excitation lets the network learn the weighting instead of assuming it.
* :class:`AttentionPool` averages over time with learned weights rather than a
  flat mean. A spoofing cue is often localised — a handful of frames carry the
  vocoder's fingerprint — and a flat mean over 300 frames dilutes it by the
  frames that carry nothing. Attention keeps the informative frames and the
  flat mean is the special case where every weight is equal.
* :class:`SpecAugment` masks frequency and time bands during training only. It is
  the standard regulariser for small audio corpora, and it is applied here with a
  hard rule that it never runs at evaluation time: augmentation that leaks into
  scoring changes what the reported metric measures.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on the environment
    _MISSING = (
        "the logmel_crnn family needs PyTorch; install torch, or choose "
        "mfcc_logreg / mfcc_xgboost, which need only scikit-learn and XGBoost"
    )
    raise ImportError(_MISSING) from exc

__all__ = [
    "AttentionPool",
    "LogMelCRNNNet",
    "SEBlock",
    "SpecAugment",
    "seed_everything",
]


def seed_everything(seed: int) -> None:
    """Seed torch's global and per-device RNGs.

    Args:
        seed: Non-negative integer seed.
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():  # pragma: no cover - depends on the machine
        torch.cuda.manual_seed_all(int(seed))


class SEBlock(nn.Module):
    """Squeeze-excitation channel attention.

    A channel descriptor is formed by global average pooling, squeezed through a
    bottleneck of ``channels // reduction``, and expanded back to one weight per
    channel in ``(0, 1)``. Multiplying the feature map by those weights lets the
    network decide which channels matter before the pooled representation is
    formed.

    Args:
        channels: Number of input (and output) channels.
        reduction: Bottleneck divisor. ``0`` disables the block, in which case
            :meth:`forward` is the identity. Disabling is expressed as a real
            parameter-free path rather than a flag read at call time, so a model
            built with ``se_reduction=0`` has no squeeze-excitation weights in its
            state dict at all and cannot be silently rebuilt with them.

    Raises:
        ValueError: If ``channels`` is not positive or ``reduction`` is negative.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        if channels < 1:
            msg = f"SEBlock needs a positive channel count, got {channels}"
            raise ValueError(msg)
        if reduction < 0:
            msg = f"SEBlock reduction must be non-negative, got {reduction}"
            raise ValueError(msg)
        self.channels = int(channels)
        self.reduction = int(reduction)
        self.enabled = reduction > 0
        if self.enabled:
            hidden = max(1, channels // reduction)
            self.squeeze = nn.Linear(channels, hidden)
            self.excite = nn.Linear(hidden, channels)
        else:
            self.squeeze = None
            self.excite = None

    def forward(self, x: Any) -> Any:
        """Reweight channels by their pooled activation.

        Args:
            x: Float tensor of shape ``(batch, channels, height, width)``.

        Returns:
            A tensor of the same shape. When disabled, ``x`` is returned
            unchanged, so a disabled block is a genuine no-op.
        """
        if not self.enabled:
            return x
        # (B, C, H, W) -> (B, C): one number summarising each channel.
        descriptor = x.mean(dim=(2, 3))
        weights = torch.sigmoid(self.excite(torch.relu(self.squeeze(descriptor))))
        return x * weights.unsqueeze(-1).unsqueeze(-1)


class AttentionPool(nn.Module):
    """Learned additive attention over the time axis.

    Each time step is scored by a small linear layer, the scores are normalised
    with a softmax over time, and the pooled output is the weighted sum. With all
    scores equal this reduces to the flat mean, so the block can only improve on
    average pooling in the representational sense; a badly conditioned attention
    layer degrades to the same mean rather than to something worse.

    Args:
        dim: Feature width per time step.

    Raises:
        ValueError: If ``dim`` is not positive.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            msg = f"AttentionPool needs a positive feature width, got {dim}"
            raise ValueError(msg)
        self.dim = int(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Any) -> Any:
        """Pool a sequence over time with learned weights.

        Args:
            x: Float tensor of shape ``(batch, time, dim)``.

        Returns:
            Float tensor of shape ``(batch, dim)``.
        """
        # (B, T, D) -> (B, T, 1) -> (B, T): one importance score per time step.
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class SpecAugment(nn.Module):
    """Frequency and time masking, applied only while training.

    Two axes are masked, because they remove different shortcuts. Frequency
    masking stops the network keying on one narrow band; time masking stops it
    requiring a cue to be present at a fixed offset. Both are drawn per sample and
    per batch, so no single masked pattern is memorised.

    The masks are applied when ``self.training`` is true and skipped otherwise.
    That conditional is the whole contract: an evaluation forward pass must see
    the unmasked spectrogram, or the reported score describes a randomly perturbed
    input rather than the audio.

    The layer sits after the channel dimension is added, so it operates on the
    ``(batch, channel, time, bands)`` tensor the convolutions consume. Masking the
    3-D input before the unsqueeze would work equally well and is harder to place:
    keeping the augmentation immediately before the conv stack is what makes it
    obvious from the module list that it touches the spectrum and nothing else.

    Args:
        freq_mask: Maximum width, in mel bands, of a frequency mask. ``0``
            disables frequency masking.
        time_mask: Maximum width, in frames, of a time mask. ``0`` disables time
            masking.
        num_masks: How many masks to apply along each enabled axis.
        fill: Value written into the masked region. Zero is the log-mel silence
            floor, which is the least informative value available.

    Raises:
        ValueError: If any count or width is negative.
    """

    def __init__(
        self,
        *,
        freq_mask: int = 8,
        time_mask: int = 16,
        num_masks: int = 2,
        fill: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("freq_mask", freq_mask),
            ("time_mask", time_mask),
            ("num_masks", num_masks),
        ):
            if value < 0:
                msg = f"SpecAugment {name} must be non-negative, got {value}"
                raise ValueError(msg)
        self.freq_mask = int(freq_mask)
        self.time_mask = int(time_mask)
        self.num_masks = int(num_masks)
        self.fill = float(fill)

    def forward(self, x: Any) -> Any:
        """Mask the input when training, pass it through when evaluating.

        Args:
            x: Float tensor of shape ``(batch, channels, time, bands)``.

        Returns:
            A tensor of the same shape. Unchanged when ``self.training`` is
            false, or when both mask widths are zero.
        """
        if not self.training:
            return x
        if self.freq_mask <= 0 and self.time_mask <= 0:
            return x
        masked = x.clone()
        _batch, _channels, n_frames, n_bands = masked.shape
        for _ in range(self.num_masks):
            if self.freq_mask > 0 and n_bands > 0:
                width = int(torch.randint(0, self.freq_mask + 1, (1,)).item())
                width = min(width, n_bands)
                if width > 0:
                    start = int(torch.randint(0, n_bands - width + 1, (1,)).item())
                    masked[:, :, :, start : start + width] = self.fill
            if self.time_mask > 0 and n_frames > 0:
                width = int(torch.randint(0, self.time_mask + 1, (1,)).item())
                width = min(width, n_frames)
                if width > 0:
                    start = int(torch.randint(0, n_frames - width + 1, (1,)).item())
                    masked[:, :, start : start + width, :] = self.fill
        return masked


class LogMelCRNNNet(nn.Module):
    """Convolutional stack, optional bidirectional GRU, attention pooling, head.

    The forward pass is deliberately shaped around the frequency axis being
    collapsed before the sequence model runs:

    ``(B, T, F)`` → ``(B, 1, T, F)`` → conv blocks with SE → adaptive average pool
    over frequency → ``(B, T//4, C)`` → optional bi-GRU → attention pool →
    dropout → linear → ``(B, 2)``.

    Collapsing frequency first is what makes a recurrent layer affordable. A GRU
    run over the flattened ``(time, frequency)`` grid would have a step size of
    thousands of inputs; running it over 80 channels per frame is what keeps the
    parameter count in the tens of thousands and the CPU cost below the plain CNN's.

    Args:
        in_bands: Mel bands per frame.
        widths: Output channels of the convolutional blocks. The first entry sees
            the single-channel input, so ``(16, 32)`` produces a 32-wide
            frequency-collapsed representation.
        dropout: Dropout probability before the head.
        se_reduction: Squeeze-excitation bottleneck divisor. ``0`` disables it.
        rnn_hidden: Hidden size per direction of the bidirectional GRU. ``0``
            disables the recurrent layer entirely, leaving a pure convolutional
            model with attention pooling.
        spec_augment: Whether to include a :class:`SpecAugment` layer. The layer
            is inert outside training regardless, but building it only when asked
            means a configuration that requested no augmentation produces a
            network whose module list reflects that.
        freq_mask: Frequency mask width for :class:`SpecAugment`.
        time_mask: Time mask width for :class:`SpecAugment`.
        num_masks: Masks per axis for :class:`SpecAugment`.

    Raises:
        ValueError: If the band count or any width is not positive.
    """

    def __init__(
        self,
        in_bands: int,
        widths: Sequence[int],
        dropout: float,
        *,
        se_reduction: int = 8,
        rnn_hidden: int = 32,
        spec_augment: bool = False,
        freq_mask: int = 8,
        time_mask: int = 16,
        num_masks: int = 2,
    ) -> None:
        super().__init__()
        if in_bands < 1:
            msg = f"in_bands must be positive, got {in_bands}"
            raise ValueError(msg)
        if se_reduction < 0:
            msg = f"se_reduction must be non-negative, got {se_reduction}"
            raise ValueError(msg)
        if rnn_hidden < 0:
            msg = f"rnn_hidden must be non-negative, got {rnn_hidden}"
            raise ValueError(msg)

        self.in_bands = int(in_bands)
        self.se_reduction = int(se_reduction)
        self.rnn_hidden = int(rnn_hidden)

        self.augment: nn.Module = (
            SpecAugment(
                freq_mask=freq_mask,
                time_mask=time_mask,
                num_masks=num_masks,
            )
            if spec_augment
            else nn.Identity()
        )

        layers: list[nn.Module] = []
        current = 1
        for width in widths:
            if width < 1:
                msg = f"every conv width must be positive, got {width}"
                raise ValueError(msg)
            layers += [
                nn.Conv2d(current, width, kernel_size=3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                SEBlock(width, self.se_reduction),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ]
            current = width
        if current < 1:  # pragma: no cover - guarded by the width check above
            msg = "no convolutional widths were supplied"
            raise ValueError(msg)
        self.features = nn.Sequential(*layers)
        self.channels = current

        # The recurrent layer is optional, and its absence is recorded as an
        # explicit attribute rather than inferred later from the parameter count.
        self.gru: nn.Module | None = (
            nn.GRU(current, self.rnn_hidden, batch_first=True, bidirectional=True)
            if self.rnn_hidden > 0
            else None
        )
        pooled_dim = 2 * self.rnn_hidden if self.gru is not None else current

        self.attention = AttentionPool(pooled_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(pooled_dim, 2)

    def forward(self, x: Any) -> Any:
        """Score a batch of log-mel matrices.

        Args:
            x: Float tensor of shape ``(batch, n_frames, n_bands)``.

        Returns:
            Logits of shape ``(batch, 2)``. Softmax is applied by the caller so
            that training can use cross entropy on logits, which is numerically
            better behaved than taking the log of a softmax.
        """
        # (batch, time, bands) -> (batch, 1, time, bands)
        x = x.unsqueeze(1)
        x = self.augment(x) if isinstance(self.augment, SpecAugment) else x
        x = self.features(x)
        # Collapse frequency: (B, C, T', F') -> (B, C, T') -> (B, T', C).
        x = torch.mean(x, dim=3).transpose(1, 2)
        if self.gru is not None:
            x, _ = self.gru(x)
        x = self.attention(x)
        x = self.dropout(x)
        return self.head(x)
