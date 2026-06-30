"""Spike encoders: continuous time-series ``[B, T, C]`` -> binary spikes ``[B, T, C']``.

Implemented encoders
--------------------
* ``rate``        -- Poisson / rate encoding (stochastic).
* ``temporal_diff`` -- rate on the rectified temporal derivative magnitude.
* ``threshold``   -- level-crossing (send-on-delta), single channel/input.
* ``posneg``      -- positive/negative **event** encoding: separate ON (rising)
                     and OFF (falling) spike channels per input channel. This is
                     the event-driven, Loihi-friendly encoder emphasised in the
                     project and yields ``2*C`` output channels.

All encoders are deterministic given the global seed (the stochastic ``rate``
encoder honours an optional ``torch.Generator``).
"""
from __future__ import annotations

from typing import Optional

import torch


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _minmax01(x: torch.Tensor) -> torch.Tensor:
    """Per (sample, channel) min-max scale over the time axis into [0, 1]."""
    mn = x.amin(dim=1, keepdim=True)
    mx = x.amax(dim=1, keepdim=True)
    return (x - mn) / (mx - mn + 1e-8)


class Encoder:
    """Base class. Subclasses implement :meth:`encode` and :meth:`out_channels`."""

    stochastic: bool = False

    def out_channels(self, in_channels: int) -> int:
        return in_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# --------------------------------------------------------------------------- #
# rate / Poisson
# --------------------------------------------------------------------------- #
class RateEncoder(Encoder):
    stochastic = True

    def __init__(self, gain: float = 1.0, normalize: str = "minmax",
                 generator: Optional[torch.Generator] = None):
        self.gain = gain
        self.normalize = normalize
        self.generator = generator

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "minmax":
            p = _minmax01(x)
        elif self.normalize == "sigmoid":
            p = torch.sigmoid(x)
        else:
            p = x
        p = (self.gain * p).clamp(0.0, 1.0)
        noise = torch.rand(p.shape, device=p.device, dtype=p.dtype, generator=self.generator)
        return (noise < p).to(x.dtype)


# --------------------------------------------------------------------------- #
# temporal-difference (rate on |dx/dt|)
# --------------------------------------------------------------------------- #
class TemporalDifferenceEncoder(Encoder):
    stochastic = True

    def __init__(self, gain: float = 3.0, generator: Optional[torch.Generator] = None):
        self.gain = gain
        self.generator = generator

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        dx = torch.zeros_like(x)
        dx[:, 1:] = x[:, 1:] - x[:, :-1]
        p = (self.gain * dx.abs()).clamp(0.0, 1.0)
        noise = torch.rand(p.shape, device=p.device, dtype=p.dtype, generator=self.generator)
        return (noise < p).to(x.dtype)


# --------------------------------------------------------------------------- #
# level-crossing (send-on-delta), single channel per input
# --------------------------------------------------------------------------- #
class ThresholdCrossingEncoder(Encoder):
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        ref = x[:, 0].clone()
        out = torch.zeros_like(x)
        for t in range(1, T):
            diff = x[:, t] - ref
            crossed = diff.abs() >= self.threshold
            out[:, t] = crossed.to(x.dtype)
            # move reference one delta toward the signal where a crossing occurred
            ref = ref + torch.sign(diff) * self.threshold * crossed.to(x.dtype)
        return out


# --------------------------------------------------------------------------- #
# positive/negative event encoding (ON / OFF channels)  ->  2*C channels
# --------------------------------------------------------------------------- #
class PositiveNegativeEventEncoder(Encoder):
    """Event encoder with separate channels for upward (ON) and downward (OFF)
    changes. Output channel layout: ``[ON_0..ON_{C-1}, OFF_0..OFF_{C-1}]``."""

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def out_channels(self, in_channels: int) -> int:
        return 2 * in_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        ref = x[:, 0].clone()
        on = torch.zeros_like(x)
        off = torch.zeros_like(x)
        for t in range(1, T):
            diff = x[:, t] - ref
            up = diff >= self.threshold
            down = diff <= -self.threshold
            on[:, t] = up.to(x.dtype)
            off[:, t] = down.to(x.dtype)
            step = (up.to(x.dtype) - down.to(x.dtype)) * self.threshold
            ref = ref + step
        return torch.cat([on, off], dim=-1)


# --------------------------------------------------------------------------- #
# identity / analog current injection (NOT spiking) -- standard LSM input mode
# --------------------------------------------------------------------------- #
class IdentityEncoder(Encoder):
    """Pass the (scaled) analog signal through as graded input *current*.

    The reservoir LIF neurons still produce spikes; only the *input* is analog.
    This is the classical Liquid-State-Machine current-injection input and is
    much better than event encoding for amplitude-coded tasks such as NARMA10.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
_ENCODERS = {
    "rate": RateEncoder,
    "poisson": RateEncoder,
    "temporal_diff": TemporalDifferenceEncoder,
    "threshold": ThresholdCrossingEncoder,
    "threshold_crossing": ThresholdCrossingEncoder,
    "posneg": PositiveNegativeEventEncoder,
    "posneg_event": PositiveNegativeEventEncoder,
    "identity": IdentityEncoder,
    "analog": IdentityEncoder,
    "current": IdentityEncoder,
}


def build_encoder(cfg, generator: Optional[torch.Generator] = None) -> Encoder:
    """Build an encoder from a config dict ``{'name': ..., **params}``.

    Unknown keys are ignored (so a shared ``threshold`` can sit in a config while
    switching to an encoder that does not use it, e.g. in ablations).
    """
    import inspect

    cfg = dict(cfg or {})
    name = cfg.pop("name", "posneg")
    if name not in _ENCODERS:
        raise ValueError(f"unknown encoder '{name}'. options: {sorted(_ENCODERS)}")
    cls = _ENCODERS[name]
    if cls in (RateEncoder, TemporalDifferenceEncoder):
        cfg.setdefault("generator", generator)
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in cfg.items() if k in accepted})
