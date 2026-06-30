"""Leaky integrate-and-fire (LIF) neuron primitives.

Two spike functions are provided:

* ``heaviside_spike`` -- a hard, non-differentiable threshold used by the fixed
  reservoir and by the **R-STDP** (local, gradient-free) pathway. This is the
  behaviour a Loihi-2 / Lava LIF process exposes.
* ``SurrogateSpike`` -- an autograd ``Function`` with a smooth surrogate
  gradient, used *only* by the off-chip surrogate-gradient (BPTT) baseline.

State and dynamics are deliberately simple and explicit so the mapping to a
Lava ``LIF`` process is one-to-one:

    v[t] = leak * v[t-1] + I[t]            (membrane integration)
    s[t] = 1 if v[t] >= v_th else 0        (threshold)
    v[t] = v[t] - v_th * s[t]              (subtractive reset, default)

Refractoriness is optional and modelled by clamping input during a countdown.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class LIFParams:
    """Parameters of a LIF population (shared across neurons in the layer)."""

    leak: float = 0.9            # membrane retention factor in [0,1] (1 = no leak)
    v_threshold: float = 1.0
    v_reset: float = 0.0
    reset: str = "subtract"      # 'subtract' or 'zero'
    refractory: int = 0          # refractory period in timesteps
    surrogate_scale: float = 10.0  # steepness of surrogate gradient (baseline only)

    @classmethod
    def from_dict(cls, d) -> "LIFParams":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Spike functions
# --------------------------------------------------------------------------- #
def heaviside_spike(v: torch.Tensor, v_th: float) -> torch.Tensor:
    """Hard threshold spike (no gradient). Used by reservoir + R-STDP."""
    return (v >= v_th).to(v.dtype)


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate gradient backward.

    grad = scale / (1 + scale * |v - v_th|)^2   (SuperSpike-style)
    """

    @staticmethod
    def forward(ctx, v, v_th, scale):
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        ctx.scale = scale
        return (v >= v_th).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        sg = ctx.scale / (1.0 + ctx.scale * (v - ctx.v_th).abs()) ** 2
        return grad_output * sg, None, None


def surrogate_spike(v: torch.Tensor, v_th: float, scale: float) -> torch.Tensor:
    return SurrogateSpike.apply(v, v_th, scale)


# --------------------------------------------------------------------------- #
# Functional single-step LIF
# --------------------------------------------------------------------------- #
def lif_step(
    v: torch.Tensor,
    input_current: torch.Tensor,
    params: LIFParams,
    refractory_count: Optional[torch.Tensor] = None,
    surrogate: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Advance a LIF population by one timestep.

    Parameters
    ----------
    v : membrane potential ``[..., N]``.
    input_current : synaptic input this step ``[..., N]``.
    params : :class:`LIFParams`.
    refractory_count : optional integer countdown tensor ``[..., N]``.
    surrogate : if True use the surrogate-gradient spike (for BPTT baseline);
        otherwise a hard threshold (reservoir / R-STDP).

    Returns
    -------
    (v_new, spikes, refractory_count)
    """
    if params.refractory > 0 and refractory_count is not None:
        active = (refractory_count <= 0).to(v.dtype)
        input_current = input_current * active

    v = params.leak * v + input_current

    if surrogate:
        spikes = surrogate_spike(v, params.v_threshold, params.surrogate_scale)
    else:
        spikes = heaviside_spike(v, params.v_threshold)

    if params.reset == "zero":
        v = v * (1.0 - spikes) + params.v_reset * spikes
    else:  # subtractive reset
        v = v - params.v_threshold * spikes

    if params.refractory > 0 and refractory_count is not None:
        refractory_count = torch.clamp(refractory_count - 1, min=0)
        refractory_count = refractory_count + (spikes * params.refractory).long()

    return v, spikes, refractory_count


class LIFLayer(torch.nn.Module):
    """A stateless-config LIF layer wrapping :func:`lif_step` over time.

    Weights are *not* owned here -- callers pass in the already-projected input
    current. This keeps synapses (plastic or fixed) and neuron dynamics cleanly
    separated, matching the Lava Process/Connection split.
    """

    def __init__(self, n_neurons: int, params: LIFParams, surrogate: bool = False):
        super().__init__()
        self.n_neurons = n_neurons
        self.params = params
        self.surrogate = surrogate

    def init_state(self, batch: int, device, dtype=torch.float32):
        v = torch.zeros(batch, self.n_neurons, device=device, dtype=dtype)
        ref = (
            torch.zeros(batch, self.n_neurons, device=device, dtype=torch.long)
            if self.params.refractory > 0
            else None
        )
        return v, ref

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        """``currents``: [B, T, N] pre-projected input -> spikes [B, T, N]."""
        B, T, N = currents.shape
        v, ref = self.init_state(B, currents.device, currents.dtype)
        out = torch.empty(B, T, N, device=currents.device, dtype=currents.dtype)
        for t in range(T):
            v, s, ref = lif_step(v, currents[:, t], self.params, ref, self.surrogate)
            out[:, t] = s
        return out
