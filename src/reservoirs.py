"""Fixed spiking reservoirs.

Two options are provided (both *fixed* / untrained by default, which is the
whole point of reservoir computing -- only the readout learns):

A. :class:`SpikingLIFReservoir` -- a random recurrent LIF reservoir in the
   Liquid-State-Machine style. Sparse fixed recurrent weights, configurable
   size, spectral radius, input scaling, leak and excitatory/inhibitory ratio
   (Dale's law). Consumes input *spikes* and emits reservoir *spikes*.

B. :class:`LegendreDelayReservoir` -- an LDN / LMU-style fixed linear temporal
   memory (Legendre polynomials over a sliding window). Consumes the raw signal,
   produces a rich temporal state, then **encodes that state into spikes** before
   the readout. Optional; kept lighter-weight than the LIF reservoir.

Both expose ``.run(x) -> {'spikes', 'traces', ...}`` so the model can treat them
uniformly. Weights are stored as non-trainable tensors (buffers) and may be
quantized via :class:`~src.utils.QuantConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .encoders import build_encoder
from .neurons import LIFParams, lif_step
from .utils import QuantConfig, maybe_quantize


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class ReservoirConfig:
    kind: str = "lif"                  # 'lif' or 'ldn'
    n_reservoir: int = 300
    connectivity: float = 0.1          # recurrent connection probability
    spectral_radius: float = 0.9       # target |lambda|_max of recurrent matrix
    input_scaling: float = 1.0
    input_connectivity: float = 1.0    # fraction of input->reservoir links
    exc_ratio: float = 0.8             # fraction of excitatory neurons (Dale)
    leak: float = 0.9
    v_threshold: float = 1.0
    refractory: int = 0
    trace_decay: float = 0.9           # low-pass for reservoir traces
    # LDN-specific
    ldn_order: int = 8
    ldn_theta: float = 20.0
    ldn_encoder: Optional[dict] = None  # encoder applied to LDN states

    @classmethod
    def from_dict(cls, d) -> "ReservoirConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# A. Random recurrent LIF reservoir (LSM)
# --------------------------------------------------------------------------- #
class SpikingLIFReservoir:
    """Fixed random recurrent LIF reservoir (Liquid State Machine style)."""

    def __init__(self, in_channels: int, cfg: ReservoirConfig, device,
                 seed: int = 0, qcfg: Optional[QuantConfig] = None):
        self.cfg = cfg
        self.device = device
        self.in_channels = in_channels
        self.N = cfg.n_reservoir
        self.qcfg = qcfg
        self.lif = LIFParams(leak=cfg.leak, v_threshold=cfg.v_threshold,
                             refractory=cfg.refractory)
        g = torch.Generator(device="cpu").manual_seed(seed)

        # --- input weights [C_in, N] ---
        w_in = (torch.rand(in_channels, self.N, generator=g) * 2 - 1) * cfg.input_scaling
        in_mask = (torch.rand(in_channels, self.N, generator=g) < cfg.input_connectivity)
        w_in = w_in * in_mask
        self.W_in = w_in.to(device)

        # --- recurrent weights [N, N], sparse, Dale's law, spectral-radius scaled ---
        mask = (torch.rand(self.N, self.N, generator=g) < cfg.connectivity).float()
        mask.fill_diagonal_(0.0)  # no self-connections
        magnitude = torch.rand(self.N, self.N, generator=g).abs()
        # excitatory / inhibitory assignment (Dale): sign fixed per presynaptic neuron
        n_exc = int(round(cfg.exc_ratio * self.N))
        sign = torch.ones(self.N)
        sign[n_exc:] = -1.0
        # inhibitory neurons usually stronger to balance E/I
        gain = torch.ones(self.N)
        if self.N - n_exc > 0:
            gain[n_exc:] = cfg.exc_ratio / max(1e-6, (1 - cfg.exc_ratio))
        w_rec = magnitude * mask * (sign * gain)[:, None]  # row = presynaptic
        w_rec = self._scale_spectral_radius(w_rec, cfg.spectral_radius)
        self.W_rec = w_rec.to(device)
        self.exc_mask = (sign > 0).to(device)

    @staticmethod
    def _scale_spectral_radius(w: torch.Tensor, target: float) -> torch.Tensor:
        if target is None or target <= 0:
            return w
        try:
            eig = torch.linalg.eigvals(w)
            rho = float(eig.abs().max().item())
        except Exception:
            rho = float(torch.linalg.norm(w, 2).item())
        if rho < 1e-8:
            return w
        return w * (target / rho)

    def spectral_radius(self) -> float:
        eig = torch.linalg.eigvals(self.W_rec.cpu())
        return float(eig.abs().max().item())

    def out_channels(self) -> int:
        return self.N

    def _qw(self):
        w_in = maybe_quantize(self.W_in, self.qcfg, "weights")
        w_rec = maybe_quantize(self.W_rec, self.qcfg, "weights")
        return w_in, w_rec

    @torch.no_grad()
    def run(self, input_spikes: torch.Tensor, return_traces: bool = True) -> Dict[str, torch.Tensor]:
        """``input_spikes``: [B, T, C_in] -> dict(spikes=[B,T,N], traces=[B,T,N])."""
        B, T, C = input_spikes.shape
        assert C == self.in_channels, f"expected {self.in_channels} channels, got {C}"
        dev = self.device
        x = input_spikes.to(dev)
        w_in, w_rec = self._qw()

        v = torch.zeros(B, self.N, device=dev)
        ref = (torch.zeros(B, self.N, device=dev, dtype=torch.long)
               if self.cfg.refractory > 0 else None)
        s_prev = torch.zeros(B, self.N, device=dev)
        trace = torch.zeros(B, self.N, device=dev)

        spikes = torch.empty(B, T, self.N, device=dev)
        traces = torch.empty(B, T, self.N, device=dev) if return_traces else None

        for t in range(T):
            current = x[:, t] @ w_in + s_prev @ w_rec
            v = maybe_quantize(v, self.qcfg, "states")
            v, s, ref = lif_step(v, current, self.lif, ref, surrogate=False)
            trace = self.cfg.trace_decay * trace + s
            trace = maybe_quantize(trace, self.qcfg, "traces")
            spikes[:, t] = s
            if return_traces:
                traces[:, t] = trace
            s_prev = s

        return {"spikes": spikes, "traces": traces}


# --------------------------------------------------------------------------- #
# B. Legendre Delay Network reservoir (optional, fixed)
# --------------------------------------------------------------------------- #
def _ldn_matrices(order: int, theta: float):
    """Continuous LDN/LMU A, B matrices realizing a delay of ``theta``."""
    Q = np.arange(order, dtype=np.float64)
    R = (2 * Q + 1)[:, None]
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1.0, (-1.0) ** (i - j + 1)) * R
    B = (-1.0) ** Q[:, None] * R
    return A, B[:, 0]


class LegendreDelayReservoir:
    """Fixed LDN temporal-memory reservoir; states are spike-encoded for readout."""

    def __init__(self, in_channels: int, cfg: ReservoirConfig, device,
                 seed: int = 0, qcfg: Optional[QuantConfig] = None):
        from scipy.linalg import expm

        self.cfg = cfg
        self.device = device
        self.in_channels = in_channels
        self.order = cfg.ldn_order
        self.qcfg = qcfg

        A, B = _ldn_matrices(cfg.ldn_order, cfg.ldn_theta)
        dt = 1.0
        Ad = expm(A * (dt / cfg.ldn_theta))
        # Bd = A^{-1} (Ad - I) B   (zero-order hold)
        Bd = np.linalg.solve(A, (Ad - np.eye(cfg.ldn_order)) @ B)
        self.Ad = torch.tensor(Ad, dtype=torch.float32, device=device)
        self.Bd = torch.tensor(Bd, dtype=torch.float32, device=device)

        enc_cfg = cfg.ldn_encoder or {"name": "posneg", "threshold": 0.15}
        self.encoder = build_encoder(dict(enc_cfg))
        self._state_dim = in_channels * cfg.ldn_order

    def out_channels(self) -> int:
        return self.encoder.out_channels(self._state_dim)

    @torch.no_grad()
    def run(self, signal: torch.Tensor, return_traces: bool = True) -> Dict[str, torch.Tensor]:
        """``signal``: raw [B, T, C] -> encoded spikes [B, T, out_channels]."""
        B, T, C = signal.shape
        dev = self.device
        u = signal.to(dev)
        M = torch.zeros(B, C, self.order, device=dev)
        states = torch.empty(B, T, C * self.order, device=dev)
        for t in range(T):
            # M_new[b,c,i] = sum_j Ad[i,j] M[b,c,j] + Bd[i] u[b,c,t]
            M = torch.einsum("ij,bcj->bci", self.Ad, M) + u[:, t][:, :, None] * self.Bd
            states[:, t] = M.reshape(B, C * self.order)

        # normalise states per (sample, feature) then spike-encode
        mu = states.mean(dim=1, keepdim=True)
        sd = states.std(dim=1, keepdim=True) + 1e-6
        states_n = (states - mu) / sd
        spikes = self.encoder.encode(states_n)

        traces = None
        if return_traces:
            traces = torch.empty_like(spikes)
            tr = torch.zeros(B, spikes.shape[-1], device=dev)
            for t in range(T):
                tr = self.cfg.trace_decay * tr + spikes[:, t]
                traces[:, t] = tr
        return {"spikes": spikes, "traces": traces, "ldn_states": states}


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def build_reservoir(in_channels: int, cfg: ReservoirConfig, device,
                    seed: int = 0, qcfg: Optional[QuantConfig] = None):
    if cfg.kind == "lif":
        return SpikingLIFReservoir(in_channels, cfg, device, seed, qcfg)
    if cfg.kind == "ldn":
        return LegendreDelayReservoir(in_channels, cfg, device, seed, qcfg)
    raise ValueError(f"unknown reservoir kind '{cfg.kind}' (use 'lif' or 'ldn')")
