"""Model assembly: encoder -> fixed spiking reservoir -> trainable readout.

The model exposes the pieces each training method needs while keeping spike
communication explicit (no hidden global ops), which is what keeps it
Loihi-friendly:

* classification  -> R-STDP readout (``src.rstdp.RSTDPLayer``), optional fixed
  hidden spiking projection.
* regression      -> reservoir *traces* read out by either a closed-form ridge
  map (classical reservoir computing) or a *local* error-modulated delta rule
  (three-factor, no BPTT).
* linear baseline -> reservoir rate features + (logistic / ridge) regression.

Memory is handled by streaming the dataset through the reservoir in batches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .encoders import build_encoder
from .neurons import LIFParams, lif_step
from .reservoirs import ReservoirConfig, build_reservoir
from .rstdp import RSTDPConfig, RSTDPLayer
from .utils import QuantConfig, maybe_quantize


# --------------------------------------------------------------------------- #
# Fixed (untrained) spiking projection -- used for the optional hidden layer
# --------------------------------------------------------------------------- #
class FixedSpikingProjection:
    """A fixed random feed-forward LIF layer: spikes [B,T,in] -> spikes [B,T,out]."""

    def __init__(self, n_in, n_out, device, seed=0, scale=1.0, connectivity=0.5,
                 leak=0.9, v_threshold=1.0, qcfg: Optional[QuantConfig] = None):
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = (torch.rand(n_in, n_out, generator=g) * 2 - 1) * scale
        mask = (torch.rand(n_in, n_out, generator=g) < connectivity).float()
        self.W = (W * mask).to(device)
        self.lif = LIFParams(leak=leak, v_threshold=v_threshold)
        self.device = device
        self.n_in, self.n_out = n_in, n_out
        self.qcfg = qcfg

    @torch.no_grad()
    def run(self, spikes: torch.Tensor) -> torch.Tensor:
        B, T, _ = spikes.shape
        x = spikes.to(self.device)
        W = maybe_quantize(self.W, self.qcfg, "weights")
        v = torch.zeros(B, self.n_out, device=self.device)
        out = torch.empty(B, T, self.n_out, device=self.device)
        for t in range(T):
            v, s, _ = lif_step(v, x[:, t] @ W, self.lif, surrogate=False)
            out[:, t] = s
        return out


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class SpikingReservoirModel:
    def __init__(self, bundle_meta: Dict, cfg: Dict, device, seed: int = 0,
                 qcfg: Optional[QuantConfig] = None):
        self.device = device
        self.seed = seed
        self.qcfg = qcfg
        self.task_type = bundle_meta["task_type"]
        self.in_channels = bundle_meta["in_channels"]
        self.out_dim = bundle_meta["out_dim"]

        self.res_cfg = ReservoirConfig.from_dict(cfg.get("reservoir", {}))
        self.encoder_cfg = cfg.get("encoder", {"name": "posneg", "threshold": 0.2})
        self.uses_raw_input = self.res_cfg.kind == "ldn"

        # encoder (LDN consumes the raw signal directly)
        self.encoder = None if self.uses_raw_input else build_encoder(dict(self.encoder_cfg))
        enc_channels = (self.in_channels if self.uses_raw_input
                        else self.encoder.out_channels(self.in_channels))

        self.reservoir = build_reservoir(enc_channels, self.res_cfg, device, seed, qcfg)
        self.reservoir_dim = self.reservoir.out_channels()

        # optional fixed hidden spiking projection
        hcfg = cfg.get("hidden", {}) or {}
        self.hidden_size = int(hcfg.get("size", 0))
        self.hidden = None
        readout_in = self.reservoir_dim
        if self.hidden_size > 0:
            self.hidden = FixedSpikingProjection(
                self.reservoir_dim, self.hidden_size, device, seed=seed + 7,
                scale=hcfg.get("scale", 1.0), connectivity=hcfg.get("connectivity", 0.5),
                leak=hcfg.get("leak", 0.9), v_threshold=hcfg.get("v_threshold", 1.0),
                qcfg=qcfg)
            readout_in = self.hidden_size
        self.readout_in = readout_in

        # trainable readout (classification only; regression uses linear/local readouts)
        self.rstdp_cfg = RSTDPConfig.from_dict(cfg.get("readout", {}))
        self.readout = None
        if self.task_type == "classification":
            self.readout = RSTDPLayer(readout_in, self.out_dim, self.rstdp_cfg,
                                      device, seed=seed + 13, qcfg=qcfg)

        self.feature = cfg.get("feature", "rate")  # for linear baseline
        # multi-timescale trace basis exposed to the regression readout (richer
        # fading memory than a single low-pass; standard reservoir-computing trick)
        self.reg_trace_decays = list(cfg.get("readout", {}).get(
            "trace_decays", [0.4, 0.8, 0.95]))

    # ------------------------------------------------------------------ #
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if self.uses_raw_input:
            return X.to(self.device)
        return self.encoder.encode(X.to(self.device))

    @torch.no_grad()
    def reservoir_forward(self, X_batch: torch.Tensor, return_traces=True) -> Dict[str, torch.Tensor]:
        """One batch -> dict(input_spikes, res_spikes, res_traces, readout_pre)."""
        enc = self.encode(X_batch)
        out = self.reservoir.run(enc, return_traces=return_traces)
        res_spikes = out["spikes"]
        readout_pre = res_spikes
        hidden_spikes = None
        if self.hidden is not None:
            hidden_spikes = self.hidden.run(res_spikes)
            readout_pre = hidden_spikes
        return {"input_spikes": enc, "res_spikes": res_spikes,
                "res_traces": out["traces"], "hidden_spikes": hidden_spikes,
                "readout_pre": readout_pre}

    # ------------------------------------------------------------------ #
    # classification helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def classification_features(self, X: torch.Tensor, batch_size: int = 128
                                ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Mean reservoir firing-rate features [N] per sample for linear baseline.

        Also returns mean activity stats accumulated over the whole pass.
        """
        feats, res_rate, hid_rate = [], [], []
        for i in range(0, X.shape[0], batch_size):
            fb = self.reservoir_forward(X[i:i + batch_size], return_traces=False)
            res_spikes = fb["res_spikes"]
            pre = fb["readout_pre"]
            feats.append(pre.mean(dim=1).cpu().numpy())            # [b, readout_in]
            res_rate.append(float(res_spikes.mean().item()))
            if fb["hidden_spikes"] is not None:
                hid_rate.append(float(fb["hidden_spikes"].mean().item()))
        stats = {"reservoir_rate": float(np.mean(res_rate))}
        if hid_rate:
            stats["hidden_rate"] = float(np.mean(hid_rate))
        return np.concatenate(feats, axis=0), stats

    # ------------------------------------------------------------------ #
    # regression helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def regression_states(self, X_series: torch.Tensor,
                          decays: Optional[List[float]] = None) -> torch.Tensor:
        """Single series [1,T,C] -> multi-timescale reservoir states [T, N*len(decays)].

        The reservoir spike train is low-pass filtered at several decay rates so
        the linear readout sees a fading memory at multiple timescales.
        """
        fb = self.reservoir_forward(X_series, return_traces=False)
        spikes = fb["res_spikes"][0]                               # [T, N]
        decays = decays or self.reg_trace_decays
        T, N = spikes.shape
        feats = torch.empty(T, N * len(decays), device=spikes.device)
        for k, d in enumerate(decays):
            tr = torch.zeros(N, device=spikes.device)
            col = slice(k * N, (k + 1) * N)
            for t in range(T):
                tr = d * tr + spikes[t]
                feats[t, col] = tr
        return feats


# --------------------------------------------------------------------------- #
# Readouts for regression (no BPTT)
# --------------------------------------------------------------------------- #
class RidgeReadout:
    """Closed-form ridge regression readout (classical reservoir computing).

    Standardizes features with **train statistics only**, fits in double
    precision, and (optionally) selects the ridge penalty on a validation split.
    This is the canonical, non-plastic reservoir-computing readout baseline.
    """

    def __init__(self):
        self.W = None
        self.mu = None
        self.sd = None
        self.alpha = None

    def _design(self, S):
        S = ((S.detach().cpu().double() - self.mu) / self.sd)
        return torch.cat([S, torch.ones(S.shape[0], 1, dtype=torch.double)], dim=1)

    def fit(self, S_tr, y_tr, S_val=None, y_val=None,
            alphas=(1e0, 1e1, 1e2, 1e3, 1e4)):
        # NOTE: the penalty grid is floored at 1.0. Reservoir trace features are
        # highly collinear (multi-timescale), so tiny penalties can produce a
        # near-singular solve that overfits the (short) validation split and
        # explodes on test -- heavier Tikhonov regularization is the standard,
        # stable choice for reservoir-computing readouts.
        St = S_tr.detach().cpu().double()
        self.mu = St.mean(0, keepdim=True)
        # floor the std so rare-firing reservoir neurons (std -> 0) are not blown
        # up by standardization, which would make the ridge solve ill-conditioned.
        self.sd = St.std(0, keepdim=True).clamp(min=0.05)
        A = self._design(S_tr)
        Y = y_tr.detach().cpu().double()
        I = torch.eye(A.shape[1], dtype=torch.double)
        AtA, AtY = A.T @ A, A.T @ Y

        def solve(alpha):
            return torch.linalg.solve(AtA + alpha * I, AtY)

        if S_val is not None and len(alphas) > 1:
            best = None
            for a in alphas:
                W = solve(a)
                pred = self._design(S_val) @ W
                err = float(((pred - y_val.detach().cpu().double()) ** 2).mean())
                if best is None or err < best[0]:
                    best = (err, a, W)
            _, self.alpha, self.W = best
        else:
            self.alpha = alphas[0] if isinstance(alphas, (list, tuple)) else alphas
            self.W = solve(self.alpha)
        return self

    def predict(self, S):
        return (self._design(S) @ self.W).float()


def ridge_readout(states_tr, y_tr, states_eval, alpha: float = 1e-2):
    """Functional single-alpha ridge readout (kept for tests / simple calls)."""
    r = RidgeReadout().fit(states_tr, y_tr, alphas=(alpha,))
    return r.predict(states_eval), r.W.float()


def local_delta_readout(states_tr, y_tr, states_eval, lr=0.3, epochs=15,
                        w_clip: Optional[float] = None):
    """Local error-modulated (three-factor) readout for regression.

    The update is purely local -- each weight changes by (pre-activity x error),
    where the scalar error is the third (modulatory) factor. No backprop / BPTT;
    this is the regression analogue of the R-STDP rule. We use the *normalized*
    LMS variant (step divided by the input power) so it is stable for any number
    of correlated reservoir features and ``lr`` lives in (0, 2).
    """
    # mean-center with train stats (train-only). We deliberately do NOT divide by
    # per-feature std: rare-firing reservoir neurons have tiny std, and dividing
    # would amplify their sparse activity and destabilize the update. The NLMS
    # step below already normalizes by the input power, handling feature scale.
    S = states_tr.detach()
    mu = S.mean(dim=0, keepdim=True)
    S = S - mu
    Se = states_eval.detach() - mu
    Y = y_tr.detach()
    N, out = S.shape[1], Y.shape[1]
    w = torch.zeros(N + 1, out, device=S.device)
    Sb = torch.cat([S, torch.ones(S.shape[0], 1, device=S.device)], dim=1)
    for _ in range(epochs):
        for t in range(Sb.shape[0]):
            pre = Sb[t]
            err = Y[t] - pre @ w
            w = w + lr * torch.outer(pre, err) / (pre.dot(pre) + 1.0)  # normalized LMS
            if w_clip is not None:
                w = w.clamp(-w_clip, w_clip)
    Seb = torch.cat([Se, torch.ones(Se.shape[0], 1, device=Se.device)], dim=1)
    return (Seb @ w), w
