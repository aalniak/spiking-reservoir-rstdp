"""Utility helpers: reproducibility, device handling, config IO, quantization.

Everything here is intentionally lightweight and dependency-light so the rest of
the codebase can stay focused on the spiking models and the R-STDP learning rule.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    """Return a torch device. ``prefer`` may be 'auto', 'cuda', 'cpu' or 'cuda:N'."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer.startswith("cuda") and not torch.cuda.is_available():
        print(f"[utils] requested '{prefer}' but CUDA unavailable -> falling back to CPU")
        return torch.device("cpu")
    return torch.device(prefer)


# --------------------------------------------------------------------------- #
# Config IO
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config into a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def save_config(cfg: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` style dotted overrides onto a nested config dict.

    Values are parsed as YAML scalars so ``500``, ``1e-3``, ``true`` and ``[1,2]``
    all behave as expected. Used by the CLI sweep scripts.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if isinstance(value, str):
            try:
                value = yaml.safe_load(value)
            except yaml.YAMLError:
                pass
        node[parts[-1]] = value
    return cfg


def parse_cli_overrides(tokens) -> Dict[str, Any]:
    """Parse ``key=value`` CLI tokens into an overrides dict."""
    out: Dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"override '{tok}' is not in key=value form")
        k, v = tok.split("=", 1)
        out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Filesystem / JSON
# --------------------------------------------------------------------------- #
def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not serialisable: {type(o)}")


class Timer:
    """Tiny context-manager timer: ``with Timer() as t: ...; t.seconds``."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


# --------------------------------------------------------------------------- #
# Quantization (simulated fixed-point, a.k.a. "fake quant")
# --------------------------------------------------------------------------- #
@dataclass
class QuantConfig:
    """Configuration for simulated fixed-point quantization.

    We simulate ``bits``-bit fixed point by rounding to the nearest of
    ``2**bits`` evenly spaced levels inside a symmetric ``[-abs_max, abs_max]``
    (or supplied) range, then de-quantizing back to float. This mirrors what a
    Loihi-2 / Lava integer datapath would store, while keeping autograd-free
    float math for the rest of the prototype.

    Set ``enabled=False`` (or ``bits<=0``) to disable quantization entirely.
    """

    enabled: bool = False
    bits: int = 8
    # which tensors to quantize (handled by callers / model)
    weights: bool = True
    states: bool = True   # membrane potentials
    traces: bool = True
    # optional explicit ranges; if None a symmetric per-tensor max-abs is used
    weight_range: Optional[float] = None
    state_range: Optional[float] = None
    trace_range: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "QuantConfig":
        if not d:
            return cls(enabled=False)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def quantize(
    x: torch.Tensor,
    bits: int,
    abs_max: Optional[float] = None,
    signed: bool = True,
) -> torch.Tensor:
    """Simulated uniform fixed-point quantization (quantize + de-quantize).

    Parameters
    ----------
    x : tensor to quantize.
    bits : bit-width. ``bits <= 0`` is a no-op (returns ``x``).
    abs_max : symmetric clipping range. If ``None`` it is taken from the data.
    signed : if True the range is ``[-abs_max, abs_max]`` (one level reserved
        so zero is representable), else ``[0, abs_max]``.
    """
    if bits is None or bits <= 0:
        return x
    if abs_max is None:
        abs_max = float(x.abs().max().item()) if x.numel() else 1.0
    if abs_max == 0:
        return torch.zeros_like(x)

    levels = 2 ** bits
    if signed:
        lo, hi = -abs_max, abs_max
        n = levels - 1  # number of intervals; symmetric incl. 0
    else:
        lo, hi = 0.0, abs_max
        n = levels - 1
    scale = (hi - lo) / n
    q = torch.round((x.clamp(lo, hi) - lo) / scale)
    return q * scale + lo


def maybe_quantize(x: torch.Tensor, qcfg: Optional[QuantConfig], kind: str) -> torch.Tensor:
    """Quantize ``x`` according to ``qcfg`` if enabled and ``kind`` is selected.

    ``kind`` is one of {'weights', 'states', 'traces'}.
    """
    if qcfg is None or not qcfg.enabled or qcfg.bits <= 0:
        return x
    if not getattr(qcfg, kind, True):
        return x
    rng = {
        "weights": qcfg.weight_range,
        "states": qcfg.state_range,
        "traces": qcfg.trace_range,
    }.get(kind)
    signed = kind != "traces"  # traces are non-negative
    return quantize(x, qcfg.bits, abs_max=rng, signed=signed)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict to dotted keys (handy for logging tables)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key + "."))
        else:
            out[key] = v
    return out
