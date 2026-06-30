"""Datasets: synthetic prediction tasks, synthetic IQ jamming, and UEA loaders.

Everything is returned as a :class:`DatasetBundle` with tensors shaped
``[batch, time, channels]``. Normalization always uses **training statistics
only**. Random seeds are honoured throughout.

Task types
----------
* ``regression``     -- single long series per split (``[1, T, C]``), reservoir
                        run once per split with a washout (Mackey-Glass, NARMA10).
* ``classification`` -- many short samples (``[B, T, C]``) with integer labels
                        (BasicMotions, CharacterTrajectories, IQ jamming).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
@dataclass
class DatasetBundle:
    name: str
    task_type: str                     # 'classification' | 'regression'
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    n_classes: Optional[int] = None    # classification
    out_dim: int = 1                   # regression target dimensionality / n_classes
    washout: int = 0                   # regression: states to discard per series
    channel_names: Optional[list] = None
    extra: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

    @property
    def in_channels(self) -> int:
        return self.X_train.shape[-1]


# --------------------------------------------------------------------------- #
# normalization helpers (train-stats only)
# --------------------------------------------------------------------------- #
def _channel_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean/std per channel over all but the last axis."""
    dims = tuple(range(x.dim() - 1))
    mu = x.mean(dim=dims, keepdim=True)
    sd = x.std(dim=dims, keepdim=True) + 1e-6
    return mu, sd


def _apply_norm(x, mu, sd):
    return (x - mu) / sd


# --------------------------------------------------------------------------- #
# A1. Mackey-Glass
# --------------------------------------------------------------------------- #
def mackey_glass_series(length: int, tau: int = 17, beta: float = 0.2,
                        gamma: float = 0.1, n: float = 10.0, dt: float = 1.0,
                        seed: int = 0, discard: int = 250) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hist_len = tau + 1
    x = np.empty(length + discard + hist_len)
    x[:hist_len] = 1.2 + 0.2 * (rng.random(hist_len) - 0.5)
    for t in range(hist_len - 1, len(x) - 1):
        xtau = x[t - tau]
        x[t + 1] = x[t] + dt * (beta * xtau / (1.0 + xtau ** n) - gamma * x[t])
    return x[hist_len + discard:]


def _make_mackey_glass(cfg, seed) -> DatasetBundle:
    horizon = cfg.get("horizon", 1)
    lengths = (cfg.get("train_len", 1500), cfg.get("val_len", 500), cfg.get("test_len", 1000))
    washout = cfg.get("washout", 100)
    series = {}
    for i, (split, L) in enumerate(zip(("train", "val", "test"), lengths)):
        s = mackey_glass_series(L + horizon, tau=cfg.get("tau", 17), seed=seed + i)
        series[split] = s

    mu = series["train"][:-horizon].mean()
    sd = series["train"][:-horizon].std() + 1e-6

    def to_xy(s):
        s = (s - mu) / sd
        u = s[:-horizon]
        y = s[horizon:]
        X = torch.tensor(u, dtype=torch.float32).reshape(1, -1, 1)
        Y = torch.tensor(y, dtype=torch.float32).reshape(1, -1, 1)
        return X, Y

    Xtr, ytr = to_xy(series["train"])
    Xva, yva = to_xy(series["val"])
    Xte, yte = to_xy(series["test"])
    return DatasetBundle("mackey_glass", "regression", Xtr, ytr, Xva, yva, Xte, yte,
                         out_dim=1, washout=washout, channel_names=["x"],
                         meta={"horizon": horizon})


# --------------------------------------------------------------------------- #
# A2. NARMA10
# --------------------------------------------------------------------------- #
def narma10_series(length: int, seed: int = 0, order: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=length)
    y = np.zeros(length)
    for t in range(order, length):
        y[t] = (0.3 * y[t - 1]
                + 0.05 * y[t - 1] * np.sum(y[t - order:t])
                + 1.5 * u[t - order] * u[t - 1]
                + 0.1)
        y[t] = np.clip(y[t], -2.0, 2.0)  # NARMA can diverge; clip for stability
    return u, y


def _make_narma10(cfg, seed) -> DatasetBundle:
    lengths = (cfg.get("train_len", 1500), cfg.get("val_len", 500), cfg.get("test_len", 1000))
    washout = cfg.get("washout", 100)
    order = cfg.get("order", 10)
    data = {}
    for i, (split, L) in enumerate(zip(("train", "val", "test"), lengths)):
        data[split] = narma10_series(L, seed=seed + 100 * i, order=order)

    u_tr = data["train"][0]
    umu, usd = u_tr.mean(), u_tr.std() + 1e-6
    ymu, ysd = data["train"][1].mean(), data["train"][1].std() + 1e-6

    def to_xy(uy):
        u, y = uy
        X = torch.tensor((u - umu) / usd, dtype=torch.float32).reshape(1, -1, 1)
        Y = torch.tensor((y - ymu) / ysd, dtype=torch.float32).reshape(1, -1, 1)
        return X, Y

    Xtr, ytr = to_xy(data["train"])
    Xva, yva = to_xy(data["val"])
    Xte, yte = to_xy(data["test"])
    return DatasetBundle("narma10", "regression", Xtr, ytr, Xva, yva, Xte, yte,
                         out_dim=1, washout=washout, channel_names=["u"],
                         meta={"order": order})


# --------------------------------------------------------------------------- #
# C. Synthetic IQ jamming detection
# --------------------------------------------------------------------------- #
_CONSTELLATIONS = {
    "bpsk": np.array([1 + 0j, -1 + 0j]),
    "qpsk": np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2),
    "16qam": (np.array([a + 1j * b for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)])
              / np.sqrt(10)),
}


def _modulated_signal(T, sps, mod, rng) -> np.ndarray:
    n_sym = int(np.ceil(T / sps))
    const = _CONSTELLATIONS[mod]
    syms = const[rng.integers(0, len(const), size=n_sym)]
    sig = np.repeat(syms, sps)[:T]              # rectangular pulse shaping
    return sig.astype(np.complex128)


def _add_awgn(sig, snr_db, rng):
    p_sig = np.mean(np.abs(sig) ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    noise = np.sqrt(p_noise / 2) * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
    return sig + noise


def _jammer(T, kind, jnr_db, sig_power, rng) -> np.ndarray:
    p_jam = sig_power * (10 ** (jnr_db / 10))
    t = np.arange(T)
    if kind == "tone":
        f = rng.uniform(0.05, 0.45)
        j = np.exp(1j * (2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)))
    elif kind == "broadband":
        j = (rng.standard_normal(T) + 1j * rng.standard_normal(T)) / np.sqrt(2)
    elif kind == "pulsed":
        f = rng.uniform(0.05, 0.45)
        j = np.exp(1j * 2 * np.pi * f * t)
        duty = rng.uniform(0.2, 0.5)
        mask = (rng.random(T) < duty).astype(float)
        j = j * mask
    elif kind == "swept":
        f0, f1 = 0.02, 0.48
        inst = f0 + (f1 - f0) * (t / max(1, T))
        j = np.exp(1j * 2 * np.pi * np.cumsum(inst))
    else:
        raise ValueError(f"unknown jammer '{kind}'")
    j = j / (np.sqrt(np.mean(np.abs(j) ** 2)) + 1e-12)
    return np.sqrt(p_jam) * j


def generate_iq_dataset(n_per_class, T, sps, mods, snr_range, jnr_range,
                        jammers, seed) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_per_class):
        for cls in (0, 1):
            mod = mods[rng.integers(0, len(mods))]
            snr = rng.uniform(*snr_range)
            sig = _modulated_signal(T, sps, mod, rng)
            sig = sig / (np.sqrt(np.mean(np.abs(sig) ** 2)) + 1e-12)
            rx = _add_awgn(sig, snr, rng)
            if cls == 1:
                kind = jammers[rng.integers(0, len(jammers))]
                jnr = rng.uniform(*jnr_range)
                rx = rx + _jammer(T, kind, jnr, 1.0, rng)
            X.append(np.stack([rx.real, rx.imag], axis=-1))
            y.append(cls)
    X = np.asarray(X, dtype=np.float32)         # [N, T, 2]
    y = np.asarray(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _make_iq_jamming(cfg, seed) -> DatasetBundle:
    T = cfg.get("T", 128)
    sps = cfg.get("sps", 4)
    mods = cfg.get("modulations", ["bpsk", "qpsk", "16qam"])
    jammers = cfg.get("jammers", ["tone", "broadband", "pulsed", "swept"])
    npc = cfg.get("n_per_class", 600)
    snr = cfg.get("snr_range", [10.0, 20.0])
    jnr = cfg.get("jnr_range", [5.0, 15.0])

    Xtr, ytr = generate_iq_dataset(npc, T, sps, mods, snr, jnr, jammers, seed)
    Xva, yva = generate_iq_dataset(npc // 4, T, sps, mods, snr, jnr, jammers, seed + 1)
    Xte, yte = generate_iq_dataset(npc // 3, T, sps, mods, snr, jnr, jammers, seed + 2)

    # out-of-distribution generalization split (unseen SNR/JNR)
    ood_snr = cfg.get("ood_snr_range", [0.0, 10.0])
    ood_jnr = cfg.get("ood_jnr_range", [0.0, 5.0])
    Xood, yood = generate_iq_dataset(npc // 3, T, sps, mods, ood_snr, ood_jnr, jammers, seed + 3)

    Xtr, Xva, Xte, Xood = map(lambda a: torch.tensor(a), (Xtr, Xva, Xte, Xood))
    mu, sd = _channel_stats(Xtr)
    Xtr, Xva, Xte, Xood = (_apply_norm(a, mu, sd) for a in (Xtr, Xva, Xte, Xood))
    to_t = lambda a: torch.tensor(a, dtype=torch.long)

    bundle = DatasetBundle("synthetic_iq_jamming", "classification",
                           Xtr, to_t(ytr), Xva, to_t(yva), Xte, to_t(yte),
                           n_classes=2, out_dim=2, channel_names=["I", "Q"],
                           meta={"snr_range": snr, "jnr_range": jnr,
                                 "ood_snr_range": ood_snr, "ood_jnr_range": ood_jnr})
    bundle.extra["test_ood"] = (Xood, to_t(yood))
    return bundle


# --------------------------------------------------------------------------- #
# B. UEA classification via aeon (BasicMotions, CharacterTrajectories, ...)
# --------------------------------------------------------------------------- #
def _collection_to_btc(X) -> np.ndarray:
    """Convert an aeon collection to a dense [N, T, C] array (edge-padded)."""
    if isinstance(X, np.ndarray) and X.ndim == 3:
        return np.transpose(X, (0, 2, 1))       # [N,C,T] -> [N,T,C]
    # unequal-length: list / object array of [C, t_i]
    seqs = [np.asarray(s) for s in X]
    Tmax = max(s.shape[-1] for s in seqs)
    C = seqs[0].shape[0]
    out = np.zeros((len(seqs), Tmax, C), dtype=np.float32)
    for i, s in enumerate(seqs):
        ti = s.shape[-1]
        out[i, :ti] = s.T
        if ti < Tmax:
            out[i, ti:] = s[:, -1]              # edge pad to avoid delta artifacts
    return out


def load_uea(name, seed, val_fraction=0.2, max_train=None):
    from aeon.datasets import load_classification
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xtr, ytr = load_classification(name, split="train")
        Xte, yte = load_classification(name, split="test")
    Xtr, Xte = _collection_to_btc(Xtr), _collection_to_btc(Xte)

    classes = sorted(set(ytr.tolist()) | set(yte.tolist()))
    cmap = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cmap[c] for c in ytr], dtype=np.int64)
    yte = np.array([cmap[c] for c in yte], dtype=np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    Xtr, ytr = Xtr[idx], ytr[idx]
    if max_train:
        Xtr, ytr = Xtr[:max_train], ytr[:max_train]
    n_val = max(1, int(val_fraction * len(ytr)))
    Xva, yva = Xtr[:n_val], ytr[:n_val]
    Xtr, ytr = Xtr[n_val:], ytr[n_val:]
    return (Xtr, ytr), (Xva, yva), (Xte, yte), len(classes)


def _make_uea(cfg, seed) -> DatasetBundle:
    name = cfg["uea_name"]
    (Xtr, ytr), (Xva, yva), (Xte, yte), n_classes = load_uea(
        name, seed, cfg.get("val_fraction", 0.2), cfg.get("max_train", None))
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xva = torch.tensor(Xva, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    mu, sd = _channel_stats(Xtr)
    Xtr, Xva, Xte = (_apply_norm(a, mu, sd) for a in (Xtr, Xva, Xte))
    return DatasetBundle(name.lower(), "classification",
                         Xtr, torch.tensor(ytr), Xva, torch.tensor(yva),
                         Xte, torch.tensor(yte),
                         n_classes=n_classes, out_dim=n_classes,
                         channel_names=[f"c{i}" for i in range(Xtr.shape[-1])],
                         meta={"uea_name": name})


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
_BUILDERS = {
    "mackey_glass": _make_mackey_glass,
    "narma10": _make_narma10,
    "synthetic_iq_jamming": _make_iq_jamming,
    "iq_jamming": _make_iq_jamming,
    "uea": _make_uea,
}


def load_dataset(cfg: Dict, seed: int = 0) -> DatasetBundle:
    """Build a dataset from a config dict ``{'name': ..., **params}``."""
    name = cfg.get("name")
    if name in ("basicmotions", "charactertrajectories", "epilepsy",
                "natops", "racketsports", "selfregulationscp1"):
        cfg = dict(cfg)
        cfg.setdefault("uea_name", {
            "basicmotions": "BasicMotions",
            "charactertrajectories": "CharacterTrajectories",
            "epilepsy": "Epilepsy",
            "natops": "NATOPS",
            "racketsports": "RacketSports",
            "selfregulationscp1": "SelfRegulationSCP1",
        }[name])
        return _make_uea(cfg, seed)
    if name not in _BUILDERS:
        raise ValueError(f"unknown dataset '{name}'. options: "
                         f"{sorted(_BUILDERS) + ['basicmotions','charactertrajectories']}")
    return _BUILDERS[name](cfg, seed)
