"""Metrics for classification, prediction and spiking-activity reporting.

All spike tensors follow the convention ``[batch, time, neurons]`` and are 0/1.
Continuous signals follow ``[batch, time, channels]``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .utils import to_numpy


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def accuracy(y_true, y_pred) -> float:
    yt, yp = to_numpy(y_true).ravel(), to_numpy(y_pred).ravel()
    if yt.size == 0:
        return float("nan")
    return float((yt == yp).mean())


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    yt, yp = to_numpy(y_true).ravel().astype(int), to_numpy(y_pred).ravel().astype(int)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(yt, yp):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


# --------------------------------------------------------------------------- #
# Regression / prediction
# --------------------------------------------------------------------------- #
def mse(y_true, y_pred) -> float:
    yt, yp = to_numpy(y_true).ravel(), to_numpy(y_pred).ravel()
    return float(np.mean((yt - yp) ** 2))


def mae(y_true, y_pred) -> float:
    yt, yp = to_numpy(y_true).ravel(), to_numpy(y_pred).ravel()
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def nrmse(y_true, y_pred) -> float:
    """Normalized RMSE (by target std). Standard for Mackey-Glass / NARMA."""
    yt = to_numpy(y_true).ravel()
    denom = yt.std()
    if denom == 0:
        return float("nan")
    return float(rmse(y_true, y_pred) / denom)


# --------------------------------------------------------------------------- #
# Spiking activity
# --------------------------------------------------------------------------- #
def spike_rate(spikes: torch.Tensor) -> float:
    """Mean spikes per neuron per timestep (a 'firing probability' in [0,1])."""
    if spikes is None or spikes.numel() == 0:
        return 0.0
    return float(spikes.float().mean().item())


def sparsity(spikes: torch.Tensor) -> float:
    """Fraction of (neuron, timestep) entries that did NOT spike.

    Higher = sparser activity. ``sparsity = 1 - mean(spikes)``.
    """
    if spikes is None or spikes.numel() == 0:
        return 1.0
    return float(1.0 - spikes.float().mean().item())


def mean_spike_count_per_sample(spikes: torch.Tensor) -> float:
    """Average total number of spikes emitted per input sample (over the seq)."""
    if spikes is None or spikes.numel() == 0:
        return 0.0
    # [B, T, N] -> sum over T,N -> mean over B
    return float(spikes.float().sum(dim=(1, 2)).mean().item())


# --------------------------------------------------------------------------- #
# Parameter / structure counts
# --------------------------------------------------------------------------- #
def count_structure(
    n_input_channels: int,
    n_reservoir: int,
    reservoir_connectivity: float,
    n_hidden: int,
    n_output: int,
    trainable_layers: List[str],
) -> Dict[str, int]:
    """Count neurons, synapses and trainable synapses for the assembled network.

    ``trainable_layers`` is a subset of {'in_hidden','res_hidden','res_out',
    'hidden_out'} naming which projections carry plastic (R-STDP) weights.
    """
    n_neurons = n_reservoir + n_hidden + n_output

    res_rec = int(round(n_reservoir * n_reservoir * reservoir_connectivity))
    in_res = n_input_channels * n_reservoir
    if n_hidden > 0:
        res_hidden = n_reservoir * n_hidden
        hidden_out = n_hidden * n_output
        res_out = 0
        readout_syn = res_hidden + hidden_out
    else:
        res_hidden = hidden_out = 0
        res_out = n_reservoir * n_output
        readout_syn = res_out

    n_synapses = in_res + res_rec + readout_syn

    trainable = 0
    name_to_count = {
        "res_hidden": res_hidden,
        "hidden_out": hidden_out,
        "res_out": res_out,
        "in_res": in_res,
    }
    for name in trainable_layers:
        trainable += name_to_count.get(name, 0)

    return {
        "n_neurons": int(n_neurons),
        "n_synapses": int(n_synapses),
        "n_trainable_synapses": int(trainable),
        "n_reservoir_recurrent_synapses": int(res_rec),
        "n_input_synapses": int(in_res),
        "n_readout_synapses": int(readout_syn),
    }


# --------------------------------------------------------------------------- #
# Assembly helper
# --------------------------------------------------------------------------- #
def summarize_run(
    task_type: str,
    y_true=None,
    y_pred=None,
    n_classes: Optional[int] = None,
    activity: Optional[Dict[str, float]] = None,
    structure: Optional[Dict[str, int]] = None,
    timing: Optional[Dict[str, float]] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Assemble the standard results dict requested by the experiment spec."""
    res: Dict = {"task_type": task_type}
    if task_type == "classification" and y_true is not None:
        res["accuracy"] = accuracy(y_true, y_pred)
        if n_classes is not None:
            res["confusion_matrix"] = confusion_matrix(y_true, y_pred, n_classes).tolist()
    elif task_type == "regression" and y_true is not None:
        res["mse"] = mse(y_true, y_pred)
        res["mae"] = mae(y_true, y_pred)
        res["rmse"] = rmse(y_true, y_pred)
        res["nrmse"] = nrmse(y_true, y_pred)
    if activity:
        res["activity"] = activity
    if structure:
        res["structure"] = structure
    if timing:
        res["timing"] = timing
    if extra:
        res.update(extra)
    return res
