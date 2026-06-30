"""Comparison baselines (OFF-CHIP, backpropagation-based -- NOT Loihi-friendly):

* ``surrogate`` : a spiking readout trained on the SAME fixed reservoir by
  surrogate-gradient BPTT. Isolates "local R-STDP vs. gradient" with everything
  else held fixed.
* ``gru``       : a small GRU/LSTM operating on the raw signal -- a conventional
  deep-learning reference.

These use ``torch.autograd`` / BPTT and are provided only for comparison; the
main method (``src.train_rstdp``) uses neither.

    python -m src.train_surrogate_baseline --config configs/basicmotions.yaml
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from . import evaluate as ev
from . import metrics
from .datasets import load_dataset
from .model import SpikingReservoirModel
from .neurons import surrogate_spike
from .train_rstdp import precompute_readout_pre
from .utils import (QuantConfig, Timer, apply_overrides, get_device, load_config,
                    parse_cli_overrides, save_json, set_seed)


# --------------------------------------------------------------------------- #
# surrogate-gradient spiking readout on the fixed reservoir
# --------------------------------------------------------------------------- #
class SurrogateReadout(nn.Module):
    """Spiking readout trained by surrogate-gradient BPTT.

    A learnable per-output ``bias`` and global ``log_gain`` let the optimizer find
    a firing regime where the surrogate gradient flows, regardless of how strongly
    a given reservoir drives the outputs (the bias is common-mode and so does not
    bias the argmax decision -- it only keeps neurons near threshold)."""

    def __init__(self, n_pre, n_out, leak=1.0, v_th=1.0, scale=5.0, current_scale=None):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_pre, n_out) * (1.0 / n_pre ** 0.5))
        self.bias = nn.Parameter(torch.full((n_out,), 0.2))
        self.log_gain = nn.Parameter(torch.zeros(1))
        self.leak, self.v_th, self.scale = leak, v_th, scale
        self.current_scale = current_scale or (15.0 / n_pre)

    def forward(self, pre):                       # pre [B,T,P] -> logits [B,n_out], spikes
        B, T, P = pre.shape
        v = torch.zeros(B, self.W.shape[1], device=pre.device)
        count = torch.zeros_like(v)
        gain = self.current_scale * torch.exp(self.log_gain)
        spikes = []
        for t in range(T):
            v = self.leak * v + gain * (pre[:, t] @ self.W) + self.bias
            s = surrogate_spike(v, self.v_th, self.scale)
            v = v - self.v_th * s
            count = count + s
            spikes.append(s)
        return count, torch.stack(spikes, dim=1)


def train_surrogate_classification(cfg, bundle, model, device) -> Dict:
    tr = cfg.get("training", {})
    epochs = tr.get("surrogate_epochs", 150)
    lr = tr.get("surrogate_lr", 5e-3)
    pre_tr = precompute_readout_pre(model, bundle.X_train, device)
    pre_te = precompute_readout_pre(model, bundle.X_test, device)
    pre_va = precompute_readout_pre(model, bundle.X_val, device)
    ytr = bundle.y_train.to(device)

    readout = SurrogateReadout(
        pre_tr.shape[-1], bundle.out_dim,
        scale=tr.get("surrogate_slope", 5.0),
        current_scale=tr.get("surrogate_current_scale", None)).to(device)
    opt = torch.optim.Adam(readout.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    loss_curve = []
    with Timer() as t_train:
        for ep in range(epochs):
            opt.zero_grad()
            logits, _ = readout(pre_tr)
            loss = lossf(logits, ytr)
            loss.backward(); opt.step()
            loss_curve.append(float(loss.item()))

    @torch.no_grad()
    def acc(pre, y):
        logits, sp = readout(pre)
        return float((logits.argmax(1) == y.to(device)).float().mean()), sp
    a_tr, _ = acc(pre_tr, bundle.y_train)
    a_va, _ = acc(pre_va, bundle.y_val)
    a_te, sp_te = acc(pre_te, bundle.y_test)
    cm = metrics.confusion_matrix(bundle.y_test.numpy(),
                                  readout(pre_te)[0].argmax(1).cpu().numpy(), bundle.n_classes)
    info = ev.activity_and_structure(model, bundle.X_train, bundle)
    res = {"method": "surrogate", "task_type": "classification",
           "train_accuracy": a_tr, "val_accuracy": a_va, "test_accuracy": a_te,
           "accuracy": a_te, "confusion_matrix": cm.tolist(),
           "activity": {**info["activity"], "output_spike_rate": metrics.spike_rate(sp_te)},
           "structure": info["structure"],
           "timing": {"train_time_s": t_train.seconds, "epochs": epochs},
           "training_note": "OFF-CHIP surrogate-gradient BPTT on a fixed reservoir.",
           "loss_curve": loss_curve}
    for split, (Xo, yo) in bundle.extra.items():
        pre_o = precompute_readout_pre(model, Xo, device)
        res[f"{split}_accuracy"], _ = acc(pre_o, yo)
    mdir = ev.method_dir(cfg, "surrogate")
    ev.plot_curve(loss_curve, os.path.join(mdir, "loss_curve.png"), ylabel="CE loss",
                  title="surrogate BPTT loss")
    return res


# --------------------------------------------------------------------------- #
# GRU / LSTM reference
# --------------------------------------------------------------------------- #
class GRUNet(nn.Module):
    def __init__(self, in_ch, hidden, out_dim, task, kind="gru"):
        super().__init__()
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn(in_ch, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)
        self.task = task

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.task == "classification":
            return self.fc(out[:, -1])          # last step -> logits
        return self.fc(out)                     # per-step regression


def train_gru(cfg, bundle, device) -> Dict:
    tr = cfg.get("training", {})
    hidden = tr.get("gru_hidden", 32)
    kind = tr.get("rnn_kind", "gru")
    epochs = tr.get("gru_epochs", 150)
    lr = tr.get("gru_lr", 5e-3)
    net = GRUNet(bundle.in_channels, hidden, bundle.out_dim, bundle.task_type, kind).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_curve = []

    if bundle.task_type == "classification":
        Xtr, ytr = bundle.X_train.to(device), bundle.y_train.to(device)
        lossf = nn.CrossEntropyLoss()
        bs = tr.get("batch_size", 128)
        with Timer() as t_train:
            for ep in range(epochs):
                perm = torch.randperm(Xtr.shape[0], device=device)
                el = 0.0
                for i in range(0, len(perm), bs):
                    idx = perm[i:i + bs]
                    opt.zero_grad(); loss = lossf(net(Xtr[idx]), ytr[idx])
                    loss.backward(); opt.step(); el += float(loss.item())
                loss_curve.append(el)

        @torch.no_grad()
        def acc(X, y):
            return float((net(X.to(device)).argmax(1) == y.to(device)).float().mean())
        res = {"method": "gru", "task_type": "classification",
               "train_accuracy": acc(bundle.X_train, bundle.y_train),
               "val_accuracy": acc(bundle.X_val, bundle.y_val),
               "test_accuracy": acc(bundle.X_test, bundle.y_test),
               "accuracy": acc(bundle.X_test, bundle.y_test),
               "timing": {"train_time_s": t_train.seconds},
               "training_note": f"OFF-CHIP {kind.upper()} BPTT on raw signal (DL reference)."}
        for split, (Xo, yo) in bundle.extra.items():
            res[f"{split}_accuracy"] = acc(Xo, yo)
    else:
        w0 = bundle.washout
        Xtr, ytr = bundle.X_train.to(device), bundle.y_train.to(device)
        lossf = nn.MSELoss()
        with Timer() as t_train:
            for ep in range(epochs):
                opt.zero_grad()
                pred = net(Xtr)[:, w0:]
                loss = lossf(pred, ytr[:, w0:])
                loss.backward(); opt.step(); loss_curve.append(float(loss.item()))

        @torch.no_grad()
        def predseries(X):
            return net(X.to(device))[:, w0:].cpu()
        pte = predseries(bundle.X_test)[0]; yte = bundle.y_test[0][w0:]
        res = {"method": "gru", "task_type": "regression",
               "mse": metrics.mse(yte, pte), "mae": metrics.mae(yte, pte),
               "rmse": metrics.rmse(yte, pte), "nrmse": metrics.nrmse(yte, pte),
               "timing": {"train_time_s": t_train.seconds},
               "training_note": f"OFF-CHIP {kind.upper()} BPTT (DL reference)."}
    mdir = ev.method_dir(cfg, "gru")
    ev.plot_curve(loss_curve, os.path.join(mdir, "loss_curve.png"), ylabel="loss",
                  title=f"{kind.upper()} training loss")
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-surrogate", action="store_true")
    ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), parse_cli_overrides(args.overrides))
    seed = cfg.get("seed", 0)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    qcfg = QuantConfig.from_dict(cfg.get("quantization"))
    bundle = load_dataset(cfg["dataset"], seed=seed)
    meta = {"task_type": bundle.task_type, "in_channels": bundle.in_channels,
            "out_dim": bundle.out_dim}
    model = SpikingReservoirModel(meta, cfg, device, seed=seed, qcfg=qcfg)
    print(f"[surrogate_baseline] {cfg['name']} | task={bundle.task_type} | device={device}")

    if not args.skip_surrogate and bundle.task_type == "classification":
        res = train_surrogate_classification(cfg, bundle, model, device)
        save_json(res, os.path.join(ev.method_dir(cfg, "surrogate"), "metrics.json"))
        print(f"  surrogate-BPTT test accuracy = {res['test_accuracy']:.4f}")

    if not args.skip_gru:
        res = train_gru(cfg, bundle, device)
        save_json(res, os.path.join(ev.method_dir(cfg, "gru"), "metrics.json"))
        if bundle.task_type == "classification":
            print(f"  GRU test accuracy = {res['test_accuracy']:.4f}")
        else:
            print(f"  GRU test NRMSE = {res['nrmse']:.4f}")


if __name__ == "__main__":
    main()
