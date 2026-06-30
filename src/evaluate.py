"""Evaluation utilities: plotting, linear-readout baselines, metrics assembly and
results aggregation. Imported by the training scripts and runnable standalone to
(a) compute the classical reservoir+linear baseline and (b) aggregate every
method's ``metrics.json`` for an experiment into a results table.

Usage
-----
    python -m src.evaluate --config configs/basicmotions.yaml          # linear baseline + aggregate
    python -m src.evaluate --config configs/basicmotions.yaml --aggregate-only
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics
from .datasets import load_dataset
from .model import SpikingReservoirModel, RidgeReadout
from .utils import (QuantConfig, Timer, ensure_dir, get_device, load_config,
                    load_json, save_json, set_seed)


# --------------------------------------------------------------------------- #
# experiment paths
# --------------------------------------------------------------------------- #
def experiment_dir(cfg: Dict) -> str:
    return os.path.join(cfg.get("output_dir", "results"), cfg["name"])


def method_dir(cfg: Dict, method: str) -> str:
    return ensure_dir(os.path.join(experiment_dir(cfg), method))


# --------------------------------------------------------------------------- #
# activity / structure reporting
# --------------------------------------------------------------------------- #
@torch.no_grad()
def activity_and_structure(model: SpikingReservoirModel, X_sample: torch.Tensor,
                           bundle) -> Dict:
    fb = model.reservoir_forward(X_sample[: min(64, len(X_sample))])
    res_spikes = fb["res_spikes"]
    activity = {
        "reservoir_spike_rate": metrics.spike_rate(res_spikes),
        "reservoir_sparsity": metrics.sparsity(res_spikes),
        "reservoir_spikes_per_sample": metrics.mean_spike_count_per_sample(res_spikes),
    }
    if fb["hidden_spikes"] is not None:
        activity["hidden_spike_rate"] = metrics.spike_rate(fb["hidden_spikes"])
    trainable = ["res_out"] if model.hidden is None else ["hidden_out"]
    structure = metrics.count_structure(
        n_input_channels=model.reservoir.in_channels,
        n_reservoir=model.reservoir_dim,
        reservoir_connectivity=getattr(model.res_cfg, "connectivity", 0.0),
        n_hidden=model.hidden_size,
        n_output=model.out_dim,
        trainable_layers=trainable,
    )
    return {"activity": activity, "structure": structure}


# --------------------------------------------------------------------------- #
# linear-readout baselines (Baseline 1: classical reservoir computing)
# --------------------------------------------------------------------------- #
def linear_baseline(cfg: Dict, device=None) -> Dict:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    device = device or get_device(cfg.get("device", "auto"))
    qcfg = QuantConfig.from_dict(cfg.get("quantization"))
    bundle = load_dataset(cfg["dataset"], seed=seed)
    meta = {"task_type": bundle.task_type, "in_channels": bundle.in_channels,
            "out_dim": bundle.out_dim}
    model = SpikingReservoirModel(meta, cfg, device, seed=seed, qcfg=qcfg)

    if bundle.task_type == "classification":
        from sklearn.linear_model import LogisticRegression
        Ftr, stats = model.classification_features(bundle.X_train.to(device))
        Fva, _ = model.classification_features(bundle.X_val.to(device))
        Fte, _ = model.classification_features(bundle.X_test.to(device))
        clf = LogisticRegression(max_iter=1000, C=cfg.get("linear", {}).get("C", 1.0))
        clf.fit(Ftr, bundle.y_train.numpy())
        out = metrics.summarize_run(
            "classification", bundle.y_test.numpy(), clf.predict(Fte),
            n_classes=bundle.n_classes,
            activity={"reservoir_spike_rate": stats["reservoir_rate"]},
            extra={"train_accuracy": metrics.accuracy(bundle.y_train.numpy(), clf.predict(Ftr)),
                   "val_accuracy": metrics.accuracy(bundle.y_val.numpy(), clf.predict(Fva)),
                   "test_accuracy": metrics.accuracy(bundle.y_test.numpy(), clf.predict(Fte))})
        for split, (Xo, yo) in bundle.extra.items():
            Fo, _ = model.classification_features(Xo.to(device))
            out[f"{split}_accuracy"] = metrics.accuracy(yo.numpy(), clf.predict(Fo))
    else:
        w0 = bundle.washout
        Str = model.regression_states(bundle.X_train.to(device))[w0:]
        Sva = model.regression_states(bundle.X_val.to(device))[w0:]
        Ste = model.regression_states(bundle.X_test.to(device))[w0:]
        ytr = bundle.y_train[0][w0:]; yva = bundle.y_val[0][w0:]; yte = bundle.y_test[0][w0:]
        ridge = RidgeReadout().fit(Str, ytr, Sva, yva,
                                   alphas=cfg.get("ridge", {}).get("alphas",
                                                                   [1e-4, 1e-2, 1, 10, 100, 1000]))
        pred = ridge.predict(Ste)
        out = metrics.summarize_run("regression", yte, pred,
                                    extra={"alpha": ridge.alpha,
                                           "val_nrmse": metrics.nrmse(yva, ridge.predict(Sva))})
        res_rate = float(model.reservoir_forward(bundle.X_train.to(device))["res_spikes"].mean())
        out["activity"] = {"reservoir_spike_rate": res_rate}
    out["method"] = "linear"
    return out


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_curve(values, path, xlabel="epoch", ylabel="value", title=""):
    plt.figure(figsize=(5, 3.2))
    plt.plot(values)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_reward_and_acc(reward_curve, acc_curve, path):
    fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
    ax1.plot(reward_curve, color="tab:red", label="reward (correct neuron)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("reward", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(acc_curve, color="tab:blue", label="train acc")
    ax2.set_ylabel("train accuracy", color="tab:blue")
    plt.title("R-STDP learning curve"); fig.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def plot_weight_hist(w_before, w_after, path):
    plt.figure(figsize=(5, 3.2))
    wb = np.asarray(w_before).ravel(); wa = np.asarray(w_after).ravel()
    bins = np.linspace(min(wb.min(), wa.min()), max(wb.max(), wa.max()), 60)
    plt.hist(wb, bins=bins, alpha=0.55, label="before", color="gray")
    plt.hist(wa, bins=bins, alpha=0.55, label="after R-STDP", color="tab:green")
    plt.xlabel("weight"); plt.ylabel("count"); plt.legend(); plt.title("Readout weights")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_confusion(cm, classes, path):
    cm = np.asarray(cm)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(fraction=0.046)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.ylabel("true"); plt.xlabel("predicted"); plt.title("Confusion matrix")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_raster(spikes_TN, path, title="spike raster", max_neurons=80):
    s = np.asarray(spikes_TN)
    T, N = s.shape
    N = min(N, max_neurons)
    plt.figure(figsize=(6, 3.2))
    ys, xs = np.where(s[:, :N].T > 0)
    plt.scatter(xs, ys, s=2, marker="|", color="black")
    plt.xlabel("time step"); plt.ylabel("neuron"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_prediction(y_true, y_pred, path, n=300):
    plt.figure(figsize=(7, 3))
    plt.plot(np.asarray(y_true).ravel()[:n], label="target", lw=1.2)
    plt.plot(np.asarray(y_pred).ravel()[:n], label="prediction", lw=1.0, alpha=0.8)
    plt.legend(); plt.xlabel("time step"); plt.ylabel("value"); plt.title("Prediction vs target")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_metric_vs_x(xs, series: Dict[str, List[float]], xlabel, ylabel, path, title=""):
    plt.figure(figsize=(5.2, 3.4))
    for name, ys in series.items():
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


# --------------------------------------------------------------------------- #
# aggregation across methods -> results table
# --------------------------------------------------------------------------- #
def aggregate(cfg: Dict) -> str:
    exp = experiment_dir(cfg)
    rows = []
    for mpath in sorted(glob.glob(os.path.join(exp, "*", "metrics.json"))):
        m = load_json(mpath)
        method = m.get("method", os.path.basename(os.path.dirname(mpath)))
        row = {"method": method, "task": m.get("task_type", "")}
        if m.get("task_type") == "classification":
            row["test_acc"] = m.get("test_accuracy", m.get("accuracy"))
            row["val_acc"] = m.get("val_accuracy")
            for k in m:
                if k.endswith("_ood_accuracy") or k == "test_ood_accuracy":
                    row["ood_acc"] = m[k]
        else:
            row["test_nrmse"] = m.get("nrmse")
            row["test_mse"] = m.get("mse")
            row["test_mae"] = m.get("mae")
        act = m.get("activity", {})
        row["res_rate"] = act.get("reservoir_spike_rate")
        row["out_rate"] = act.get("output_spike_rate")
        st = m.get("structure", {})
        row["trainable_syn"] = st.get("n_trainable_synapses")
        tm = m.get("timing", {})
        row["train_s"] = tm.get("train_time_s")
        rows.append(row)

    # markdown table
    if not rows:
        return "(no metrics found yet)"
    cols = list({k for r in rows for k in r})
    order = ["method", "task", "test_acc", "val_acc", "ood_acc",
             "test_nrmse", "test_mse", "test_mae", "res_rate", "out_rate",
             "trainable_syn", "train_s"]
    cols = [c for c in order if c in cols] + [c for c in cols if c not in order]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return "" if v is None else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    table = "\n".join(lines)
    out_md = os.path.join(exp, "results_table.md")
    with open(out_md, "w") as f:
        f.write(f"# Results: {cfg['name']}\n\n{table}\n")
    print(f"[evaluate] wrote {out_md}")
    print(table)
    return table


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    from .utils import apply_overrides, parse_cli_overrides
    cfg = apply_overrides(load_config(args.config), parse_cli_overrides(args.overrides))

    if not args.aggregate_only:
        with Timer() as t:
            res = linear_baseline(cfg)
        res.setdefault("timing", {})["train_time_s"] = t.seconds
        mdir = method_dir(cfg, "linear")
        save_json(res, os.path.join(mdir, "metrics.json"))
        print(f"[evaluate] linear baseline -> {mdir}/metrics.json")
        if res["task_type"] == "classification":
            print(f"  test_accuracy = {res.get('test_accuracy'):.4f}")
        else:
            print(f"  test NRMSE = {res.get('nrmse'):.4f}  MSE = {res.get('mse'):.5f}")
    aggregate(cfg)


if __name__ == "__main__":
    main()
