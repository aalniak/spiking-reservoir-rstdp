#!/usr/bin/env python3
"""Compose a single 'results at a glance' figure from saved artifacts.

Reads (no training):
  results/<exp>/<method>/metrics.json   (per-experiment method results, curves)
  results/summary.json                  (sweeps + quant; emitted by make_report_figures.py)

Writes:
  results/figures/combined_results.png

Run the experiments + `scripts/make_report_figures.py` first so the inputs exist.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CLR = {"rstdp": "#d62728", "linear": "#1f77b4", "surrogate": "#2ca02c", "gru": "#7f7f7f"}
LBL = {"rstdp": "R-STDP", "linear": "Linear RC", "surrogate": "Surrogate (BPTT)", "gru": "GRU"}


def load(exp, method):
    p = f"results/{exp}/{method}/metrics.json"
    return json.load(open(p)) if os.path.exists(p) else None


def grouped_bars(ax, datasets, methods, value_fn, ylabel, title, ylim=None, annotate=True):
    n = len(methods)
    width = 0.8 / n
    x = np.arange(len(datasets))
    for k, m in enumerate(methods):
        vals = [value_fn(ds, m) for ds, _ in datasets]
        bars = ax.bar(x + (k - (n - 1) / 2) * width, vals, width,
                      label=LBL.get(m, m), color=CLR.get(m, None))
        if annotate:
            for b, v in zip(bars, vals):
                if v is not None and not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([d for _, d in datasets])
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)


def main():
    summary = json.load(open("results/summary.json")) if os.path.exists("results/summary.json") else {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # (0,0) classification accuracy ---------------------------------------------
    clf_ds = [("basicmotions", "BasicMotions"), ("synthetic_iq_jamming", "IQ-jamming")]
    clf_methods = ["rstdp", "linear", "surrogate", "gru"]

    def clf_val(exp, m):
        d = load(exp, m)
        return d.get("test_accuracy") if d else None
    grouped_bars(axes[0, 0], clf_ds, clf_methods, clf_val,
                 "test accuracy (↑)", "Classification accuracy by method", ylim=(0, 1.08))

    # (0,1) prediction NRMSE ----------------------------------------------------
    reg_ds = [("mackey_glass", "Mackey-Glass"), ("narma10", "NARMA10")]
    reg_methods = ["rstdp", "linear", "gru"]

    def reg_val(exp, m):
        d = load(exp, m)
        return d.get("nrmse") if d else None
    grouped_bars(axes[0, 1], reg_ds, reg_methods, reg_val,
                 "test NRMSE (↓)", "Prediction error by method", ylim=(0, 1.0))
    axes[0, 1].legend([LBL[m] if m != "rstdp" else "R-STDP (local)" for m in reg_methods],
                      fontsize=7)

    # (0,2) R-STDP learning curve (BasicMotions) --------------------------------
    ax = axes[0, 2]
    bm = load("basicmotions", "rstdp")
    if bm and bm.get("reward_curve"):
        ax.plot(bm["reward_curve"], color="#d62728", label="reward (1 − p_correct)")
        ax.set_ylabel("reward signal", color="#d62728"); ax.set_xlabel("epoch")
        ax2 = ax.twinx()
        ax2.plot(bm.get("train_acc_curve", []), color="#1f77b4", label="train accuracy")
        ax2.set_ylabel("train accuracy", color="#1f77b4"); ax2.set_ylim(0, 1.0)
    ax.set_title("R-STDP learning curve (BasicMotions)", fontsize=11)
    ax.grid(alpha=0.3)

    # (1,0) reservoir-size sweep ------------------------------------------------
    ax = axes[1, 0]
    ss = summary.get("size_sweep", {})
    style = {"mackey_glass": "#1f77b4", "narma10": "#ff7f0e"}
    for name, d in ss.items():
        c = style.get(name, None)
        ax.plot(d["sizes"], d["local"], "o-", color=c, label=f"{name} R-STDP")
        ax.plot(d["sizes"], d["ridge"], "s--", color=c, alpha=0.6, label=f"{name} ridge")
    ax.set_xlabel("reservoir size N"); ax.set_ylabel("test NRMSE (↓)")
    ax.set_title("Reservoir-size sweep (prediction)", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (1,1) quantization robustness ---------------------------------------------
    ax = axes[1, 1]
    qs = summary.get("quant_sweep", {})
    for name, d in qs.items():
        ax.plot(range(len(d["acc"])), d["acc"], "o-", label=name)
    ax.axhline(0.25, ls=":", color="gray", lw=1, label="chance (BM)")
    if qs:
        bits = next(iter(qs.values()))["bits"]
        xlabels = ["FP" if b == 0 else str(b) for b in bits]
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels)
    ax.set_xlabel("weight/state/trace bits (decreasing precision →)")
    ax.set_ylabel("R-STDP test accuracy")
    ax.set_title("Quantization robustness", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    # (1,2) IQ-jamming OOD generalization ---------------------------------------
    ax = axes[1, 2]
    methods = ["rstdp", "linear", "surrogate", "gru"]
    indist, ood = [], []
    for m in methods:
        d = load("synthetic_iq_jamming", m)
        indist.append(d.get("test_accuracy") if d else None)
        ood.append(d.get("test_ood_accuracy") if d else None)
    x = np.arange(len(methods))
    ax.bar(x - 0.2, indist, 0.4, label="in-distribution", color="#4c72b0")
    ax.bar(x + 0.2, ood, 0.4, label="unseen SNR/JNR", color="#dd8452")
    for xi, (a, b) in enumerate(zip(indist, ood)):
        if a:
            ax.text(xi - 0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
        if b:
            ax.text(xi + 0.2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([LBL[m] for m in methods], fontsize=8, rotation=15)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.08)
    ax.set_title("IQ-jamming: generalization to unseen SNR/JNR", fontsize=11)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Reservoir-based spiking models + R-STDP local learning — results "
                 "(CPU/GPU prototype)", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = "results/figures/combined_results.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    if not os.path.exists("results/summary.json"):
        print("NOTE: results/summary.json not found -- run "
              "`python scripts/make_report_figures.py` first (sweep/quant panels need it).",
              file=sys.stderr)
    main()
