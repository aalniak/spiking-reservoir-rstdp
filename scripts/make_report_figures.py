#!/usr/bin/env python3
"""Run the sweeps + ablations and generate the cross-cutting report figures/tables.

Produces:
  results/figures/mse_vs_reservoir_size.png
  results/figures/acc_vs_quant_bits.png
  results/figures/spike_rate_vs_accuracy.png
  results/figures/ablations_basicmotions.png
  results/SUMMARY.md                (master results + sweep + ablation tables)

This calls the library directly (no subprocess) for speed. It is the reproducible
source of the numbers quoted in reports/technical_report.md.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import evaluate as ev
from src import metrics
from src.datasets import load_dataset
from src.model import SpikingReservoirModel, RidgeReadout, local_delta_readout
from src.utils import QuantConfig, get_device, load_config, set_seed

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)


# --------------------------------------------------------------------------- #
def build(cfg, bundle, device, seed=0, qcfg=None):
    meta = {"task_type": bundle.task_type, "in_channels": bundle.in_channels,
            "out_dim": bundle.out_dim}
    return SpikingReservoirModel(meta, cfg, device, seed=seed, qcfg=qcfg)


def train_rstdp_clf(model, bundle, device, epochs=200, batch=32):
    from src.train_rstdp import precompute_readout_pre
    pre_tr = precompute_readout_pre(model, bundle.X_train, device)
    pre_te = precompute_readout_pre(model, bundle.X_test, device)
    ytr = bundle.y_train.to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(ytr), device=device)
        for i in range(0, len(perm), batch):
            model.readout.train_step(pre_tr[perm[i:i + batch]], ytr[perm[i:i + batch]])
    pred = model.readout.predict(pre_te)
    acc = metrics.accuracy(bundle.y_test.numpy(), pred["pred"].cpu().numpy())
    rate = float(model.reservoir_forward(bundle.X_train[:64].to(device))["res_spikes"].mean())
    out_rate = metrics.spike_rate(pred["spikes"])
    return acc, rate, out_rate


def eval_rstdp_reg(model, bundle, device):
    w0 = bundle.washout
    Str = model.regression_states(bundle.X_train.to(device))[w0:]
    Sva = model.regression_states(bundle.X_val.to(device))[w0:]
    Ste = model.regression_states(bundle.X_test.to(device))[w0:]
    ytr, yva, yte = (bundle.y_train[0][w0:].to(device), bundle.y_val[0][w0:],
                     bundle.y_test[0][w0:])
    ridge = RidgeReadout().fit(Str, ytr, Sva, yva)
    nrmse_ridge = metrics.nrmse(yte, ridge.predict(Ste))
    pred_local, _ = local_delta_readout(Str, ytr, Ste)
    nrmse_local = metrics.nrmse(yte, pred_local.cpu())
    rate = float(model.reservoir_forward(bundle.X_train.to(device))["res_spikes"].mean())
    return nrmse_ridge, nrmse_local, rate


# --------------------------------------------------------------------------- #
def sweep_reservoir_size(device):
    print("== reservoir-size sweep (MG, NARMA) ==")
    sizes = [100, 300, 500, 1000]
    out = {}
    for name in ["mackey_glass", "narma10"]:
        cfg = load_config(f"configs/{name}.yaml")
        cfg["dataset"] = dict(cfg["dataset"],
                              train_len=1200, val_len=400, test_len=600)  # faster sweep
        bundle = load_dataset(cfg["dataset"], seed=0)
        ridge_e, local_e, rates = [], [], []
        for N in sizes:
            set_seed(0)
            c = dict(cfg); c["reservoir"] = dict(cfg["reservoir"]); c["reservoir"]["n_reservoir"] = N
            model = build(c, bundle, device, seed=0)
            nr, nl, rate = eval_rstdp_reg(model, bundle, device)
            ridge_e.append(nr); local_e.append(nl); rates.append(rate)
            print(f"  {name} N={N}: ridge NRMSE={nr:.3f} local NRMSE={nl:.3f} rate={rate:.3f}")
        out[name] = {"sizes": sizes, "ridge": ridge_e, "local": local_e, "rates": rates}
    # plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, name in zip(axes, out):
        ax.plot(sizes, out[name]["ridge"], "o-", label="linear (ridge)")
        ax.plot(sizes, out[name]["local"], "s-", label="R-STDP (local)")
        ax.set_title(name); ax.set_xlabel("reservoir size N"); ax.set_ylabel("test NRMSE")
        ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/mse_vs_reservoir_size.png", dpi=120); plt.close()
    return out


def sweep_quant(device):
    print("== quantization sweep (BasicMotions, IQ) ==")
    out = {}
    for name in ["basicmotions", "synthetic_iq_jamming"]:
        cfg = load_config(f"configs/{name}.yaml")
        bundle = load_dataset(cfg["dataset"], seed=0)
        bits_list = [0, 16, 8, 4]          # 0 = full precision; decreasing precision
        accs, rates = [], []
        ep = cfg.get("training", {}).get("epochs", 200)
        for bits in bits_list:
            set_seed(0)
            qcfg = QuantConfig(enabled=bits > 0, bits=bits or 8)
            model = build(cfg, bundle, device, seed=0, qcfg=qcfg)
            acc, rate, _ = train_rstdp_clf(model, bundle, device, epochs=ep,
                                           batch=cfg["training"]["batch_size"])
            accs.append(acc); rates.append(rate)
            print(f"  {name} bits={bits or 'FP'}: acc={acc:.3f} rate={rate:.3f}")
        out[name] = {"bits": bits_list, "acc": accs, "rate": rates}
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.4, 3.6))
    labels = {0: "FP"}
    xs = [str(b) if b else "FP" for b in out["basicmotions"]["bits"]]
    for name in out:
        plt.plot(xs, out[name]["acc"], "o-", label=name)
    plt.xlabel("weight/state/trace bits"); plt.ylabel("R-STDP test accuracy")
    plt.title("R-STDP accuracy vs quantization"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/acc_vs_quant_bits.png", dpi=120); plt.close()
    return out


def ablations_basicmotions(device):
    print("== ablations (BasicMotions) ==")
    base = load_config("configs/basicmotions.yaml")
    bundle = load_dataset(base["dataset"], seed=0)
    rows = []

    def run(tag, mut):
        set_seed(0)
        cfg = load_config("configs/basicmotions.yaml")
        mut(cfg)
        model = build(cfg, bundle, device, seed=0)
        acc, rate, out_rate = train_rstdp_clf(model, bundle, device,
                                              epochs=cfg["training"]["epochs"])
        rows.append((tag, acc, rate, out_rate))
        print(f"  {tag:28s} acc={acc:.3f} res_rate={rate:.3f} out_rate={out_rate:.3f}")

    run("encoder=posneg (default)", lambda c: None)
    run("encoder=rate", lambda c: c.__setitem__("encoder", {"name": "rate"}))
    run("encoder=threshold", lambda c: c.__setitem__("encoder", {"name": "threshold", "threshold": 0.3}))
    run("encoder=temporal_diff", lambda c: c.__setitem__("encoder", {"name": "temporal_diff"}))
    run("reward=pos+neg (default)", lambda c: None)
    run("reward=pos-only", lambda c: c["readout"].update({"reward_mode": "class_specific",
                                                           "neg_reward_scale": 0.0}))
    run("reward=class_specific+neg", lambda c: c["readout"].update({"reward_mode": "class_specific",
                                                                    "neg_reward_scale": 1.0}))
    run("homeostasis=on", lambda c: c["readout"].update({"homeo_beta": 0.02}))
    run("eligibility=pre (default)", lambda c: None)
    run("eligibility=stdp", lambda c: c["readout"].update({"eligibility_mode": "stdp"}))
    run("eligibility=hybrid", lambda c: c["readout"].update({"eligibility_mode": "hybrid"}))
    run("reservoir_sparsity conn=0.3", lambda c: c["reservoir"].update({"connectivity": 0.3}))
    return rows


def spike_rate_vs_acc_plot():
    import glob, json
    import matplotlib.pyplot as plt
    pts = []
    for mp in glob.glob("results/*/*/metrics.json"):
        m = json.load(open(mp))
        if m.get("task_type") != "classification":
            continue
        act = m.get("activity", {})
        if "reservoir_spike_rate" in act and m.get("test_accuracy") is not None:
            pts.append((act["reservoir_spike_rate"], m["test_accuracy"],
                        m.get("method", "?"), os.path.basename(os.path.dirname(os.path.dirname(mp)))))
    if not pts:
        return
    plt.figure(figsize=(5.4, 3.6))
    for rate, acc, method, ds in pts:
        plt.scatter(rate, acc, s=40)
        plt.annotate(f"{method}", (rate, acc), fontsize=7, alpha=0.7)
    plt.xlabel("reservoir spike rate"); plt.ylabel("test accuracy")
    plt.title("spike rate vs accuracy (classification)"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/spike_rate_vs_accuracy.png", dpi=120); plt.close()


def write_summary(size_sweep, quant_sweep, ablations):
    import glob, json
    lines = ["# Master results summary\n",
             "Generated by `scripts/make_report_figures.py`. "
             "This is a **CPU/GPU prototype**; not a Loihi-2 deployment.\n",
             "## Per-experiment method comparison\n"]
    # master table from saved metrics.json
    lines.append("| experiment | method | test_acc | test_nrmse | res_rate | out_rate | trainable_syn |")
    lines.append("|---|---|---|---|---|---|---|")
    for mp in sorted(glob.glob("results/*/*/metrics.json")):
        m = json.load(open(mp))
        exp = os.path.basename(os.path.dirname(os.path.dirname(mp)))
        act = m.get("activity", {}); st = m.get("structure", {})
        def f(v): return f"{v:.4g}" if isinstance(v, (int, float)) else ""
        lines.append(f"| {exp} | {m.get('method','')} | {f(m.get('test_accuracy'))} | "
                     f"{f(m.get('nrmse'))} | {f(act.get('reservoir_spike_rate'))} | "
                     f"{f(act.get('output_spike_rate'))} | {f(st.get('n_trainable_synapses'))} |")

    lines.append("\n## Reservoir-size sweep (test NRMSE)\n")
    lines.append("| dataset | N | ridge NRMSE | R-STDP-local NRMSE | reservoir rate |")
    lines.append("|---|---|---|---|---|")
    for name, d in size_sweep.items():
        for N, r, l, rate in zip(d["sizes"], d["ridge"], d["local"], d["rates"]):
            lines.append(f"| {name} | {N} | {r:.3f} | {l:.3f} | {rate:.3f} |")

    lines.append("\n## Quantization sweep (R-STDP test accuracy)\n")
    bits_hdr = next(iter(quant_sweep.values()))["bits"]
    hdr = ["FP" if b == 0 else f"{b}-bit" for b in bits_hdr]
    lines.append("| dataset | " + " | ".join(hdr) + " |")
    lines.append("|---|" + "|".join(["---"] * len(hdr)) + "|")
    for name, d in quant_sweep.items():
        lines.append(f"| {name} | " + " | ".join(f"{a:.3f}" for a in d["acc"]) + " |")

    lines.append("\n## Ablations (BasicMotions, R-STDP readout)\n")
    lines.append("| ablation | test_acc | reservoir_rate | output_rate |")
    lines.append("|---|---|---|---|")
    for tag, acc, rate, out_rate in ablations:
        lines.append(f"| {tag} | {acc:.3f} | {rate:.3f} | {out_rate:.3f} |")

    with open("results/SUMMARY.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # machine-readable dump so the combined-figure composer can read it without
    # re-running any training.
    import json
    json.dump({"size_sweep": size_sweep, "quant_sweep": quant_sweep,
               "ablations": ablations}, open("results/summary.json", "w"), indent=2)
    print("wrote results/SUMMARY.md, results/summary.json and figures in results/figures/")


def main():
    device = get_device("auto")
    t0 = time.time()
    size_sweep = sweep_reservoir_size(device)
    quant_sweep = sweep_quant(device)
    ablations = ablations_basicmotions(device)
    spike_rate_vs_acc_plot()
    write_summary(size_sweep, quant_sweep, ablations)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
