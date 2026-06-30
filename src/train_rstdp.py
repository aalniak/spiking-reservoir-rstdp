"""Train the MAIN method: R-STDP (classification) / local error-modulated readout
(regression). No backpropagation, no BPTT -- the only plastic weights are the
readout synapses, updated by the local three-factor rule in ``src.rstdp``.

    python -m src.train_rstdp --config configs/basicmotions.yaml
    python -m src.train_rstdp --config configs/mackey_glass.yaml reservoir.n_reservoir=500
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch

from . import evaluate as ev
from . import metrics
from .datasets import load_dataset
from .model import SpikingReservoirModel, local_delta_readout
from .utils import (QuantConfig, Timer, apply_overrides, get_device, load_config,
                    parse_cli_overrides, save_json, set_seed, to_numpy)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def precompute_readout_pre(model, X, device, batch_size=128):
    """Run encoder+reservoir(+hidden) once over X -> readout pre-spikes [B,T,P]."""
    outs = []
    for i in range(0, X.shape[0], batch_size):
        fb = model.reservoir_forward(X[i:i + batch_size].to(device), return_traces=False)
        outs.append(fb["readout_pre"])
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------- #
def run_classification(cfg, bundle, model, device) -> Dict:
    tr = cfg.get("training", {})
    epochs = tr.get("epochs", 200)
    batch_size = tr.get("batch_size", 128)
    seed = cfg.get("seed", 0)

    Xtr, ytr = bundle.X_train.to(device), bundle.y_train.to(device)
    pre_tr = precompute_readout_pre(model, bundle.X_train, device)
    pre_va = precompute_readout_pre(model, bundle.X_val, device)
    pre_te = precompute_readout_pre(model, bundle.X_test, device)

    readout = model.readout
    W_before = to_numpy(readout.initial_weights())

    reward_curve, acc_curve = [], []
    g = torch.Generator(device=device).manual_seed(seed)
    with Timer() as t_train:
        for ep in range(epochs):
            perm = torch.randperm(Xtr.shape[0], generator=g, device=device)
            ep_acc, ep_rew, nb = 0.0, 0.0, 0
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                info = readout.train_step(pre_tr[idx], ytr[idx],
                                          teacher_forcing=cfg.get("readout", {}).get(
                                              "teacher_forcing", False))
                ep_acc += info["acc"]; ep_rew += info["reward_correct"]; nb += 1
            reward_curve.append(ep_rew / nb); acc_curve.append(ep_acc / nb)

    # --- evaluation ---
    def evprd(pre, y):
        out = readout.predict(pre)
        return out, metrics.accuracy(y.cpu().numpy(), out["pred"].cpu().numpy())

    out_tr, acc_tr = evprd(pre_tr, ytr)
    _, acc_va = evprd(pre_va, bundle.y_val.to(device))
    out_te, acc_te = evprd(pre_te, bundle.y_test.to(device))

    # timing: inference per sample
    with Timer() as t_inf:
        _ = readout.predict(pre_te)
    inf_per_sample = t_inf.seconds / max(1, pre_te.shape[0])

    cm = metrics.confusion_matrix(bundle.y_test.numpy(), out_te["pred"].cpu().numpy(),
                                  bundle.n_classes)
    info = ev.activity_and_structure(model, bundle.X_train, bundle)
    info["activity"]["output_spike_rate"] = metrics.spike_rate(out_te["spikes"])

    res = {
        "method": "rstdp", "task_type": "classification",
        "train_accuracy": acc_tr, "val_accuracy": acc_va, "test_accuracy": acc_te,
        "accuracy": acc_te, "confusion_matrix": cm.tolist(),
        "activity": info["activity"], "structure": info["structure"],
        "timing": {"train_time_s": t_train.seconds,
                   "inference_time_per_sample_s": inf_per_sample,
                   "epochs": epochs},
        "reward_curve": reward_curve, "train_acc_curve": acc_curve,
        "rstdp_config": {k: getattr(readout.cfg, k) for k in
                         ["lr", "eligibility_mode", "reward_mode", "elig_decay",
                          "pre_decay", "post_decay", "reward_scale", "neg_reward_scale",
                          "homeo_beta", "w_min", "w_max"]},
        "quantization": {"enabled": model.qcfg.enabled if model.qcfg else False,
                         "bits": model.qcfg.bits if model.qcfg else None},
    }
    # out-of-distribution generalization splits (e.g., IQ jamming unseen SNR/JNR)
    for split, (Xo, yo) in bundle.extra.items():
        pre_o = precompute_readout_pre(model, Xo, device)
        _, acc_o = evprd(pre_o, yo.to(device))
        res[f"{split}_accuracy"] = acc_o

    # --- plots ---
    mdir = ev.method_dir(cfg, "rstdp")
    W_after = to_numpy(readout.weights())
    ev.plot_reward_and_acc(reward_curve, acc_curve, os.path.join(mdir, "reward_curve.png"))
    ev.plot_weight_hist(W_before, W_after, os.path.join(mdir, "weight_hist.png"))
    classes = bundle.meta.get("class_names") or list(range(bundle.n_classes))
    ev.plot_confusion(cm, classes, os.path.join(mdir, "confusion_matrix.png"))
    # rasters for one test sample
    fb = model.reservoir_forward(bundle.X_test[:1].to(device))
    ev.plot_raster(to_numpy(fb["res_spikes"][0]), os.path.join(mdir, "raster_reservoir.png"),
                   "reservoir spikes (1 test sample)")
    ev.plot_raster(to_numpy(out_te["spikes"][0]), os.path.join(mdir, "raster_output.png"),
                   "output spikes (1 test sample)")
    np.savez(os.path.join(mdir, "weights.npz"), before=W_before, after=W_after)
    return res


# --------------------------------------------------------------------------- #
def run_regression(cfg, bundle, model, device) -> Dict:
    w0 = bundle.washout
    Str = model.regression_states(bundle.X_train.to(device))[w0:]
    Sva = model.regression_states(bundle.X_val.to(device))[w0:]
    Ste = model.regression_states(bundle.X_test.to(device))[w0:]
    ytr = bundle.y_train[0][w0:].to(device)
    yte = bundle.y_test[0][w0:].to(device)

    rc = cfg.get("readout", {})
    with Timer() as t_train:
        pred, W = local_delta_readout(Str, ytr, Ste, lr=rc.get("local_lr", 2e-3),
                                      epochs=rc.get("local_epochs", 20))
    pred = pred.cpu()
    yte_c = yte.cpu()
    res_rate = float(model.reservoir_forward(bundle.X_train.to(device))["res_spikes"].mean())

    res = {
        "method": "rstdp", "task_type": "regression",
        "mse": metrics.mse(yte_c, pred), "mae": metrics.mae(yte_c, pred),
        "rmse": metrics.rmse(yte_c, pred), "nrmse": metrics.nrmse(yte_c, pred),
        "activity": {"reservoir_spike_rate": res_rate,
                     "reservoir_sparsity": 1 - res_rate},
        "structure": metrics.count_structure(
            model.reservoir.in_channels, model.reservoir_dim,
            getattr(model.res_cfg, "connectivity", 0.0), 0, bundle.out_dim, ["res_out"]),
        "timing": {"train_time_s": t_train.seconds},
        "note": "regression uses the LOCAL error-modulated (three-factor / delta) "
                "readout -- no BPTT. The closed-form ridge readout is the linear baseline.",
        "quantization": {"enabled": model.qcfg.enabled if model.qcfg else False,
                         "bits": model.qcfg.bits if model.qcfg else None},
    }
    mdir = ev.method_dir(cfg, "rstdp")
    ev.plot_prediction(yte_c, pred, os.path.join(mdir, "prediction.png"))
    fb = model.reservoir_forward(bundle.X_test.to(device))
    ev.plot_raster(to_numpy(fb["res_spikes"][0][:300]), os.path.join(mdir, "raster_reservoir.png"),
                   "reservoir spikes")
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
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

    print(f"[train_rstdp] {cfg['name']} | task={bundle.task_type} | device={device} | "
          f"reservoir={model.reservoir_dim} | quant={'on' if qcfg.enabled else 'off'}")

    if bundle.task_type == "classification":
        res = run_classification(cfg, bundle, model, device)
        print(f"  R-STDP test accuracy = {res['test_accuracy']:.4f} "
              f"(train {res['train_accuracy']:.3f} / val {res['val_accuracy']:.3f})")
        print(f"  reservoir rate={res['activity']['reservoir_spike_rate']:.3f} "
              f"output rate={res['activity']['output_spike_rate']:.3f}")
    else:
        res = run_regression(cfg, bundle, model, device)
        print(f"  R-STDP(local) test NRMSE = {res['nrmse']:.4f}  MSE = {res['mse']:.5f}")

    mdir = ev.method_dir(cfg, "rstdp")
    save_json(res, os.path.join(mdir, "metrics.json"))
    print(f"  saved -> {mdir}/metrics.json")


if __name__ == "__main__":
    main()
