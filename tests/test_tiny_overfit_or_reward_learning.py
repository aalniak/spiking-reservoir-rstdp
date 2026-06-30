"""End-to-end sanity check: the R-STDP readout actually *learns* (reward and
accuracy increase) on a tiny, easily separable 2-class spiking task."""
import torch

from src.encoders import build_encoder
from src.model import SpikingReservoirModel
from src.utils import set_seed

DEVICE = torch.device("cpu")


def _tiny_two_class(B=60, T=50):
    set_seed(0)
    t = torch.linspace(0, 12.0, T)
    X, y = [], []
    for i in range(B):
        c = i % 2
        f = 1.0 if c == 0 else 2.0
        ph = torch.rand(1) * 6.28
        sig = torch.stack([torch.sin(f * t + ph), torch.cos(f * t + ph)], dim=-1)
        X.append(sig + 0.05 * torch.randn(T, 2))
        y.append(c)
    return torch.stack(X), torch.tensor(y)


def test_rstdp_learns_tiny_task():
    X, y = _tiny_two_class()
    cfg = {
        "encoder": {"name": "posneg", "threshold": 0.2},
        "reservoir": {"kind": "lif", "n_reservoir": 100, "connectivity": 0.15,
                      "spectral_radius": 0.9, "input_scaling": 2.0, "leak": 0.9},
        "readout": {"lr": 0.05, "eligibility_mode": "pre", "reward_mode": "error",
                    "current_scale": 0.04, "leak": 1.0},
    }
    model = SpikingReservoirModel(
        {"task_type": "classification", "in_channels": 2, "out_dim": 2},
        cfg, DEVICE, seed=1)
    pre = model.reservoir_forward(X)["readout_pre"]

    acc0 = float((model.readout.predict(pre)["pred"] == y).float().mean())
    rewards = []
    for _ in range(60):
        perm = torch.randperm(X.shape[0])
        info = model.readout.train_step(pre[perm], y[perm])
        rewards.append(info["reward_correct"])
    accf = float((model.readout.predict(pre)["pred"] == y).float().mean())

    assert accf > acc0                      # learning improved accuracy
    assert accf > 0.85                      # tiny separable task is essentially solved
    # reward_correct = (1 - p_target): its magnitude SHRINKS toward 0 as R-STDP
    # makes the correct neuron win more confidently -> evidence of convergence.
    assert sum(rewards[-10:]) / 10 < sum(rewards[:10]) / 10


def test_reservoir_features_are_separable_baseline():
    # the reservoir representation should be (near-)linearly separable -> the
    # R-STDP ceiling is high. Quick logistic check.
    from sklearn.linear_model import LogisticRegression
    X, y = _tiny_two_class()
    cfg = {"encoder": {"name": "posneg", "threshold": 0.2},
           "reservoir": {"kind": "lif", "n_reservoir": 100, "connectivity": 0.15,
                         "spectral_radius": 0.9, "input_scaling": 2.0, "leak": 0.9}}
    model = SpikingReservoirModel(
        {"task_type": "classification", "in_channels": 2, "out_dim": 2},
        cfg, DEVICE, seed=1)
    F, _ = model.classification_features(X)
    clf = LogisticRegression(max_iter=500).fit(F, y.numpy())
    assert clf.score(F, y.numpy()) > 0.9
