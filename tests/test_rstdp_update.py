import torch

from src.rstdp import RSTDPConfig, RSTDPLayer
from src.utils import QuantConfig

DEVICE = torch.device("cpu")


def _layer(**kw):
    cfg = RSTDPConfig(**kw)
    return RSTDPLayer(n_pre=5, n_post=3, cfg=cfg, device=DEVICE, seed=0)


def test_eligibility_accumulates_with_activity():
    layer = _layer(eligibility_mode="pre")
    pre = torch.ones(4, 12, 5)                 # all pre-neurons active
    layer.forward(pre, track_eligibility=True)
    assert layer._elig is not None
    assert layer._elig.abs().sum() > 0


def test_positive_reward_potentiates_target_column():
    layer = _layer(eligibility_mode="pre", lr=0.5, w_min=-5, w_max=5)
    pre = torch.ones(4, 12, 5)
    w_before = layer.W.clone()
    layer.forward(pre, track_eligibility=True)
    reward = torch.zeros(4, 3)
    reward[:, 0] = 1.0                         # reward only output neuron 0
    layer.apply_reward(reward)
    assert (layer.W[:, 0] > w_before[:, 0]).all()        # column 0 potentiated
    assert torch.allclose(layer.W[:, 1], w_before[:, 1])  # others unchanged


def test_negative_reward_depresses_column():
    layer = _layer(eligibility_mode="pre", lr=0.5, w_min=-5, w_max=5)
    pre = torch.ones(4, 12, 5)
    w_before = layer.W.clone()
    layer.forward(pre, track_eligibility=True)
    reward = torch.zeros(4, 3)
    reward[:, 2] = -1.0
    layer.apply_reward(reward)
    assert (layer.W[:, 2] < w_before[:, 2]).all()


def test_weight_clipping_is_respected():
    layer = _layer(eligibility_mode="pre", lr=100.0, w_min=-0.5, w_max=0.5)
    pre = torch.ones(4, 12, 5)
    layer.forward(pre, track_eligibility=True)
    reward = torch.ones(4, 3)
    layer.apply_reward(reward)
    assert layer.W.max() <= 0.5 + 1e-6 and layer.W.min() >= -0.5 - 1e-6


def test_quantization_limits_distinct_weights():
    qcfg = QuantConfig(enabled=True, bits=4, weights=True)
    cfg = RSTDPConfig(eligibility_mode="pre", lr=0.5, w_min=-1, w_max=1)
    layer = RSTDPLayer(5, 3, cfg, DEVICE, seed=0, qcfg=qcfg)
    pre = torch.ones(4, 10, 5)
    layer.forward(pre, track_eligibility=True)
    layer.apply_reward(torch.ones(4, 3))
    # 4-bit -> at most 2**4 distinct levels
    assert len(torch.unique(layer.W)) <= 16


def test_reward_from_counts_class_specific():
    cfg = RSTDPConfig(reward_mode="class_specific", reward_scale=1.0, neg_reward_scale=1.0)
    layer = RSTDPLayer(5, 3, cfg, DEVICE, seed=0)
    counts = torch.tensor([[5.0, 1.0, 0.0],     # predicts 0
                           [0.0, 0.0, 9.0]])     # predicts 2
    targets = torch.tensor([0, 1])               # sample0 correct, sample1 wrong
    r = layer.reward_from_counts(counts, targets)
    assert r[0, 0] > 0                            # correct target reinforced
    assert r[1, 1] > 0 and r[1, 2] < 0           # reinforce target, suppress wrong pred
