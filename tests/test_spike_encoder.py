import torch

from src.encoders import build_encoder
from src.utils import set_seed

DEVICE = torch.device("cpu")


def _signal(B=3, T=40, C=2):
    set_seed(0)
    return torch.randn(B, T, C)


def test_spiking_encoders_are_binary_and_keep_shape():
    x = _signal()
    for name in ["rate", "temporal_diff", "threshold"]:
        s = build_encoder({"name": name}).encode(x)
        assert s.shape == x.shape
        assert set(torch.unique(s).tolist()) <= {0.0, 1.0}


def test_posneg_doubles_channels_and_is_binary():
    x = _signal(C=2)
    enc = build_encoder({"name": "posneg", "threshold": 0.2})
    assert enc.out_channels(2) == 4
    s = enc.encode(x)
    assert s.shape == (x.shape[0], x.shape[1], 4)
    assert set(torch.unique(s).tolist()) <= {0.0, 1.0}


def test_posneg_on_off_semantics():
    # strictly increasing -> only ON channel fires; decreasing -> only OFF
    T = 30
    up = torch.linspace(0, 5, T).reshape(1, T, 1)
    enc = build_encoder({"name": "posneg", "threshold": 0.2})
    s = enc.encode(up)               # channels [ON, OFF]
    on, off = s[..., 0], s[..., 1]
    assert on.sum() > 0 and off.sum() == 0
    down = torch.linspace(5, 0, T).reshape(1, T, 1)
    s2 = enc.encode(down)
    assert s2[..., 1].sum() > 0 and s2[..., 0].sum() == 0


def test_rate_encoder_is_deterministic_with_generator():
    x = _signal()
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    s1 = build_encoder({"name": "rate"}, generator=g1).encode(x)
    s2 = build_encoder({"name": "rate"}, generator=g2).encode(x)
    assert torch.equal(s1, s2)


def test_rate_encoder_monotonic_in_value():
    # higher constant value -> higher firing probability
    set_seed(1)
    low = torch.zeros(1, 2000, 1)
    high = torch.ones(1, 2000, 1)
    x = torch.cat([low, high], dim=1)        # min-max maps low->0, high->1
    s = build_encoder({"name": "rate"}).encode(x)
    assert s[:, :2000].mean() < s[:, 2000:].mean()


def test_identity_encoder_passthrough():
    x = _signal()
    enc = build_encoder({"name": "identity", "scale": 2.0})
    assert enc.out_channels(2) == 2
    assert torch.allclose(enc.encode(x), x * 2.0)
