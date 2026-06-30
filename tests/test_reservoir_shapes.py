import torch

from src.reservoirs import ReservoirConfig, build_reservoir

DEVICE = torch.device("cpu")


def test_lif_reservoir_shapes_and_binary():
    cfg = ReservoirConfig(kind="lif", n_reservoir=50, connectivity=0.2, spectral_radius=0.9)
    res = build_reservoir(in_channels=4, cfg=cfg, device=DEVICE, seed=0)
    spikes_in = (torch.rand(3, 25, 4) < 0.3).float()
    out = res.run(spikes_in)
    assert out["spikes"].shape == (3, 25, 50)
    assert out["traces"].shape == (3, 25, 50)
    assert set(torch.unique(out["spikes"]).tolist()) <= {0.0, 1.0}
    assert res.out_channels() == 50


def test_spectral_radius_scaling():
    cfg = ReservoirConfig(kind="lif", n_reservoir=80, connectivity=0.3, spectral_radius=0.7)
    res = build_reservoir(in_channels=2, cfg=cfg, device=DEVICE, seed=1)
    assert abs(res.spectral_radius() - 0.7) < 0.05


def test_reservoir_is_fixed_after_run():
    cfg = ReservoirConfig(kind="lif", n_reservoir=40, connectivity=0.2, spectral_radius=0.9)
    res = build_reservoir(in_channels=3, cfg=cfg, device=DEVICE, seed=2)
    w_before = res.W_rec.clone()
    res.run((torch.rand(2, 15, 3) < 0.3).float())
    assert torch.equal(res.W_rec, w_before)         # no learning inside the reservoir


def test_excitatory_inhibitory_ratio():
    cfg = ReservoirConfig(kind="lif", n_reservoir=100, connectivity=0.2,
                          spectral_radius=0.9, exc_ratio=0.8)
    res = build_reservoir(in_channels=2, cfg=cfg, device=DEVICE, seed=3)
    frac_exc = res.exc_mask.float().mean().item()
    assert abs(frac_exc - 0.8) < 1e-6


def test_ldn_reservoir_runs_and_encodes_to_spikes():
    cfg = ReservoirConfig(kind="ldn", ldn_order=6, ldn_theta=20.0)
    res = build_reservoir(in_channels=2, cfg=cfg, device=DEVICE, seed=0)
    out = res.run(torch.randn(3, 30, 2))
    assert out["spikes"].shape[0] == 3 and out["spikes"].shape[1] == 30
    assert set(torch.unique(out["spikes"]).tolist()) <= {0.0, 1.0}
    # posneg encoder over 2*order states -> 2 * (2*6) = 24 channels
    assert out["spikes"].shape[2] == res.out_channels()
