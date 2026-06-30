import torch

from src.neurons import LIFLayer, LIFParams, heaviside_spike, lif_step, surrogate_spike


def test_suprathreshold_current_spikes():
    p = LIFParams(leak=0.9, v_threshold=1.0)
    layer = LIFLayer(4, p)
    spikes = layer(torch.ones(2, 20, 4) * 0.5)
    assert spikes.sum() > 0


def test_zero_input_no_spikes():
    p = LIFParams(leak=0.9, v_threshold=1.0)
    layer = LIFLayer(4, p)
    assert layer(torch.zeros(2, 20, 4)).sum() == 0


def test_leak_decays_membrane():
    # one input pulse then silence -> membrane decays toward 0
    p = LIFParams(leak=0.5, v_threshold=10.0)  # high threshold: no spike/reset
    v = torch.tensor([[1.0]])
    v, s, _ = lif_step(v, torch.tensor([[0.0]]), p)
    assert torch.isclose(v, torch.tensor([[0.5]]))
    v, s, _ = lif_step(v, torch.tensor([[0.0]]), p)
    assert torch.isclose(v, torch.tensor([[0.25]]))


def test_subtractive_reset():
    p = LIFParams(leak=1.0, v_threshold=1.0, reset="subtract")
    v = torch.tensor([[0.6]])
    v, s, _ = lif_step(v, torch.tensor([[0.6]]), p)  # v=1.2 -> spike -> 1.2-1.0=0.2
    assert s.item() == 1.0
    assert torch.isclose(v, torch.tensor([[0.2]]), atol=1e-6)


def test_refractory_silences_neuron():
    p = LIFParams(leak=1.0, v_threshold=1.0, refractory=3)
    v = torch.zeros(1, 1)
    ref = torch.zeros(1, 1, dtype=torch.long)
    spikes = []
    for _ in range(6):
        v, s, ref = lif_step(v, torch.ones(1, 1) * 2.0, p, ref)
        spikes.append(s.item())
    # cannot spike every step due to refractory period
    assert sum(spikes) < 6


def test_surrogate_matches_heaviside_forward_and_has_gradient():
    v = torch.linspace(-1, 2, 50, requires_grad=True)
    s = surrogate_spike(v, 1.0, 10.0)
    assert torch.equal(s.detach(), heaviside_spike(v.detach(), 1.0))
    s.sum().backward()
    assert v.grad.abs().sum() > 0          # surrogate provides nonzero gradient
