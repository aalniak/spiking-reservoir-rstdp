"""Reward-modulated STDP (R-STDP / three-factor) learning for spiking readouts.

This is the **main learning rule** of the project. It is *local* and
*gradient-free*: a synapse's weight change depends only on
(1) its own pre-synaptic spike trace, (2) its post-synaptic spike trace, and
(3) a scalar/vector reward (the "third factor"). No backpropagation, no BPTT --
which is exactly what makes it a candidate for on-chip Loihi-2 / Lava learning.

Per-synapse computation each timestep (pair-based STDP into an eligibility trace)::

    x_pre  <- pre_decay  * x_pre  + pre_spike            # pre trace
    x_post <- post_decay * x_post + post_spike           # post trace
    dE     =  A_plus  * outer(x_pre, post_spike)         # LTP (pre-before-post)
           -  A_minus * outer(pre_spike, x_post)         # LTD (post-before-pre)
    E      <- elig_decay * E + dE                        # eligibility trace

At the end of a sequence the eligibility is gated by a reward to produce the
weight update (the three-factor product)::

    dW = lr * mean_batch( reward[:, None, :] * E )  +  homeostasis
    W  <- clip( W + dW )   (optionally quantized)

The reward for classification is derived from output spike counts (predicted =
argmax), with an optional class-specific form that reinforces the correct output
neuron and suppresses the wrongly-predicted one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .neurons import LIFParams, lif_step
from .utils import QuantConfig, maybe_quantize


@dataclass
class RSTDPConfig:
    # learning
    lr: float = 0.05
    pre_decay: float = 0.9
    post_decay: float = 0.9
    elig_decay: float = 0.97
    a_plus: float = 1.0
    a_minus: float = 0.5
    # how the eligibility trace is formed from pre/post activity:
    #   'stdp'   : pair-based pre/post spike-timing (most on-chip / Loihi faithful)
    #   'hybrid' : product of graded pre and post traces (+ STDP depression)
    #   'pre'    : graded pre-trace only -> reward-modulated delta rule (best acc)
    eligibility_mode: str = "pre"
    # reward
    reward_scale: float = 1.0
    neg_reward_scale: float = 1.0          # set 0 for positive-only reward
    # 'error'          : per-output three-factor error signal (target - softmax(counts)).
    #                    Local delta-rule flavour; best classification accuracy.
    # 'class_specific' : reinforce correct neuron, suppress wrongly-predicted one.
    # 'global'         : single scalar reward (+/-) broadcast by target sign.
    reward_mode: str = "error"
    # weights
    w_min: float = -3.0
    w_max: float = 3.0
    w_init_scale: float = 0.05
    # homeostasis
    homeo_beta: float = 0.0                # 0 disables homeostatic regularization
    target_rate: float = 0.2               # desired post spikes / neuron / step
    # post-neuron dynamics (leak=1.0 -> non-leaky integrator; spike count is a
    # clean linear function of weighted input, which the readout argmax relies on)
    leak: float = 1.0
    v_threshold: float = 1.0
    # readout input current scaling. If None, auto = 1/sqrt(n_pre) so the output
    # neurons stay in a graded (non-saturating) regime regardless of reservoir
    # size -- essential for spike-count argmax to carry class margin.
    current_scale: Optional[float] = None
    # supervised "teacher" current pushed into the target neuron during training.
    # NOTE: with reward_mode='error' the teacher inflates the target's spike
    # count and cancels the error signal, so it defaults OFF. It is useful with
    # reward_mode='class_specific' (classic supervised R-STDP).
    teacher_strength: float = 0.0

    @classmethod
    def from_dict(cls, d) -> "RSTDPConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RSTDPLayer:
    """A plastic projection (pre population -> LIF post population) trained by R-STDP.

    Weights live here; neuron dynamics are LIF. The layer keeps the eligibility
    trace produced by the most recent forward pass so that :meth:`apply_reward`
    can turn it into a weight update.
    """

    def __init__(self, n_pre: int, n_post: int, cfg: RSTDPConfig, device,
                 seed: int = 0, qcfg: Optional[QuantConfig] = None):
        self.n_pre = n_pre
        self.n_post = n_post
        self.cfg = cfg
        self.device = device
        self.qcfg = qcfg
        self.lif = LIFParams(leak=cfg.leak, v_threshold=cfg.v_threshold)

        # auto scale keeps output neurons graded (~O(1) input current) regardless
        # of reservoir size; 3/n_pre matches the tuned regime (configs may override).
        self.current_scale = (cfg.current_scale if cfg.current_scale is not None
                              else 3.0 / n_pre)

        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.rand(n_pre, n_post, generator=g) * cfg.w_init_scale
        self.W = W.to(device)
        self._init_W = self.W.clone()

        # filled by forward()
        self._elig: Optional[torch.Tensor] = None
        self._post_rate: Optional[torch.Tensor] = None
        self._pre_mean: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def num_synapses(self) -> int:
        return self.n_pre * self.n_post

    def weights(self) -> torch.Tensor:
        return self.W

    def initial_weights(self) -> torch.Tensor:
        return self._init_W

    def _Wq(self) -> torch.Tensor:
        return maybe_quantize(self.W, self.qcfg, "weights")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, pre_spikes: torch.Tensor, teacher: Optional[torch.Tensor] = None,
                track_eligibility: bool = True) -> Dict[str, torch.Tensor]:
        """Run the post LIF neurons over time and accumulate eligibility.

        Parameters
        ----------
        pre_spikes : [B, T, n_pre] reservoir (or hidden) spikes.
        teacher : optional [B, n_post] one-hot target. When given, a positive
            ``teacher_strength`` current is injected into the target neuron each
            step (supervised R-STDP) so eligibility forms for the correct class.
        track_eligibility : disable during pure inference to save compute.

        Returns dict(spikes=[B,T,n_post], counts=[B,n_post]).
        """
        B, T, n_pre = pre_spikes.shape
        assert n_pre == self.n_pre
        dev = self.device
        pre = pre_spikes.to(dev)
        W = self._Wq()

        v = torch.zeros(B, self.n_post, device=dev)
        x_pre = torch.zeros(B, self.n_pre, device=dev)
        x_post = torch.zeros(B, self.n_post, device=dev)
        E = torch.zeros(B, self.n_pre, self.n_post, device=dev) if track_eligibility else None

        counts = torch.zeros(B, self.n_post, device=dev)
        out = torch.empty(B, T, self.n_post, device=dev)
        teach_cur = None
        if teacher is not None:
            teach_cur = teacher.to(dev) * self.cfg.teacher_strength

        c = self.cfg
        for t in range(T):
            pre_t = pre[:, t]
            current = self.current_scale * (pre_t @ W)
            if teach_cur is not None:
                current = current + teach_cur
            v = maybe_quantize(v, self.qcfg, "states")
            v, post_t, _ = lif_step(v, current, self.lif, surrogate=False)

            # update traces
            x_pre = c.pre_decay * x_pre + pre_t
            x_post_old = x_post
            x_post = c.post_decay * x_post + post_t
            x_pre = maybe_quantize(x_pre, self.qcfg, "traces")
            x_post = maybe_quantize(x_post, self.qcfg, "traces")

            if track_eligibility:
                if c.eligibility_mode == "pre":
                    # graded pre-trace only -> reward-modulated delta rule
                    E = c.elig_decay * E + x_pre.unsqueeze(2)
                elif c.eligibility_mode == "hybrid":
                    dE_ltp = c.a_plus * torch.einsum("bi,bj->bij", x_pre, x_post)
                    dE_ltd = c.a_minus * torch.einsum("bi,bj->bij", pre_t, x_post_old)
                    E = c.elig_decay * E + (dE_ltp - dE_ltd)
                else:  # 'stdp' -- pair-based spike-timing eligibility
                    # LTP: pre-before-post; LTD: post-before-pre
                    dE_ltp = c.a_plus * torch.einsum("bi,bj->bij", x_pre, post_t)
                    dE_ltd = c.a_minus * torch.einsum("bi,bj->bij", pre_t, x_post_old)
                    E = c.elig_decay * E + (dE_ltp - dE_ltd)

            counts += post_t
            out[:, t] = post_t

        if track_eligibility:
            self._elig = E
            self._post_rate = counts.mean(dim=0) / T            # [n_post]
            self._pre_mean = pre.mean(dim=(0, 1))               # [n_pre]
        return {"spikes": out, "counts": counts}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reward_from_counts(self, counts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Build the per-output reward (modulator) vector from spike counts.

        Returns ``reward`` of shape [B, n_post].
        """
        B = counts.shape[0]
        dev = counts.device
        targets = targets.to(dev)
        pred = counts.argmax(dim=1)
        correct = (pred == targets)
        c = self.cfg
        reward = torch.zeros(B, self.n_post, device=dev)
        idx = torch.arange(B, device=dev)
        onehot = torch.zeros(B, self.n_post, device=dev)
        onehot[idx, targets] = 1.0

        if c.reward_mode == "error":
            # per-output error: (target - softmax(standardized counts)). This is a
            # local three-factor delta rule -- correct neuron gets a positive
            # modulator, others negative in proportion to their (wrong) confidence.
            z = counts - counts.mean(dim=1, keepdim=True)
            z = z / (counts.std(dim=1, keepdim=True) + 1e-6)
            p = torch.softmax(z, dim=1)
            return c.reward_scale * (onehot - p)

        if c.reward_mode == "global":
            r = torch.where(correct, torch.full_like(pred, 1, dtype=torch.float) * c.reward_scale,
                            torch.full_like(pred, 1, dtype=torch.float) * (-c.neg_reward_scale))
            # push correct class up, others down, scaled by global correctness sign
            onehot = torch.zeros(B, self.n_post, device=dev)
            onehot[torch.arange(B), targets.to(dev)] = 1.0
            reward = r[:, None] * (2 * onehot - 1)
            return reward

        # class-specific (default): reinforce target, suppress wrong prediction
        idx = torch.arange(B, device=dev)
        reward[idx, targets.to(dev)] = c.reward_scale
        wrong = ~correct
        if wrong.any():
            reward[idx[wrong], pred[wrong]] = -c.neg_reward_scale
        return reward

    @torch.no_grad()
    def apply_reward(self, reward: torch.Tensor) -> None:
        """Three-factor update: gate eligibility by reward, add homeostasis, clip."""
        if self._elig is None:
            raise RuntimeError("call forward(track_eligibility=True) before apply_reward")
        c = self.cfg
        # normalize eligibility scale so lr is comparable across eligibility modes
        # / sequence lengths and weights don't slam into the clip bounds.
        scale = self._elig.abs().mean() + 1e-8
        # dW[i,j] = lr * mean_b reward[b,j] * E[b,i,j]
        dW = (reward[:, None, :] * self._elig).mean(dim=0) / scale
        self.W = self.W + c.lr * dW

        # homeostatic regularization: pull post firing rate toward target_rate
        if c.homeo_beta > 0 and self._post_rate is not None:
            dev = self.W.device
            dev_term = (self._post_rate - c.target_rate)            # [n_post]
            homeo = -c.homeo_beta * self._pre_mean[:, None] * dev_term[None, :]
            self.W = self.W + homeo

        self.W = self.W.clamp(c.w_min, c.w_max)
        self.W = maybe_quantize(self.W, self.qcfg, "weights")

    # convenience: full supervised training step on one batch
    @torch.no_grad()
    def train_step(self, pre_spikes: torch.Tensor, targets: torch.Tensor,
                   teacher_forcing: bool = False) -> Dict[str, float]:
        teacher = None
        if teacher_forcing:
            teacher = torch.zeros(pre_spikes.shape[0], self.n_post, device=self.device)
            teacher[torch.arange(pre_spikes.shape[0]), targets.to(self.device)] = 1.0
        res = self.forward(pre_spikes, teacher=teacher, track_eligibility=True)
        reward = self.reward_from_counts(res["counts"], targets)
        self.apply_reward(reward)
        tgt = targets.to(self.device)
        pred = res["counts"].argmax(dim=1)
        acc = float((pred == tgt).float().mean().item())
        # signed reward delivered to the *correct* output neuron -- a meaningful
        # scalar for the reward/learning curve (rises toward 0 as confidence grows)
        reward_correct = float(reward[torch.arange(len(tgt), device=self.device), tgt].mean().item())
        return {"acc": acc, "mean_reward": float(reward.abs().sum(dim=1).mean().item()),
                "reward_correct": reward_correct,
                "out_rate": float(res["spikes"].mean().item())}

    @torch.no_grad()
    def predict(self, pre_spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        res = self.forward(pre_spikes, teacher=None, track_eligibility=False)
        res["pred"] = res["counts"].argmax(dim=1)
        return res
