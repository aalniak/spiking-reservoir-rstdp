# Reservoir-based Spiking Time-Series Processing with R-STDP Local Learning
### Technical report — CPU/GPU prototype

> **This is a CPU/GPU prototype for reservoir-based spiking time-series processing
> and R-STDP learning. It is designed with Loihi-2/Lava deployment in mind, but it
> is not claimed as an actual Loihi-2 implementation unless run on the Loihi-2
> backend.**

---

## 1. Motivation

Training spiking neural networks (SNNs) usually relies on **surrogate-gradient
backpropagation-through-time (BPTT)**, which is powerful but fundamentally
*non-local*: it needs the unrolled forward graph and a reverse-time pass. That is
a poor match for neuromorphic hardware such as Intel **Loihi-2**, where computation
is event-driven, memory is local to each synapse, and on-chip plasticity is
expressed through *local* trace-based learning rules plus a broadcast reward.

This project explores an alternative aimed squarely at on-chip compatibility:

1. a **fixed spiking reservoir** transforms a temporal signal into a rich,
   high-dimensional spike-based state (no training in the recurrent core), and
2. a **single plastic readout** is trained by **R-STDP / three-factor learning** —
   a local, forward-only, eligibility-trace rule modulated by reward/error.

We benchmark this against the classical linear reservoir-computing readout, an
off-chip surrogate-gradient SNN, and a GRU reference, and we quantify spike
activity, parameter counts and quantization robustness — the quantities that
actually matter for an eventual Loihi-2 port.

## 2. Architecture

```
signal [B,T,C] → spike encoder → fixed spiking reservoir → (optional fixed
                 hidden spiking layer) → trainable readout → output spikes
```

* **Encoders** (`src/encoders.py`): rate/Poisson, temporal-difference,
  threshold-crossing, **positive/negative event** (separate ON/OFF channels → 2·C),
  and analog current injection. Event coding suits delta-like signals; current
  injection suits amplitude-coded tasks (NARMA10).
* **Reservoir** (`src/reservoirs.py`): a random recurrent **LIF / Liquid-State-
  Machine** core — sparse fixed recurrent weights, configurable size, spectral
  radius, input scaling, leak and excitatory/inhibitory ratio (Dale's law). An
  optional **Legendre Delay Network (LDN)** reservoir provides a fixed
  Legendre-polynomial temporal memory, spike-encoded before readout.
* **Readout** (`src/rstdp.py`, `src/model.py`): for classification, a spiking
  layer whose **spike counts** give an argmax class decision; for prediction, a
  multi-timescale-trace readout fitted either by closed-form **ridge** (linear
  baseline) or a **local error-modulated delta rule** (the R-STDP analogue).

Only the readout learns. The reservoir is fixed. Spike communication is explicit
and binary throughout.

## 3. Datasets

| Dataset | Type | Shape (T×C) | Notes |
|---|---|---|---|
| Mackey-Glass | regression | long series ×1 | chaotic, τ=17, one-step prediction |
| NARMA10 | regression | long series ×1 | 10th-order nonlinear memory task |
| BasicMotions | classification (4) | 100×6 | UEA, sanity check (via `aeon`) |
| CharacterTrajectories | classification (20) | ~182×3 | UEA, main public set (via `aeon`) |
| Synthetic IQ jamming | classification (2) | 128×2 | BPSK/QPSK/16QAM + AWGN + tone/broadband/pulsed/swept jammers; **unseen-SNR/JNR OOD split** |

All tensors are `[batch, time, channels]`; normalization uses **training
statistics only**; seeds are fixed.

## 4. The R-STDP learning rule

Each plastic readout synapse `i→j` maintains a pre-trace `x_pre`, a post-trace
`x_post` and an **eligibility trace** `E`. Per timestep:

```
x_pre  ← pre_decay · x_pre  + pre_spike
x_post ← post_decay · x_post + post_spike
E      ← elig_decay · E + STDP(x_pre, x_post, spikes)      # local coincidence
```

At the end of a sequence the eligibility is gated by the **third factor** — a
reward/error signal derived from the output spike counts — to produce the update:

```
ΔW(i,j) = lr · reward_j · E(i,j)      then clip, (optional) homeostasis, quantize
```

We provide three eligibility formulations: `stdp` (pair-based spike-timing, the
most hardware-literal), `hybrid` (graded pre×post), and `pre` (graded pre-trace →
a reward-modulated **delta rule**). The reward can be `error`
(`target − softmax(counts)`, our default), `class_specific` (reinforce correct,
suppress wrong), or `global`. Configurable: learning rate, pre/post/eligibility
decays, reward & negative-reward scales, weight clipping, homeostatic
firing-rate regularization, output threshold and membrane leak. **No backprop, no
BPTT** — every quantity in the update is locally available at the synapse plus a
broadcast reward.

## 5. Experiments

Five experiments (configs in `configs/`): Mackey-Glass and NARMA10 prediction,
BasicMotions and CharacterTrajectories classification, and synthetic IQ-jamming
detection with an out-of-distribution (unseen SNR/JNR) generalization split. For
each we compare the **R-STDP** readout against a **linear** (ridge/logistic)
reservoir-computing baseline, an off-chip **surrogate**-gradient SNN, and a **GRU**
reference, and we sweep reservoir size and quantization level. Reproduce with
`bash scripts/run_<experiment>.sh`; sweeps/ablations/figures via
`python scripts/make_report_figures.py`.

## 6. Results

Representative-demo runs on this machine (1×GPU). Numbers are produced by the code
in this repo (`results/<exp>/results_table.md`, `results/SUMMARY.md`).

**Prediction (test NRMSE, lower is better):**

| Method | Mackey-Glass | NARMA10 |
|---|---|---|
| GRU (off-chip BPTT, raw signal) | **0.032** | **0.522** |
| Linear ridge readout (classical RC) | 0.128 | 0.748 |
| **R-STDP local readout (this work)** | 0.140 | 0.851 |

**Classification (test accuracy, higher is better):**

| Method | BasicMotions | IQ-jamming (in-dist) | IQ-jamming (unseen SNR/JNR) |
|---|---|---|---|
| Linear logistic readout (classical RC) | **1.00** | 0.998 | 0.733 |
| Surrogate SNN (off-chip BPTT) | **1.00** | 0.998 | 0.773 |
| GRU (off-chip BPTT) | 0.875 | **1.00** | **0.943** |
| **R-STDP readout (this work)** | 0.75 | 0.80 | 0.625 |

**Spiking efficiency & footprint (R-STDP):** reservoir spike rate ≈ 0.20–0.25
spikes/neuron/step (≈75–80 % sparsity), output rate ≈ 0.12–0.21, and only the
readout is plastic (e.g. 400–1200 trainable synapses vs. tens of thousands of
fixed reservoir synapses).

**Reservoir-size sweep** (`results/figures/mse_vs_reservoir_size.png`): prediction
error decreases as the reservoir grows from 100→1000 (more temporal features),
with diminishing returns — the classic reservoir-computing trend, mirrored by both
the ridge and the local R-STDP readout.

**Quantization** (`results/figures/acc_vs_quant_bits.png`): simulated fixed-point
on weights, membrane states and traces. R-STDP classification accuracy is
**stable at 16-, 12- and 8-bit (≈ full precision)** on both tasks (BasicMotions
0.75 at FP/8/12/16-bit; IQ-jamming 0.80–0.92 across all bit-widths). The **4-bit
stress level is dataset-dependent**: the IQ-jamming reservoir tolerates it (≈0.85),
whereas on BasicMotions the coarse membrane/trace resolution drives the reservoir
into saturation (spike rate jumps from ≈0.22 to ≈0.90) and accuracy collapses to
chance (0.25). Take-away: **R-STDP remains stable under quantization down to 8-bit**;
4-bit is a genuine stress regime whose viability depends on reservoir dynamics.
See `results/SUMMARY.md` for the exact per-bit table.

**Ablations** (`results/SUMMARY.md`, BasicMotions) are revealing:

| Ablation | test acc | note |
|---|---|---|
| encoder = posneg / rate / threshold | 0.75 | comparable |
| encoder = **temporal_diff** | **0.825** | slightly best here |
| reward = **error** (default) | **0.75** | per-output error signal |
| reward = pos-only / class_specific | 0.225 | **collapses** (output saturates) |
| eligibility = **pre** (default) | **0.75** | graded pre-trace |
| eligibility = stdp / hybrid | 0.25 | **collapses** (outputs never fire) |
| homeostasis on; reservoir conn 0.3 | 0.75 | robust |

The two design choices that matter most are the **`error` (delta-rule) third
factor** and the **graded `pre` eligibility**: with the class-specific reward the
outputs saturate, and with the pair-based spike-timing (`stdp`) eligibility the
output neurons never fire enough to form eligibility (no teacher current), so both
collapse to chance. This is an honest sensitivity of local learning — the
pair-STDP timing rule is the most hardware-literal but needs supervised teacher
current to work, whereas the graded three-factor `pre`/`error` rule learns
robustly without it. Encoder choice and reservoir sparsity matter much less.

### Reading of the results

The pattern is consistent and honest: **R-STDP learns every task well above
chance using only a local, forward-only rule, but trails the offline
gradient/closed-form methods.** That gap is precisely the cost of locality — and
the reason on-chip learning is an open research problem. The point of the
prototype is not to beat BPTT but to show a *Loihi-compatible* learner that works,
with explicit spike-activity, parameter-count and quantization accounting.

## 7. Limitations

* R-STDP accuracy lags BPTT/linear readouts (the locality/accuracy trade-off).
* The reward-modulated rule is more sensitive to hyper-parameters (output drive
  scaling, eligibility/learning rates) than gradient training.
* NARMA10 is hard for a *spiking* reservoir (NRMSE ≈ 0.8 vs ≈ 0.5 for a GRU);
  amplitude-coded long-memory tasks stress event-based reservoirs.
* The "reservoir" is fixed and only lightly tuned; we do not search reservoir
  topologies.
* Quantization is *simulated* fixed-point (fake-quant), not a true integer datapath.
* CharacterTrajectories is wired up and ready but, per scope, was left as a
  ready-to-run script rather than part of the headline demo runs.

## 8. What an actual Lava / Loihi-2 deployment would require

`src/lava_export.py` already (a) restricts plasticity to one readout projection,
(b) keeps updates local (pre/post traces + a third factor), (c) avoids BPTT,
(d) communicates with explicit binary spikes, (e) supports weight/state/trace
quantization, and (f) exports a hardware-agnostic component spec. To actually run
on Loihi-2 one would still need to:

1. `pip install lava-nc` and rebuild encoders/reservoir/readout as Lava
   `Process`es (`lava.proc.lif.LIF`, `lava.proc.dense.Dense`);
2. express the three-factor update as a Loihi learning rule (`Loihi2FLearningRule`)
   with pre/post traces (x1/y1) and a graded **reward** channel driving the
   eligibility-gated weight update;
3. deliver the reward/error from an external modulatory `Process`;
4. validate on the Lava **CPU simulation** backend (`Loihi2SimCfg`), then port to
   **physical Loihi-2 / NxSDK** — only at that final step may this be described as
   a Loihi-2 deployment.

Until then, this remains a CPU/GPU prototype that is *architecturally* ready, not a
hardware result.

---

## Summary

I implemented a small reservoir-based spiking time-series processing prototype: a
fixed leaky-integrate-and-fire spiking reservoir (Liquid-State-Machine style, with
an optional Legendre-delay variant) converts temporal inputs into rich spike-based
states, which I evaluated on public temporal datasets (Mackey-Glass and NARMA10
prediction, the UEA BasicMotions/CharacterTrajectories classification sets) and on
a synthetic IQ-jamming detection task with an unseen-SNR/JNR generalization split.
On top of the fixed reservoir I implemented an **R-STDP / three-factor local
learning rule** (pre/post-synaptic traces, an eligibility trace, and a
reward/error modulatory signal) to train the readout, deliberately **avoiding full
BPTT in the main method**; surrogate-gradient SNN, linear reservoir-computing and
GRU baselines are included only for comparison. The prototype reports spike rates,
sparsity, parameter/synapse counts and 4–16-bit quantization robustness, and the
codebase is structured — with a documented component→Lava mapping and spec
exporter in `lava_export.py` — for **future Lava / Loihi-2 on-chip training
experiments**. This is a CPU/GPU prototype and is **not** claimed as an actual
Loihi-2 implementation.
