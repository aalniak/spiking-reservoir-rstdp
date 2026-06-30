# rstpd_spiking_reservoir_timeseries

**Reservoir-based spiking time-series processing + R-STDP (reward-modulated STDP /
three-factor) local learning — a CPU/GPU research prototype designed with future
Lava / Loihi-2 deployment in mind.**

> ⚠️ **Honesty note.** This is a **CPU/GPU prototype**. It is **not** an actual
> Loihi-2 implementation. Nothing here runs on Loihi-2 unless explicitly executed
> on the Lava Loihi-2 backend (which is *not* done here — see
> [`src/lava_export.py`](src/lava_export.py)). The design is *architecturally
> compatible* with an eventual Lava/Loihi-2 port; that is the claim, no more.

---

## 1. What is reservoir computing?

A **reservoir** is a fixed (untrained) recurrent dynamical system that nonlinearly
projects an input sequence into a high-dimensional space of temporal "echoes".
Because the reservoir is recurrent, its state at time *t* depends on the recent
input history — it provides **fading memory**. Only a simple **readout** is
trained to map reservoir states to the desired output. This is the core idea of
Echo State Networks (ESNs) and, for spiking neurons, **Liquid State Machines
(LSMs)**. Training is cheap and stable because the hard, recurrent part is never
trained — you just fit a linear (or here, locally-learned) readout on top.

## 2. Why spiking reservoirs for temporal data?

* **Event-driven & sparse.** Spiking LIF neurons communicate with binary spikes;
  activity is naturally sparse, which is energy-efficient on neuromorphic hardware.
* **Intrinsic temporal dynamics.** Membrane leak + recurrent connectivity give a
  spectrum of time constants — a rich temporal basis for sequence tasks.
* **Hardware match.** A fixed spiking reservoir + a small plastic readout maps
  almost one-to-one onto neuromorphic substrates (Loihi-2 / Lava), where neurons
  and synapses are the native primitives and only a thin learning layer is needed.

## 3. What is R-STDP / three-factor learning?

Spike-Timing-Dependent Plasticity (STDP) changes a synapse based on the *relative
timing* of pre- and post-synaptic spikes (pre-before-post → potentiation,
post-before-pre → depression). On its own STDP is unsupervised. **Reward-modulated
STDP (R-STDP)** adds a **third factor** — a global/per-neuron reward or error
signal — that gates an **eligibility trace** (a slowly-decaying memory of recent
STDP events). The weight update is the product of three local quantities:

```
ΔW(i→j) = learning_rate · reward_j · eligibility(i,j)
eligibility(i,j) = decaying trace of pre/post coincidences (STDP)
```

This is exactly what we implement in [`src/rstdp.py`](src/rstdp.py): each plastic
readout synapse maintains a pre-trace, a post-trace and an eligibility trace, and
is updated by a reward/error third factor. **No backpropagation, no BPTT.**

We expose three eligibility formulations (selectable, compared in the ablations):
`stdp` (pair-based spike-timing, most hardware-literal), `hybrid` (graded
pre×post traces), and `pre` (graded pre-trace only → a reward-modulated **delta
rule**, the best-learning variant and our default for the headline numbers).

## 4. Why is R-STDP more Loihi / on-chip-friendly than BPTT?

| Property | BPTT / surrogate-gradient | **R-STDP (this project)** |
|---|---|---|
| Locality | non-local (gradients flow backward through time & layers) | **local** (pre/post traces + a third factor) |
| Memory | stores the full forward activation/graph | **O(1) traces per synapse** |
| Time | requires reverse-time passes | **forward-only, online** |
| Hardware | needs a separate off-chip trainer | **expressible with Loihi learning rules** |

Backprop-through-time needs the whole unrolled computational graph and a backward
pass — neither exists naturally on a neuromorphic chip. R-STDP needs only locally
available signals (its own traces) plus a broadcast reward, which Loihi-2's
on-chip learning engine (graded reward + trace-based learning rules) is built for.

## 5. What is implemented (and runs here)

* **Encoders** ([`encoders.py`](src/encoders.py)): rate/Poisson, temporal-difference,
  threshold-crossing, **positive/negative event** (separate ON/OFF channels), and
  analog current-injection.
* **Neurons** ([`neurons.py`](src/neurons.py)): LIF with leak/threshold/reset/
  refractory; hard-threshold spikes (reservoir/R-STDP) **and** a surrogate-gradient
  spike (baseline only).
* **Reservoirs** ([`reservoirs.py`](src/reservoirs.py)): (A) random recurrent **LIF
  / LSM** reservoir — sparse fixed recurrent weights, configurable size, spectral
  radius, input scaling, leak, E/I ratio (Dale's law); (B) optional **Legendre
  Delay Network (LDN)** fixed temporal-memory reservoir, spike-encoded.
* **R-STDP** ([`rstdp.py`](src/rstdp.py)): eligibility-trace three-factor learning
  with configurable learning rate, trace/eligibility decays, reward/neg-reward
  scales, weight clipping, homeostatic regularization, output threshold, leak.
* **Model** ([`model.py`](src/model.py)): encoder → reservoir → (optional fixed
  hidden spiking layer) → trainable readout; spike-count argmax classifier; ridge
  & local-delta regression readouts.
* **Training/eval**: [`train_rstdp.py`](src/train_rstdp.py) (main),
  [`train_surrogate_baseline.py`](src/train_surrogate_baseline.py) (surrogate SNN +
  GRU/LSTM), [`evaluate.py`](src/evaluate.py) (linear baseline + aggregation + plots).
* **Quantization** ([`utils.py`](src/utils.py)): simulated fixed-point for weights,
  membrane states and traces at 4/8/12/16-bit.
* **Datasets** ([`datasets.py`](src/datasets.py)): Mackey-Glass, NARMA10, synthetic
  IQ-jamming (BPSK/QPSK/16QAM + AWGN + tone/broadband/pulsed/swept jammers, with an
  unseen-SNR/JNR generalization split), and UEA loaders (BasicMotions,
  CharacterTrajectories, …) via `aeon`.

## 6. What is Loihi-2-oriented but **not** actually deployed

* [`src/lava_export.py`](src/lava_export.py) defines a **clean abstraction +
  documented component→Lava/Loihi-2 mapping + a hardware-agnostic spec exporter**,
  and auto-detects whether `lava` is importable. **Lava is not installed here and
  Loihi-2 is not used.** If `lava` were present, a *minimal CPU-backend* skeleton
  becomes available; running on real Loihi-2 silicon would additionally require
  NxSDK/hardware and is explicitly out of scope. The module prints a Loihi-2
  *readiness checklist* describing exactly which constraints the prototype already
  satisfies and what a real port still needs.

## 7. How to run each experiment

```bash
pip install -r requirements.txt
python scripts/download_datasets.py          # caches UEA sets via aeon; verifies synthetic

# Full pipeline per experiment (R-STDP + surrogate/GRU + linear + Lava spec + table):
bash scripts/run_mackey_glass.sh             # Exp 1: Mackey-Glass prediction
bash scripts/run_narma10.sh                  # Exp 2: NARMA10 prediction
bash scripts/run_basicmotions.sh             # Exp 3: BasicMotions classification (sanity)
bash scripts/run_charactertrajectories.sh    # Exp 4: CharacterTrajectories classification
bash scripts/run_iq_jamming.sh               # Exp 5: synthetic IQ jamming detection

# Individual steps (any config), with dotted-key overrides:
python -m src.train_rstdp --config configs/basicmotions.yaml reservoir.n_reservoir=500
python -m src.train_rstdp --config configs/basicmotions.yaml quantization.enabled=true quantization.bits=8
python -m src.evaluate    --config configs/basicmotions.yaml          # linear baseline + aggregate

# Sweeps & ablations:
bash scripts/sweep_reservoir_size.sh configs/mackey_glass.yaml "100 300 500 1000"
bash scripts/sweep_rstdp_params.sh  configs/basicmotions.yaml

# Tests:
python -m pytest tests/ -q
```

Results for `<exp>` land in `results/<exp>/<method>/metrics.json` (+ plots), with a
combined `results/<exp>/results_table.md`.

## 8. How to interpret the results

* **Classification**: compare `test_accuracy` of **rstdp** (local, on-chip-style)
  against **linear** (closed-form RC), **surrogate** (off-chip BPTT) and **gru**
  (DL reference). The expected, honest story is that R-STDP **learns well above
  chance** but **trails** the offline gradient/closed-form methods — the
  **accuracy-vs-locality trade-off** that motivates on-chip learning research.
* **Prediction**: compare `nrmse`/`mse` of the **rstdp** local readout vs the
  **linear** ridge readout vs the **gru** reference. NRMSE ≈ 1.0 means "no better
  than predicting the mean".
* **Spiking activity**: `reservoir_spike_rate` (and output rate) report mean spikes
  per neuron per step; `reservoir_sparsity = 1 − rate`. Lower rates = cheaper.
* **Quantization**: R-STDP stays stable down to **8-bit**; the **4-bit** stress
  test typically collapses (reported as a finding).
* **Structure**: `n_trainable_synapses` shows how few weights actually learn (only
  the readout) — the key Loihi-friendliness metric.

See [`reports/technical_report.md`](reports/technical_report.md) for the full
write-up, results tables, limitations, and the Lava/Loihi-2 deployment gap.

## Repository layout

```
configs/   five YAML experiment configs
src/        datasets, encoders, neurons, reservoirs, rstdp, model,
            train_rstdp, train_surrogate_baseline, evaluate, lava_export, metrics, utils
scripts/    download_datasets.py, run_*.sh, sweep_*.sh
tests/      pytest suite (encoders, LIF, reservoir, R-STDP update, end-to-end learning)
results/    per-experiment metrics.json, plots, results_table.md (generated)
reports/    technical_report.md
```
