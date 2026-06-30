"""Lava / Loihi-2 orientation layer.

This module defines a clean abstraction between the algorithmic prototype in this
repo and an eventual Lava / Loihi-2 implementation. It does **not** run on Loihi-2
and does not require Lava to be installed.

>>> IMPORTANT HONESTY NOTE <<<
This is a CPU/GPU prototype. Nothing here is claimed as an actual Loihi-2
implementation. ``lava`` is *auto-detected*: if it happens to be importable a
minimal CPU-backend example becomes available; otherwise this module exposes a
documented placeholder plus a hardware-agnostic spec exporter describing exactly
how each component would map to Lava Processes / Loihi-2 primitives.

Component -> Lava / Loihi-2 mapping
-----------------------------------
    spike encoders        -> Lava Process (e.g. custom `AbstractProcess` emitting
                             spikes; delta/event encoders map well to Loihi sigma-
                             delta / threshold units).
    LIF reservoir         -> `lava.proc.lif.LIF` population + `lava.proc.dense.Dense`
                             for fixed recurrent + input weights (no on-chip learning).
    R-STDP readout        -> `lava.proc.dense.Dense` with a Loihi learning rule
                             (`Loihi2FLearningRule`): pre/post traces (x1/y1) and a
                             reward/third-factor tag drive the synaptic update. Our
                             eligibility-trace three-factor rule is expressible with
                             Loihi's trace + reward graded-spike machinery.
    output spike counters -> Lava Process accumulating output spikes (or a LIF with
                             very high threshold acting as an integrator).
    reward / error signal -> external modulatory Process delivering a graded
                             "reward" spike to the plastic Dense each epoch/sample.

Why this maps cleanly: the prototype already (1) keeps the reservoir fixed,
(2) restricts plasticity to a single readout projection, (3) uses only local
pre/post traces + a scalar/vector third factor, (4) communicates via explicit
binary spikes, and (5) supports weight/state/trace quantization -- all Loihi-2
constraints.
"""
from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
def lava_available() -> bool:
    return importlib.util.find_spec("lava") is not None


# --------------------------------------------------------------------------- #
@dataclass
class ComponentSpec:
    name: str
    kind: str                       # 'process' | 'connection' | 'learning' | 'io'
    n_neurons: int
    n_synapses: int
    plastic: bool
    lava_target: str
    notes: str = ""


def export_model_spec(model, quant_bits: Optional[int] = None) -> Dict:
    """Produce a hardware-agnostic spec dict from a built ``SpikingReservoirModel``.

    A future Lava builder could consume this to instantiate Processes/Connections.
    """
    res_dim = model.reservoir_dim
    in_ch = model.reservoir.in_channels
    conn = getattr(model.res_cfg, "connectivity", 0.0)
    components: List[ComponentSpec] = [
        ComponentSpec("encoder", "io", 0, in_ch, False,
                      "lava Process (spike source)",
                      f"encoder={model.encoder_cfg}"),
        ComponentSpec("input_projection", "connection", 0, in_ch * res_dim, False,
                      "lava.proc.dense.Dense", "fixed input weights"),
        ComponentSpec("reservoir", "process", res_dim,
                      int(res_dim * res_dim * conn), False,
                      "lava.proc.lif.LIF + Dense (recurrent)",
                      f"leak={model.res_cfg.leak}, v_th={model.res_cfg.v_threshold}, "
                      f"fixed sparse recurrent (conn={conn})"),
    ]
    if model.hidden is not None:
        components.append(ComponentSpec(
            "hidden", "process", model.hidden_size, res_dim * model.hidden_size, False,
            "lava.proc.lif.LIF + Dense", "fixed feed-forward hidden projection"))
        readout_pre = model.hidden_size
    else:
        readout_pre = res_dim
    components.append(ComponentSpec(
        "readout", "learning", model.out_dim, readout_pre * model.out_dim, True,
        "lava.proc.dense.Dense + Loihi2FLearningRule",
        "PLASTIC: three-factor (pre/post traces + reward). The only learned weights."))
    components.append(ComponentSpec(
        "output_counter", "io", model.out_dim, 0, False,
        "lava Process (spike accumulator)", "argmax over output spike counts"))
    components.append(ComponentSpec(
        "reward", "io", 0, 0, False, "external modulatory Process",
        "delivers per-output reward / error (third factor)"))

    return {
        "task_type": model.task_type,
        "lava_installed": lava_available(),
        "is_loihi2_deployment": False,
        "quantization_bits": quant_bits,
        "components": [asdict(c) for c in components],
        "loihi2_readiness": loihi2_readiness_checklist(),
    }


def loihi2_readiness_checklist() -> Dict[str, bool]:
    """What the prototype already satisfies for an eventual on-chip port."""
    return {
        "fixed_reservoir_no_on_chip_learning": True,
        "plasticity_restricted_to_one_readout_projection": True,
        "updates_are_local_pre_post_traces_plus_third_factor": True,
        "no_bptt_in_main_method": True,
        "explicit_binary_spike_communication": True,
        "supports_weight_state_trace_quantization": True,
        "modest_layer_sizes": True,
        "graded_reward_delivered_as_external_process": True,
    }


def describe_mapping() -> str:
    """Return the component->Lava mapping as a markdown table (for docs/reports)."""
    rows = [
        ("Spike encoders", "Lava `Process` (spike source)", "no"),
        ("Input projection", "`lava.proc.dense.Dense` (fixed)", "no"),
        ("LIF reservoir", "`lava.proc.lif.LIF` + recurrent `Dense` (fixed)", "no"),
        ("Hidden layer (optional)", "`lava.proc.lif.LIF` + `Dense` (fixed)", "no"),
        ("R-STDP readout", "`Dense` + `Loihi2FLearningRule` (pre/post traces + reward)", "YES"),
        ("Output counters", "Lava `Process` / high-threshold integrator", "no"),
        ("Reward / error", "external modulatory `Process` (third factor)", "no"),
    ]
    out = ["| Component | Lava / Loihi-2 target | Plastic on-chip |",
           "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def build_lava_network(spec: Dict):
    """Build a minimal Lava CPU-backend network from a spec.

    Only available when ``lava`` is importable. Raises an informative error
    otherwise, explaining what is needed for an actual Lava / Loihi-2 run.
    """
    if not lava_available():
        raise NotImplementedError(
            "Lava is not installed, so no Lava network is built (this prototype runs "
            "on CPU/GPU). To enable an actual Lava/Loihi-2 path:\n"
            "  1. pip install lava-nc  (CPU simulation backend)\n"
            "  2. map this spec's components onto lava.proc.lif.LIF / dense.Dense\n"
            "  3. attach a Loihi2FLearningRule to the readout Dense with pre/post\n"
            "     traces and a reward channel for the three-factor update\n"
            "  4. run with Loihi2SimCfg (CPU sim) -- and only on real hardware/NxSDK\n"
            "     may this be described as a Loihi-2 deployment.\n"
            "The exported spec (src.lava_export.export_model_spec) is the hand-off point."
        )
    # Minimal CPU-sim skeleton (executed only if lava is present).
    from lava.proc.lif.process import LIF          # type: ignore
    from lava.proc.dense.process import Dense       # type: ignore
    res = next(c for c in spec["components"] if c["name"] == "reservoir")
    lif = LIF(shape=(res["n_neurons"],))
    dense = Dense(weights=None)                      # weights filled by a real builder
    return {"reservoir_lif": lif, "input_dense": dense,
            "note": "skeleton only; populate weights from the trained prototype"}


# --------------------------------------------------------------------------- #
def main():
    import argparse
    from .datasets import load_dataset
    from .model import SpikingReservoirModel
    from .utils import (QuantConfig, get_device, load_config, save_json, set_seed,
                        apply_overrides, parse_cli_overrides, ensure_dir)
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), parse_cli_overrides(args.overrides))
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    qcfg = QuantConfig.from_dict(cfg.get("quantization"))
    bundle = load_dataset(cfg["dataset"], seed=cfg.get("seed", 0))
    meta = {"task_type": bundle.task_type, "in_channels": bundle.in_channels,
            "out_dim": bundle.out_dim}
    model = SpikingReservoirModel(meta, cfg, device, seed=cfg.get("seed", 0), qcfg=qcfg)

    spec = export_model_spec(model, quant_bits=qcfg.bits if qcfg.enabled else None)
    print(f"Lava installed: {lava_available()}   (is_loihi2_deployment=False)\n")
    print(describe_mapping(), "\n")
    print("Loihi-2 readiness:")
    for k, v in spec["loihi2_readiness"].items():
        print(f"  [{'x' if v else ' '}] {k}")
    out = os.path.join(cfg.get("output_dir", "results"), cfg["name"])
    ensure_dir(out)
    save_json(spec, os.path.join(out, "lava_model_spec.json"))
    print(f"\nexported hardware-agnostic spec -> {out}/lava_model_spec.json")


if __name__ == "__main__":
    main()
