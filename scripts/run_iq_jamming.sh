#!/usr/bin/env bash
# Full pipeline for the synthetic_iq_jamming experiment: R-STDP (main) + surrogate/GRU baselines
# + linear baseline + results aggregation + Lava spec export.
# Extra CLI args are forwarded to every step, e.g.:
#   bash scripts/run_iq_jamming.sh quantization.enabled=true quantization.bits=8
set -e
cd "$(dirname "$0")/.."
CFG=configs/synthetic_iq_jamming.yaml
echo ">> [1/5] R-STDP (main, local learning)"            && python -m src.train_rstdp               --config $CFG "$@"
echo ">> [2/5] surrogate-gradient + GRU baselines (BPTT)" && python -m src.train_surrogate_baseline   --config $CFG "$@"
echo ">> [3/5] linear readout baseline + aggregate"       && python -m src.evaluate                   --config $CFG "$@"
echo ">> [4/5] Lava / Loihi-2 spec export (placeholder)"  && python -m src.lava_export                --config $CFG "$@"
echo ">> [5/5] done. Results in results/synthetic_iq_jamming/"
