#!/usr/bin/env bash
# Sweep key R-STDP hyper-parameters (learning rate x reward scale) for a
# classification experiment. Each combo writes to its own results subdir.
#   bash scripts/sweep_rstdp_params.sh configs/basicmotions.yaml
set -e
cd "$(dirname "$0")/.."
CFG=${1:-configs/basicmotions.yaml}
NAME=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['name'])")
LRS=${2:-"0.02 0.05 0.1"}
REWARDS=${3:-"0.5 1.0 2.0"}
for LR in $LRS; do
  for R in $REWARDS; do
    OUT=results/${NAME}/sweep_rstdp/lr${LR}_r${R}
    echo ">> lr=$LR reward_scale=$R -> $OUT"
    python -m src.train_rstdp --config "$CFG" \
      readout.lr=$LR readout.reward_scale=$R output_dir="$OUT"
  done
done
echo ">> R-STDP param sweep done under results/${NAME}/sweep_rstdp/."
