#!/usr/bin/env bash
# Sweep reservoir size (100, 300, 500, 1000) for an experiment and aggregate.
# Each size writes to results/<name>/sweep_size/N<size>/.
#   bash scripts/sweep_reservoir_size.sh configs/mackey_glass.yaml
set -e
cd "$(dirname "$0")/.."
CFG=${1:-configs/mackey_glass.yaml}
SIZES=${2:-"100 300 500 1000"}
NAME=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['name'])")
for N in $SIZES; do
  OUT=results/${NAME}/sweep_size/N${N}
  echo ">> reservoir size N=$N -> $OUT"
  python -m src.train_rstdp --config "$CFG" reservoir.n_reservoir=$N output_dir="$OUT"
  python -m src.evaluate    --config "$CFG" reservoir.n_reservoir=$N output_dir="$OUT"
done
echo ">> sweep done. Use src.analyze or evaluate plots to compare across $OUT/.."
