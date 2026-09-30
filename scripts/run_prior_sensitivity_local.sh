#!/usr/bin/env bash
# Small smoke run only; use run_prior_sensitivity.sh on the cluster.
set -euo pipefail

cd "$(dirname "$(dirname "$0")")"
mkdir -p results/prior_sensitivity

conda run -n surveysim_dev python prior_sensitivity_analysis.py \
  --n-samples "${N_SAMPLES:-1000}" \
  --seed "${SEED:-20260930}" \
  --rate "${RATE:-1000.0}" \
  --output-dir "${OUTPUT_DIR:-results/prior_sensitivity}"
