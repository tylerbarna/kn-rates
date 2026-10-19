#!/bin/bash
set -euo pipefail

cd /users/8/barna314/kn-rates/

array_job_id=$(sbatch --parsable scripts/run_detection_criteria_sweep.sh)
quality_job_id=$(sbatch --parsable \
  --dependency="afterok:${array_job_id}" \
  scripts/run_detection_criteria_quality.sh)
rate_limit_job_id=$(sbatch --parsable \
  --dependency="afterok:${array_job_id}" \
  scripts/run_rate_limit.sh)

printf 'Submitted detection array: %s\n' "$array_job_id"
printf 'Submitted quality audit: %s (afterok:%s)\n' "$quality_job_id" "$array_job_id"
printf 'Submitted rate limit: %s (afterok:%s)\n' "$rate_limit_job_id" "$array_job_id"
