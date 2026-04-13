#!/usr/bin/env bash
set -euo pipefail
cd /home/ensismoebius/Repos/doutorado/software/nn
run_id="20260413_014410_resume"
while true; do
  done_count=$(ls src/experiments/03/results/*grid*.json 2>/dev/null | wc -l)
  if [[ "$done_count" -ge 402 ]]; then
    break
  fi
  sleep 120
done
python3 scripts/analyze_grid_results.py >/tmp/analyze_final.log 2>&1 || true
{
  echo "FINAL_RUN_ID=$run_id"
  echo "FINAL_TIMESTAMP=$(date -Iseconds)"
  echo "FINAL_COMPLETED=$(ls src/experiments/03/results/*grid*.json 2>/dev/null | wc -l)"
  echo "MASTER_ROWS=$(wc -l < analysis/master_comparison.csv)"
  echo "TOP10="
  head -11 analysis/master_comparison.csv
  echo
  echo "SUMMARY_STATS="
  cat analysis/summary_statistics.csv
} > logs/grid_runs/${run_id}_final_findings.txt
