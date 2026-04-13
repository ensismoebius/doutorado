#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/experiments/03/scripts/run_full_grid_and_analyze.sh \
#     --project-root <path> \
#     --profiles-output-dir <path> \
#     --results-dir <path> \
#     --analysis-output-dir <path> \
#     [--jobs <n>] \
#     [--timeout <seconds>]
#
# Required CLI parameters:
#   --project-root        Project root directory.
#   --profiles-output-dir Output directory for create_test_profiles.py.
#   --results-dir         Directory where experiment results JSON files are written/read.
#   --analysis-output-dir Directory where analysis CSVs are written.

usage() {
  cat <<'EOF'
Usage:
  bash run_full_grid_and_analyze.sh \
    --project-root <path> \
    --profiles-output-dir <path> \
    --results-dir <path> \
    --analysis-output-dir <path> \
    [--jobs <n>] \
    [--timeout <seconds>]
Required CLI parameters:
    --project-root        Project root directory.
    --profiles-output-dir Output directory for create_test_profiles.py.
    --results-dir         Directory where experiment results JSON files are written/read.
    --analysis-output-dir Directory where analysis CSVs are written.
    Optional CLI parameters:
    --jobs                Number of parallel jobs to run (default: 12).
    --timeout             Timeout in seconds for each job (default: 600).
Example:
    bash run_full_grid_and_analyze.sh \
    --project-root /home/ensismoebius/Repos/doutorado/software/nn \
    --profiles-output-dir /home/ensismoebius/Repos/doutorado/software/nn/profiles \
    --results-dir /home/ensismoebius/Repos/doutorado/software/nn/results \
    --analysis-output-dir /home/ensismoebius/Repos/doutorado/software/nn/analysis \
    --jobs 16 \
    --timeout 1200
EOF
}

PROJECT_ROOT=""
PROFILES_OUTPUT_DIR=""
RESULTS_DIR=""
ANALYSIS_OUTPUT_DIR=""
JOBS=12
TIMEOUT=600

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="${2:-}"
      shift 2
      ;;
    --profiles-output-dir)
      PROFILES_OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="${2:-}"
      shift 2
      ;;
    --analysis-output-dir)
      ANALYSIS_OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PROJECT_ROOT}" || -z "${PROFILES_OUTPUT_DIR}" || -z "${RESULTS_DIR}" || -z "${ANALYSIS_OUTPUT_DIR}" ]]; then
  echo "error: missing required CLI parameters" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "error: --project-root is not a directory: ${PROJECT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NN_EXPERIMENT03_LOG_LEVEL=warn

run_id="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/grid_runs

bin="./out/build/Clang_20.1.8_x86_64-pc-linux-gnu/src/experiments/03/experiment03"
profiles_glob="${PROFILES_OUTPUT_DIR%/}/*.json"

# Regenerate profiles with the current exhaustive grid definition before execution.
python3 src/experiments/03/scripts/create_test_profiles.py --output-dir "${PROFILES_OUTPUT_DIR}"

if [[ ! -x "${bin}" ]]; then
  echo "error: experiment binary not found or not executable: ${bin}" >&2
  exit 1
fi

if ! compgen -G "${profiles_glob}" >/dev/null; then
  echo "error: no profile files match ${profiles_glob}" >&2
  exit 1
fi

parallel -j "${JOBS}" \
  --timeout "${TIMEOUT}" \
  --joblog "logs/grid_runs/${run_id}_joblog.tsv" \
  "${bin} --profile {}" \
  ::: ${profiles_glob}

touch "logs/grid_runs/${run_id}_DONE"
python3 src/experiments/03/scripts/analyze_grid_results.py \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${ANALYSIS_OUTPUT_DIR}"

echo "Grid run and analysis complete."
echo "Run ID: ${run_id}"
echo "Job log: logs/grid_runs/${run_id}_joblog.tsv"
echo "Done marker: logs/grid_runs/${run_id}_DONE"
echo "Analysis: ${ANALYSIS_OUTPUT_DIR%/}/master_comparison.csv"
