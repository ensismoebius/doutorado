#!/bin/bash
# thesis_warm_feature_cache.sbatch -- fill the feature cache the per-fold phases read
# (src/experiments/thesis/lib/include/FeatureDiskCache.hpp), once per deploy, BEFORE any
# THESIS_PHASE=perfold|pfbiomex|pfworld|pfbxworld array of thesis_gridunesp.sbatch.
#
# Why. Every per-fold unit ranks all Phase 00 candidates of its signal; the 138 handcrafted
# voice candidates alone take ~3.4 h of one CPU per unit (GridUnesp jobs 5330057/5330058,
# 2026-11-18), and they are the same matrices in every unit that loads the same samples.
# This array computes each of them once; the phases then only read them, and a unit that
# finds one missing stops instead of recomputing it in silence.
#
# The units: one per distinct data view of the per-fold profiles -- the profile's "dataset"
# block without results_dir (it decides which samples load: base, condition, fusion), plus
# the ranked signal. Coretto has 7 (voice, eegpron, eegimag and the four fusions, whose view
# also holds the EEG: a different fingerprint), BIOMEX-DB 4. The first profile of each group
# (sorted) does the work; the cache key is the view's content, so two groups that happen to
# load identical samples write the same files (harmlessly: atomic rename).
#
# Usage (from ~/software/nn on GridUnesp, after gridunesp_deploy.sh):
#   sbatch --array=0-10 scripts/pipeline/thesis/thesis_warm_feature_cache.sbatch
#   (the array must be 0..N-1 for the N groups it lists; it refuses another N and prints N)
# Then, when every task COMPLETED, submit the phases. The cache lives in
# results/thesis/feature_cache/ (not synced by the deploy), keyed by the deploy's revision:
# a redeploy needs a new warm run.
#
#SBATCH --job-name=thesis-warm
#SBATCH --partition=short
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=thesis_warm_%A_%a.out
#SBATCH --error=thesis_warm_%A_%a.err

set -euo pipefail
refuse() { echo "REFUSED: $*" >&2; exit 1; }

module load gridunesp
module load miniconda/24.4.0-libmamba
eval "$(conda shell.bash hook)"
conda activate meeting01-build
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || refuse "submit as an array (see the header)"
BUILD_DIR="out/build/max-performance"
BIN="${BUILD_DIR}/src/experiments/thesis/thesis"
for f in SOURCE_REVISION "${BUILD_DIR}/SOURCE_REVISION.thesis.built" "$BIN"; do
  [[ -e "$f" ]] || refuse "$PWD/$f is missing: run scripts/pipeline/meeting01/gridunesp_deploy.sh"
done
cmp -s SOURCE_REVISION "${BUILD_DIR}/SOURCE_REVISION.thesis.built" ||
  refuse "the thesis binary was not built from the sources on disk; rerun gridunesp_deploy.sh"
source scripts/pipeline/meeting01/_provenance.sh

# The groups, one profile path per line (see the header).
GROUPS_FILE="$(mktemp)"
python3 - src/experiments/thesis/profiles > "$GROUPS_FILE" <<'PY' ||
import glob, json, os, sys
root = sys.argv[1]
groups = {}
for phase in ("perfold", "pfbiomex", "pfworld", "pfbxworld"):
    for path in sorted(glob.glob(os.path.join(root, phase, "*.json"))):
        p = json.load(open(path))
        if not p.get("selection", {}).get("enabled"):
            sys.exit(f"{path}: a per-fold phase holds a profile without selection.enabled")
        ds = {k: v for k, v in p["dataset"].items() if k != "results_dir"}
        key = json.dumps([ds, p["selection"]["rank_modality"]], sort_keys=True)
        groups.setdefault(key, path)
if not groups:
    sys.exit("no per-fold profile found")
for key in sorted(groups):
    print(groups[key])
PY
  { rm -f "$GROUPS_FILE"; refuse "could not list the groups (above)"; }
mapfile -t PROFILES < "$GROUPS_FILE"
rm -f "$GROUPS_FILE"
(( SLURM_ARRAY_TASK_MIN == 0 && SLURM_ARRAY_TASK_COUNT == ${#PROFILES[@]} &&
   SLURM_ARRAY_TASK_MAX + 1 == ${#PROFILES[@]} )) ||
  refuse "the array must be 0..$(( ${#PROFILES[@]} - 1 )) (${#PROFILES[@]} groups), got" \
         "${SLURM_ARRAY_TASK_MIN}..${SLURM_ARRAY_TASK_MAX}: sbatch --array=0-$(( ${#PROFILES[@]} - 1 )) $0"

PROFILE="${PROFILES[$SLURM_ARRAY_TASK_ID]}"
export THESIS_FEATURE_CACHE="$PWD/results/thesis/feature_cache"
export THESIS_FEATURE_CACHE_REVISION="$(provenance_label SOURCE_REVISION)"
export THESIS_FEATURE_CACHE_MODE=write
mkdir -p "$THESIS_FEATURE_CACHE"
echo "[thesis-warm] $(date '+%F %T') task ${SLURM_ARRAY_TASK_ID}/${#PROFILES[@]}: ${PROFILE}" \
     "revision ${THESIS_FEATURE_CACHE_REVISION}"
start=$SECONDS
"$BIN" --config "$PROFILE" --warm-feature-cache
echo "[thesis-warm] $(date '+%F %T') done ${PROFILE} ($(( SECONDS - start )) s)"
