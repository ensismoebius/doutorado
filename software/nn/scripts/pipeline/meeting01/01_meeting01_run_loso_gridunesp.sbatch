#!/bin/bash
# 01_meeting01_run_loso_gridunesp.sbatch — submits the nested-LOSO training loop
# (01_meeting01_run_loso.sh) to GridUnesp's Slurm queue under job-nanny, the cluster's mandatory
# data-staging wrapper. Read .wiki/Guides/GridUnesp-Deployment.md before the first submission: the
# binary must already be BUILT and stamped (gridunesp_deploy.sh does both); this script only runs it.
#
# Usage (from ~/software/nn on GridUnesp, after gridunesp_deploy.sh):
#   sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch            # fresh run
#   RESUME=1 sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch   # continue
# Smoke test of this whole path before a long queue wait (~15 min; it ends in TIMEOUT on purpose,
# then move results/meeting01 aside -- a fresh run refuses a non-empty one):
#   WAIT_CHECKPOINT=120 sbatch -p short -t 00:15:00 -c 4 --job-name=meeting01-smoke \
#     scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch
#
#SBATCH --job-name=meeting01-loso
#SBATCH --partition=long
#SBATCH --time=30-00:00:00
# --cpus-per-task=20 (the user's choice, from their own earlier timing tests; a short
# synthetic probe on 2026-10-07 had pointed the other way, so this is not settled). Each node
# has 56 CPUs (28 cores x 2 threads) but the partition is shared and nearly full: requests of
# 28, 26 and 12 all sat in PENDING ("Nodes required for job are DOWN, DRAINED or reserved ..."),
# a queue-priority effect of the `long` tier, not of the CPU count. The run is restartable
# (RESUME=1). The deploy's build step uses its own count (GRIDUNESP_BUILD_CPUS).
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=meeting01_loso_%j.out
#SBATCH --error=meeting01_loso_%j.err

set -euo pipefail

# job-nanny is NOT on a batch job's PATH: it comes with the `gridunesp` module
# (/opt/gridunesp/dist/gridunesp/bin/job-nanny -> goo-job-nanny-v2), which only a LOGIN shell
# loads automatically (GridUnesp docs v3, "Melhorando o Script de Submissão"). Without this line
# the first submission (job 5245836) waited five days in the queue and then died in one second:
# "job-nanny: command not found".
module load gridunesp
module load miniconda/24.4.0-libmamba
# `module load` only puts conda on PATH -- `conda activate` needs the shell hook that `conda init`
# normally installs into an interactive login shell's rc file, which a batch job never sources.
eval "$(conda shell.bash hook)"
conda activate meeting01-build

# The docs' OpenMP example does this; it also stops OpenMP from defaulting to every CPU of the
# node should the allocation not be enforced as an affinity mask.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export EXPERIMENT_CONFIRMED=1
export SKIP_BUILD=1
export SKIP_POSTPROCESS=1
export MEETING01_BUILD=max-performance
export RESUME="${RESUME:-0}"

# ── job-nanny, as its source (goo-job-nanny-v2, read 2026-10-13) actually behaves ──────────────
#   - INPUT and OUTPUT are required ("INPUT is missing ..." and exit 3 otherwise);
#   - `rsync -a -u $INPUT $WORK_DIR`, WORK_DIR=/store/$USER/$SLURM_JOB_ID/ under SHARED_FS=true,
#     WITHOUT -R: a nested path arrives flattened to its basename, so INPUT is ONE directory that
#     already has the layout the run expects (built below);
#   - it cd's into WORK_DIR, runs the command, and every WAIT_CHECKPOINT seconds copies OUTPUT
#     back to the submit directory (rsync -a -u, nothing deleted there);
#   - at the end it copies OUTPUT back, deletes WORK_DIR and exits with the command's own code;
#   - killed (TIMEOUT, scancel): no final copy, no cleanup. The submit directory then has the last
#     periodic copy, and WORK_DIR keeps the newest state on /store -- recover it by hand.
# /store rather than node-local /tmp: no automatic cleanup and it survives a node failure (docs
# v3, storage guide), while /tmp is wiped by a cron for users with no job on that node.
export SHARED_FS=true
export WAIT_CHECKPOINT="${WAIT_CHECKPOINT:-1800}"   # results reach ~/software/nn every 30 min
export VERBOSE=1   # job-nanny's own log lines (its flag is inverted: this also quiets its rsync)

RES=results/meeting01
BUILD_DIR="out/build/${MEETING01_BUILD}"
for f in SOURCE_REVISION "${BUILD_DIR}/SOURCE_REVISION.built" \
         "${BUILD_DIR}/src/experiments/meeting01/meeting01"; do
  [[ -e "$f" ]] || {
    echo "REFUSED: $PWD/$f is missing -- this tree is not deployed and built. Run" \
         "scripts/pipeline/meeting01/gridunesp_deploy.sh on your machine first." >&2
    exit 1
  }
done

# A fresh run must start from nothing. Before job-nanny the run script cleared the old
# checkpoints/models in place; now it clears them inside WORK_DIR, and the copy back (which never
# deletes) would leave the old ones in ~/software/nn next to the new rows.
if [[ "$RESUME" != "1" && -d "$RES" && -n "$(ls -A "$RES")" ]]; then
  echo "REFUSED: fresh run (RESUME unset) but $PWD/$RES is not empty. Move it aside first" \
       "(mv $RES ${RES}_before_${SLURM_JOB_ID}), or continue it with RESUME=1." >&2
  exit 1
fi

# INPUT: exactly what the run reads relative to its own root, in the same layout. Datasets are
# read from their absolute /home paths (the profile's dataset.sources[].root), once per fold, and
# the binary loads its shared libraries from the build tree on /home -- so do not redeploy while
# this job runs. latex_data_dir (absolute) also receives a few small files per fold.
STAGE=".job-input-${SLURM_JOB_ID}"
trap 'rm -rf -- "$STAGE"' EXIT
trap 'exit 143' TERM
mkdir -p "$STAGE/scripts/pipeline" "$STAGE/src/experiments/meeting01/profiles" \
         "$STAGE/${BUILD_DIR}/src/experiments/meeting01" "$STAGE/results"
cp -a SOURCE_REVISION "$STAGE/"
cp -a scripts/pipeline/meeting01 "$STAGE/scripts/pipeline/"
cp -a src/experiments/meeting01/profiles/meeting01-loso.json "$STAGE/src/experiments/meeting01/profiles/"
cp -a "${BUILD_DIR}/SOURCE_REVISION.built" "$STAGE/${BUILD_DIR}/"
cp -a "${BUILD_DIR}/src/experiments/meeting01/meeting01" "$STAGE/${BUILD_DIR}/src/experiments/meeting01/"
if [[ "$RESUME" == "1" && -d "$RES" ]]; then
  cp -a "$RES" "$STAGE/results/"
fi

# OUTPUT is relative to WORK_DIR; rsync copies the directory `results` into the submit
# directory, i.e. onto ~/software/nn/results.
export INPUT="$STAGE"
export OUTPUT="$STAGE/results"

job-nanny bash "$STAGE/scripts/pipeline/meeting01/01_meeting01_run_loso.sh"
