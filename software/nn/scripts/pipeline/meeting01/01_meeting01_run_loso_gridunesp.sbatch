#!/bin/bash
# 01_meeting01_run_loso_gridunesp.sbatch — submits the nested-LOSO training loop
# (01_meeting01_run_loso.sh) to GridUnesp's Slurm queue. Read
# .wiki/Guides/GridUnesp-Deployment.md before the first submission -- in particular,
# the binary must already be BUILT (see that guide's step 2) before this job runs;
# this script only runs it, it does not configure or compile.
#
# Usage (from the software/nn checkout on GridUnesp, after a successful build):
#   sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch
#   RESUME=1 sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch   # continue an interrupted run
#
#SBATCH --job-name=meeting01-loso
#SBATCH --partition=long
#SBATCH --time=30-00:00:00
#SBATCH --cpus-per-task=28
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=meeting01_loso_%j.out
#SBATCH --error=meeting01_loso_%j.err

set -euo pipefail

module load miniconda/24.4.0-libmamba
conda activate meeting01-build

# job-nanny's documented convention (INPUT/OUTPUT as space-separated FILE names,
# staged to per-node /tmp) is built for a handful of named files, not the
# directory trees this run reads (3 dataset roots) and writes (results/meeting01/,
# growing continuously over days). SHARED_FS=true tells job-nanny to operate
# directly against shared storage instead of staging to local /tmp -- the read
# choice here, since our total I/O (~500MB datasets, incrementally-written
# results) is small enough that shared-storage latency is not the bottleneck, and
# staying on shared storage means results survive a node failure without waiting
# on job-nanny's checkpoint interval. NOT independently confirmed against
# GridUnesp's docs for a directory-shaped workload like this one -- verify with
# support.ncc@unesp.br before the first long submission; the training loop itself
# is safe to interrupt and RESUME=1 restart if this guess is wrong.
export SHARED_FS=true

export EXPERIMENT_CONFIRMED=1
export SKIP_BUILD=1
export SKIP_POSTPROCESS=1
export MEETING01_BUILD=max-performance
export RESUME="${RESUME:-0}"

job-nanny bash scripts/pipeline/meeting01/01_meeting01_run_loso.sh
