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
# --cpus-per-task=28 is the node's full physical core count (2x14-core Xeon E5-2680
# v4); the v3 manual (2026-09-23 check, see .wiki/Guides/GridUnesp-Deployment.md
# Sec 1) says only ~26 cores/node are actually schedulable, 2 reserved for the OS --
# if this job sits in PENDING instead of starting, try --cpus-per-task=26.
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
# directory trees this run reads (4 dataset roots, ~24GB total -- see the dataset
# size note below) and writes (results/meeting01/, growing continuously over days).
# SHARED_FS=true tells job-nanny to operate directly against shared storage instead
# of staging to local /tmp -- the read choice here, since datasets are read-only
# and results are incrementally-written, so shared-storage latency is not the
# bottleneck for either direction, and
# staying on shared storage means results survive a node failure without waiting
# on job-nanny's checkpoint interval. NOT independently confirmed against
# GridUnesp's docs for a directory-shaped workload like this one -- verify with
# support.ncc@unesp.br before the first long submission; the training loop itself
# is safe to interrupt and RESUME=1 restart if this guess is wrong.
#
# Dataset size note (2026-09-23): the 4 dataset roots read here total ~24GB
# (fsdd 27MB + audiomnist 357MB + eegmmidb 3.4GB + siena 20.3GB), fetched once
# ahead of time via ensure_datasets.sh on the login node (see
# .wiki/Guides/GridUnesp-Deployment.md Sec 3) -- this job itself does not touch
# the network.
# support.ncc@unesp.br before the first long submission; the training loop itself
# is safe to interrupt and RESUME=1 restart if this guess is wrong.
export SHARED_FS=true

export EXPERIMENT_CONFIRMED=1
export SKIP_BUILD=1
export SKIP_POSTPROCESS=1
export MEETING01_BUILD=max-performance
export RESUME="${RESUME:-0}"

job-nanny bash scripts/pipeline/meeting01/01_meeting01_run_loso.sh
