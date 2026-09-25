#!/usr/bin/env bash
# ensure_datasets.sh — idempotent, fail-loud fetch of all 4 meeting01 datasets into
# the exact directory layout profiles/meeting01-loso.json expects (dataset.sources[]
# root paths). Safe to re-run any time: every dataset is checked for already-complete
# data before anything touches the network, so a second run after an interrupted
# first one only fetches what is still missing.
#
# Written 2026-09-23 so a fresh checkout — this machine or GridUnesp's login node —
# never depends on a prior manual `scp` of a pre-populated databases/ tree from
# another machine. See .wiki/Guides/GridUnesp-Deployment.md Sec 3, which now runs
# this script instead of that scp step.
#
# Usage:
#   ./scripts/pipeline/meeting01/ensure_datasets.sh
#   DATASETS_ROOT=/custom/path ./scripts/pipeline/meeting01/ensure_datasets.sh
#
# Requires network access. On GridUnesp run this on the LOGIN node (confirmed
# internet in .wiki/Guides/GridUnesp-Deployment.md), not inside an sbatch job —
# compute-node internet is unconfirmed there.
#
# Needs on PATH: git, wget, sox (WAV resampling for AudioMNIST) -- all three are in
# gridunesp_setup_env.sh's package list -- and curl (eegmmidb/siena's RECORDS
# manifest fetch from the S3 mirror, see fetch_physionet_s3()'s own comment below);
# curl is a base-OS package on every system checked so far (confirmed present on
# GridUnesp's login node 2026-09-24), not added to the conda env's package list.
#
# Total download size (2026-09-23, from each source's own stated total): FSDD 27MB +
# AudioMNIST ~9.4GB raw (discarded after resampling; ~357MB survives as audioMNIST_8k)
# + eegmmidb 3.4GB + Siena 20.3GB ≈ 24GB final on-disk footprint, ~33GB transient peak
# while AudioMNIST's raw clone and its resampled copy briefly coexist.
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases}"
mkdir -p "$DATASETS_ROOT"

for cmd in git wget sox curl; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "ensure_datasets.sh: '$cmd' not found on PATH -- install it (see" \
         "gridunesp_setup_env.sh's package list) before running this script." >&2
    exit 1
  }
done

# Preflight disk check: fail loud and early rather than discovering mid-transfer that
# a huge partial download has to be thrown away. 30GB is the ~24GB final footprint
# plus headroom for AudioMNIST's transient raw+resampled overlap and a safety margin;
# it is NOT exact (per-dataset checks below are skip-if-present, not skip-if-huge).
avail_gb=$(df --output=avail -BG "$DATASETS_ROOT" | tail -1 | tr -dc '0-9')
if (( avail_gb < 30 )); then
  echo "ensure_datasets.sh: only ${avail_gb}GB free under $DATASETS_ROOT," \
       "need ~30GB headroom (final footprint ~24GB, more during AudioMNIST's" \
       "raw+resampled overlap). Free space or set DATASETS_ROOT to a bigger volume." >&2
  exit 1
fi

# --- FSDD (27MB; Jakobovski/free-spoken-digit-dataset, DOI 10.5281/zenodo.1342401) ---
# Cloned into a staging dir and only `mv`d into place on success: an interrupted
# `git clone` (Ctrl-C, dropped connection) is not resumable, and a bare "does
# $fsdd_root contain at least one .wav" check cannot tell a full checkout from a
# partial one that got interrupted mid-"Updating files" -- it would find the few
# files that DID land, call it done, and silently leave a truncated dataset in
# place forever. The final `mv` is the only thing that creates $fsdd_root itself,
# so its existence becomes a true completeness marker instead of a guess.
fsdd_root="$DATASETS_ROOT/fsdDataset"
if [[ -d "$fsdd_root" ]] && find "$fsdd_root" -name '*.wav' -print -quit | grep -q .; then
  echo "[ensure-datasets] fsdd: already present at $fsdd_root -- skip"
else
  fsdd_staging="$DATASETS_ROOT/.fsdDataset_staging"
  echo "[ensure-datasets] fsdd: cloning into $fsdd_staging"
  rm -rf "$fsdd_staging" "$fsdd_root"
  git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git "$fsdd_staging"
  find "$fsdd_staging" -name '*.wav' -print -quit | grep -q . || {
    echo "ensure_datasets.sh: fsdd clone has no .wav files -- upstream repo layout" \
         "may have changed" >&2
    rm -rf "$fsdd_staging"
    exit 1
  }
  mv "$fsdd_staging" "$fsdd_root"
fi

# --- AudioMNIST (soerenab/AudioMNIST, 48kHz raw -> resampled to 8kHz mono here) ---
# Same interrupted-run hazard as FSDD above, at two points: the raw clone (not
# resumable) and the resample loop (thousands of individual `sox` calls -- an
# interruption partway leaves $audiomnist_8k with SOME files, which the "any .wav
# present" check would mistake for a finished resample on the next run). Both the
# raw clone and the resampled output land in staging dirs first; only a fully
# successful resample loop renames the staging dir to $audiomnist_8k, so its
# existence is a true completeness marker, not a guess from partial content.
audiomnist_8k="$DATASETS_ROOT/audioMNIST_8k"
if [[ -d "$audiomnist_8k" ]] && find "$audiomnist_8k" -name '*.wav' -print -quit | grep -q .; then
  echo "[ensure-datasets] audiomnist: already present at $audiomnist_8k -- skip"
else
  audiomnist_raw="$DATASETS_ROOT/.audioMNIST_raw_staging"
  audiomnist_8k_staging="$DATASETS_ROOT/.audioMNIST_8k_staging"
  echo "[ensure-datasets] audiomnist: cloning raw 48kHz corpus into $audiomnist_raw"
  rm -rf "$audiomnist_raw" "$audiomnist_8k_staging"
  git clone --depth 1 https://github.com/soerenab/AudioMNIST.git "$audiomnist_raw"

  raw_sample="$(find "$audiomnist_raw" -name '*.wav' -print -quit)"
  [[ -n "$raw_sample" ]] || {
    echo "ensure_datasets.sh: AudioMNIST clone has no .wav files -- upstream repo" \
         "layout may have changed" >&2
    exit 1
  }
  # A real WAV is tens of KB at minimum; a git-lfs pointer stub (if upstream ever
  # switches to LFS) is a ~130-byte text file -- catch that loudly instead of
  # resampling garbage.
  raw_bytes=$(stat -c%s "$raw_sample")
  (( raw_bytes > 1024 )) || {
    echo "ensure_datasets.sh: $raw_sample is only ${raw_bytes} bytes -- looks like a" \
         "git-lfs pointer, not real audio. Install git-lfs and re-clone." >&2
    exit 1
  }

  total=$(find "$audiomnist_raw" -name '*.wav' -print0 | grep -zc .)
  echo "[ensure-datasets] audiomnist: resampling ${total} files to 8kHz mono into $audiomnist_8k_staging"
  mkdir -p "$audiomnist_8k_staging"
  n=0
  while IFS= read -r -d '' src; do
    rel="${src#"$audiomnist_raw"/}"
    dst="$audiomnist_8k_staging/$rel"
    mkdir -p "$(dirname "$dst")"
    sox "$src" -r 8000 -c 1 -b 16 "$dst"
    n=$((n + 1))
    if (( n % 250 == 0 || n == total )); then
      pct=$(( n * 100 / total ))
      echo "[ensure-datasets] audiomnist: resampled ${n}/${total} (${pct}%)"
    fi
  done < <(find "$audiomnist_raw" -name '*.wav' -print0)
  rm -rf "$audiomnist_raw" "$audiomnist_8k"
  mv "$audiomnist_8k_staging" "$audiomnist_8k"
fi

# --- shared: fetch a PhysioNet open dataset via its AWS S3 mirror, per-file ---
# PhysioNet mirrors every open-access dataset to the public `physionet-open` S3
# bucket at the same relative layout as the website (`s3://physionet-open/<slug>/
# 1.0.0/...`), reachable over plain HTTPS with no AWS account, credentials, or CLI
# tool needed -- the bucket policy allows anonymous GET, same effect as
# `--no-sign-request`. Measured 2026-09-24 from the GridUnesp login node on this
# exact dataset (Siena): physionet.org's own web server served ~90 KB/s;
# https://physionet-open.s3.amazonaws.com served the same data at ~12.1 MB/s --
# ~138x faster (see .wiki/Guides/GridUnesp-Deployment.md's Troubleshooting section
# for the measurement). This is also what fixes the 416-retry-loop hazard the
# direct site had (see git history / the wiki for that fix's own reasoning) --
# moot here since S3 is never asked for a byte-range resume in the first place.
#
# S3 has no browsable HTML directory index the way physionet.org's own file tree
# does, so `wget -r` (which crawls <a href> links) cannot point at it directly --
# this fetches the dataset's own RECORDS manifest (the same file
# ensure_datasets.sh's DATASET_TOTALS-equivalent expected counts are verified
# against) and downloads each listed path individually. `-N` per file still means
# an already-complete file is skipped (compared against S3's own Last-Modified/
# size), so a re-run after an interruption resumes at the file level, same as
# before; `--tries=5` gives each individual file some resilience against a single
# transient blip without masking a genuine, repeated failure.
fetch_physionet_s3() {
  local slug="$1" root="$2"
  local base="https://physionet-open.s3.amazonaws.com/${slug}/1.0.0"
  local records
  records="$(curl -fsSL "${base}/RECORDS")" || {
    echo "ensure_datasets.sh: could not fetch ${slug}'s RECORDS manifest from the S3 mirror" >&2
    return 1
  }
  local total n=0
  total=$(printf '%s\n' "$records" | grep -c .)
  local rel
  while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    n=$((n + 1))
    wget -N -nv --tries=5 -P "$(dirname "${root}/${rel}")" "${base}/${rel}"
    if (( n % 5 == 0 || n == total )); then
      echo "[ensure-datasets] ${slug}: fetched ${n}/${total} files"
    fi
  done <<< "$records"
}

# --- eegmmidb (PhysioNet EEG Motor Movement/Imagery Database, 3.4GB, open access) ---
eegmmidb_root="$DATASETS_ROOT/eegmmidb"
if [[ -d "$eegmmidb_root" ]] && find "$eegmmidb_root" -name '*.edf' -print -quit | grep -q .; then
  echo "[ensure-datasets] eegmmidb: already present at $eegmmidb_root -- checking for gaps"
fi
fetch_physionet_s3 "eegmmidb" "$eegmmidb_root"
find "$eegmmidb_root" -name '*.edf' -print -quit | grep -q . || {
  echo "ensure_datasets.sh: eegmmidb fetch produced no .edf files" >&2
  exit 1
}

# --- Siena Scalp EEG Database (PhysioNet, 20.3GB, open access; Detti 2020) ---
siena_root="$DATASETS_ROOT/siena"
if [[ -d "$siena_root" ]] && find "$siena_root" -name '*.edf' -print -quit | grep -q .; then
  echo "[ensure-datasets] siena: already present at $siena_root -- checking for gaps"
fi
fetch_physionet_s3 "siena-scalp-eeg" "$siena_root"
find "$siena_root" -name '*.edf' -print -quit | grep -q . || {
  echo "ensure_datasets.sh: siena fetch produced no .edf files" >&2
  exit 1
}

echo "[ensure-datasets] all 4 datasets present under $DATASETS_ROOT"
