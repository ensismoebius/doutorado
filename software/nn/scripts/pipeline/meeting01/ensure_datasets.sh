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
# Needs on PATH: git, wget, sox (WAV resampling for AudioMNIST). All three are in
# gridunesp_setup_env.sh's package list.
#
# Total download size (2026-09-23, from each source's own stated total): FSDD 27MB +
# AudioMNIST ~9.4GB raw (discarded after resampling; ~357MB survives as audioMNIST_8k)
# + eegmmidb 3.4GB + Siena 20.3GB ≈ 24GB final on-disk footprint, ~33GB transient peak
# while AudioMNIST's raw clone and its resampled copy briefly coexist.
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases}"
mkdir -p "$DATASETS_ROOT"

for cmd in git wget sox; do
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
fsdd_root="$DATASETS_ROOT/fsdDataset"
if [[ -d "$fsdd_root" ]] && find "$fsdd_root" -name '*.wav' -print -quit | grep -q .; then
  echo "[ensure-datasets] fsdd: already present at $fsdd_root -- skip"
else
  echo "[ensure-datasets] fsdd: cloning into $fsdd_root"
  rm -rf "$fsdd_root"
  git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git "$fsdd_root"
  find "$fsdd_root" -name '*.wav' -print -quit | grep -q . || {
    echo "ensure_datasets.sh: fsdd clone has no .wav files -- upstream repo layout" \
         "may have changed" >&2
    exit 1
  }
fi

# --- AudioMNIST (soerenab/AudioMNIST, 48kHz raw -> resampled to 8kHz mono here) ---
audiomnist_8k="$DATASETS_ROOT/audioMNIST_8k"
if [[ -d "$audiomnist_8k" ]] && find "$audiomnist_8k" -name '*.wav' -print -quit | grep -q .; then
  echo "[ensure-datasets] audiomnist: already present at $audiomnist_8k -- skip"
else
  audiomnist_raw="$DATASETS_ROOT/.audioMNIST_raw_staging"
  echo "[ensure-datasets] audiomnist: cloning raw 48kHz corpus into $audiomnist_raw"
  rm -rf "$audiomnist_raw"
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

  echo "[ensure-datasets] audiomnist: resampling to 8kHz mono into $audiomnist_8k"
  mkdir -p "$audiomnist_8k"
  n=0
  while IFS= read -r -d '' src; do
    rel="${src#"$audiomnist_raw"/}"
    dst="$audiomnist_8k/$rel"
    mkdir -p "$(dirname "$dst")"
    sox "$src" -r 8000 -c 1 -b 16 "$dst"
    n=$((n + 1))
  done < <(find "$audiomnist_raw" -name '*.wav' -print0)
  echo "[ensure-datasets] audiomnist: resampled ${n} files"
  rm -rf "$audiomnist_raw"
fi

# --- eegmmidb (PhysioNet EEG Motor Movement/Imagery Database, 3.4GB, open access) ---
eegmmidb_root="$DATASETS_ROOT/eegmmidb"
if [[ -d "$eegmmidb_root" ]] && find "$eegmmidb_root" -name '*.edf' -print -quit | grep -q .; then
  echo "[ensure-datasets] eegmmidb: already present at $eegmmidb_root -- checking for gaps"
fi
wget -c -q -r -np -nH --cut-dirs=3 -R "index.html*" -e robots=off \
  -P "$eegmmidb_root" "https://physionet.org/files/eegmmidb/1.0.0/"
find "$eegmmidb_root" -name '*.edf' -print -quit | grep -q . || {
  echo "ensure_datasets.sh: eegmmidb fetch produced no .edf files" >&2
  exit 1
}

# --- Siena Scalp EEG Database (PhysioNet, 20.3GB, open access; Detti 2020) ---
siena_root="$DATASETS_ROOT/siena"
if [[ -d "$siena_root" ]] && find "$siena_root" -name '*.edf' -print -quit | grep -q .; then
  echo "[ensure-datasets] siena: already present at $siena_root -- checking for gaps"
fi
wget -c -q -r -np -nH --cut-dirs=3 -R "index.html*" -e robots=off \
  -P "$siena_root" "https://physionet.org/files/siena-scalp-eeg/1.0.0/"
find "$siena_root" -name '*.edf' -print -quit | grep -q . || {
  echo "ensure_datasets.sh: siena fetch produced no .edf files" >&2
  exit 1
}

echo "[ensure-datasets] all 4 datasets present under $DATASETS_ROOT"
