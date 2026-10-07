"""meeting01_results.py -- which files in results/meeting01/ belong to one meeting01 run,
and whether their numbers can be used. Shared by 02_, 03_ and 04_ so the three readers
cannot drift apart again (they did: 03_ matched file names strictly, 02_/04_ loosely).

Three questions every reader has to answer the same way:

1. Is this file this run's?
   Every per-fold file is named  <run_tag>_<dataset>_fold<f>_<what>.
       meeting01_loso_fsdd_fold0_per_window_errors.csv        this run, dataset fsdd
       meeting01_loso_smoke_fsdd_fold0_per_window_errors.csv  another run whose tag merely
                                                              starts with this one: left out
       meeting01_loso_fold0_per_window_errors.csv             this run, NO dataset: refused
   The last kind predates the multi-dataset pipeline (2026-09-08). Nothing in it records
   which dataset produced it, and the old reading -- "assume fsdd" -- would have filed EEG
   numbers under speech without a word.

2. Was it written by a binary whose numbers can be compared with today's?
   Every fold's <run_tag>_<dataset>_fold<f>_split_manifest.json records "results_format"
   (kResultsFormat in Meeting01Output.hpp). Readers accept exactly RESULTS_FORMAT.

3. Is a per-window file ONE run's rows, whole?
   One row per (model, seed, window). A second copy means two runs of the fold were mixed
   into one file (binaries before 2026-10-06 appended instead of replacing it).
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import defaultdict

#: The results format this code reads: kResultsFormat in Meeting01Output.hpp. Keep the two
#: equal, and bump both when a change makes new numbers incomparable with old ones.
#: A manifest WITHOUT the key was written by a binary older than 2026-10-06, and nothing
#: done after the fact makes its numbers usable:
#:   - up to commit 6f332734 (2026-09-24) a zero-padded window -- the last one of a FSDD /
#:     AudioMNIST recording -- was z-scored together with its padding and scored over it,
#:     so the families saw window VALUES no reference computed today sees;
#:   - from that commit until 2026-10-06 the LSTM/GRU/Transformer-AE path crashed on its
#:     first batch, so those binaries finished no fold;
#:   - baseline test rows said train_ms 0 (the final fit was timed, but the time was never
#:     written anywhere) and per-window rows carried no encoding.
RESULTS_FORMAT = 2

PW_HEADER = [
    "model", "encoding", "architecture", "v_th", "alpha", "run_id", "seed",
    "cv_fold", "split", "speaker_id", "recording_id", "window_id",
    "source_window_index", "mse", "mae",
]
WINDOW_FIELDS = ("speaker_id", "recording_id", "window_id", "source_window_index")
TRAINED_MODELS = ("snn-ae", "lstm-ae", "gru-ae", "transformer-ae")
REFERENCE_MODELS = ("mean", "pca")
#: 03_ writes every reference row with seed 0: a reference is fitted once,
#: deterministically, so it HAS no seed -- 0 marks that, it is not a seed value.
REFERENCE_SEED = "0"

_RERUN = "rerun the fold with the current binary"


class ResultsError(RuntimeError):
    """A result file cannot be used faithfully; the message names cause and remedy."""


def fold_files(results_dir: pathlib.Path, run_tag: str, suffix_re: str) -> list:
    """[(path, dataset, fold, match)] for every <run_tag>_<dataset>_fold<f>_<suffix> file
    in results_dir, sorted by name. `suffix_re` is a regular expression (re.escape a literal
    file ending); its named groups are available on `match`."""
    if not results_dir.is_dir():
        raise ResultsError(f"{results_dir} is not a directory.")
    tag = re.escape(run_tag)
    # Dataset tokens never contain "_", so another run's longer tag cannot match here.
    ours = re.compile(rf"^{tag}_(?P<ds>[a-z0-9-]+)_fold(?P<fold>\d+)_{suffix_re}$")
    no_dataset = re.compile(rf"^{tag}_fold\d+_{suffix_re}$")
    found = []
    for path in sorted(results_dir.iterdir()):
        m = ours.match(path.name)
        if m:
            found.append((path, m["ds"], int(m["fold"]), m))
        elif no_dataset.match(path.name):
            raise ResultsError(
                f"{path.name} has no dataset segment (<run_tag>_<dataset>_fold<f>_...): it "
                "predates the multi-dataset pipeline of 2026-09-08 and nothing in it says "
                "which dataset produced it. Move it out of the results directory; if its "
                f"numbers are needed, {_RERUN}.")
    return found


def check_results_format(manifest: dict, where: str) -> None:
    got = manifest.get("results_format")
    if got == RESULTS_FORMAT:
        return
    if got is None:
        raise ResultsError(
            f"{where} has no results_format: the fold was written by a meeting01 binary older "
            "than 2026-10-06, whose numbers cannot be compared with today's references -- "
            "padded windows were normalized together with their padding and scored over it, "
            "baseline test rows say train_ms 0 (the real time was never written down) and "
            f"per-window rows have no encoding. No post-processing recovers these: {_RERUN}.")
    raise ResultsError(
        f"{where} has results_format {got!r}, but this script reads {RESULTS_FORMAT}: update "
        "the post-processing scripts together with the binary (RESULTS_FORMAT in "
        "meeting01_results.py, kResultsFormat in Meeting01Output.hpp).")


def split_manifest(results_dir: pathlib.Path, run_tag: str, dataset: str, fold: int) -> dict:
    """The fold's split manifest, checked to be this fold's and in RESULTS_FORMAT."""
    path = results_dir / f"{run_tag}_{dataset}_fold{fold}_split_manifest.json"
    if not path.exists():
        raise ResultsError(
            f"{path.name} is missing. The binary writes it at fold start, before any "
            "training, so without it nothing says which binary produced this fold's numbers; "
            f"{_RERUN}.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset or manifest.get("cv_fold") != fold:
        raise ResultsError(
            f"{path.name} says dataset={manifest.get('dataset')!r} "
            f"fold={manifest.get('cv_fold')!r}: it was renamed or copied from another fold.")
    check_results_format(manifest, path.name)
    return manifest


def read_per_window(path: pathlib.Path, fold: int, check_references: bool = True) -> list:
    """Every row of one fold's per-window CSV (values as strings), after checking that the
    file is one run's rows, whole: the expected header, rows of this fold only, known models,
    an encoding on every trained-family row, and no window scored twice by one model and
    seed. With check_references, the mean/pca rows must also carry the reference seed and
    be identical across encodings (03_ writes them so; 03_ itself, which replaces them,
    reads with check_references=False)."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != PW_HEADER:
            raise ResultsError(
                f"{path.name}: header {reader.fieldnames} != {PW_HEADER}; written by a "
                "meeting01 version these scripts do not read.")
        rows = list(reader)

    seen: set = set()
    reference_values: dict = defaultdict(set)
    for line, r in enumerate(rows, start=2):
        where = f"{path.name} line {line}"
        if int(r["cv_fold"]) != fold:
            raise ResultsError(f"{where}: a fold-{r['cv_fold']} row in fold {fold}'s file.")
        model, window = r["model"], tuple(r[k] for k in WINDOW_FIELDS)
        if model in TRAINED_MODELS:
            if not r["encoding"]:
                raise ResultsError(
                    f"{where}: a {model} row without an encoding -- written by a binary older "
                    f"than 2026-10-06 (see RESULTS_FORMAT); {_RERUN}.")
            key = (model, r["seed"], r["split"], *window)
        elif model in REFERENCE_MODELS:
            if not check_references:
                continue
            if r["seed"] != REFERENCE_SEED:
                raise ResultsError(
                    f"{where}: a {model} reference row with seed {r['seed']}; 03_ writes "
                    f"references with seed {REFERENCE_SEED} (they have none). Rerun 03_.")
            key = (model, r["encoding"], r["split"], *window)
            reference_values[(model, r["split"], *window)].add((r["mse"], r["mae"]))
        else:
            raise ResultsError(
                f"{where}: unknown model {model!r}. Add it to TRAINED_MODELS or "
                "REFERENCE_MODELS in meeting01_results.py, so it is analysed instead of "
                "silently left out.")
        if key in seen:
            raise ResultsError(
                f"{where}: {model} scores window {r['window_id']} (seed {r['seed']}) a second "
                "time -- the file mixes two runs of this fold (binaries before 2026-10-06 "
                f"appended to it instead of replacing it); {_RERUN}.")
        seen.add(key)

    for (model, _split, *window), values in reference_values.items():
        if len(values) > 1:
            raise ResultsError(
                f"{path.name}: the {model} reference differs between encodings for window "
                f"{window[2]}. A reference is scored against the analog window, so it is the "
                "same for every encoding; these rows come from a 03_ older than 2026-10-06, "
                "which scored it on the encoded input. Rerun 03_.")
    return rows
