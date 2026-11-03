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
#: Format 2 was the first stamped one (commit b434792e) and is refused too; no production fold
#: was run with it. It had the padding fix, but the LSTM/GRU/Transformer-AE frames were a strided
#: gather and the LSTM/GRU cost counted one stack of two. 3 is the first with both fixed.
#: Format 3 fed the LSTM/GRU/Transformer-AE the window repeated at the SNN's time_steps
#: (S = 512 frames), where they did not learn; 4 reads it at model.recurrent_time_steps.
#: Format 4 scored the SNN-AE at every simulation step and wrote a spike_rate that counted
#: output values > 0 and an `energy` that added a spike count to 10 x MACs; 5 averages the
#: SNN's steps (model.snn_time_pool), counts real spikes, and writes acs / arith_energy_nj.
#: Format 6 makes T a gene of the SNN-AE search: every SNN selection records its own T.
#: Format 7 arms patience only once a run beats the mean predictor by
#: training.early_stop_min_skill and keeps the best monitor epoch's network, and adds the
#: memory-aware energy columns (energy_nj, energy_ops_nj, energy_mem_nj, mem_accesses,
#: energy_nj_sram8k/32k/1m), the paper's primary energy.
RESULTS_FORMAT = 7
_OLDER_FORMATS = {
    2: "the LSTM/GRU/Transformer-AE frames were a strided gather (one window sample at "
       "alternate steps per frame) instead of consecutive samples of one step, and the "
       "LSTM/GRU cost counted one stack of two, which also steered their architecture search",
    3: "the LSTM/GRU/Transformer-AE read the window repeated at the SNN's time_steps "
       "(512 frames), a length at which they did not learn (loss ~0.99), so those rows are "
       "an under-trained baseline",
    4: "the SNN-AE was scored at every simulation step (the per-step loss the 2026-11-01 "
       "gap diagnosis measured as ~0.05 test MSE), its spike_rate counted output values "
       "above zero and its energy added unitless terms",
    5: "the SNN-AE's T was one fixed value for every genome, so its selection manifests do "
       "not record the T each winner was trained with",
    6: "patience stopped networks still on the mean-predictor plateau and the last epoch's "
       "weights were kept (it reported 'cannot learn' for runs that had not started), and "
       "the rows carry only the arithmetic-only energy, not the memory-aware one the paper "
       "reports",
}

PW_HEADER = [
    "model", "encoding", "architecture", "v_th", "alpha", "run_id", "seed",
    "cv_fold", "split", "speaker_id", "recording_id", "window_id",
    "source_window_index", "mse", "mae",
]
WINDOW_FIELDS = ("speaker_id", "recording_id", "window_id", "source_window_index")
#: "ann-ae" is the matched control of the SNN: the SNN winner's layer widths with a leaky-ReLU
#: neuron, not searched (so its selection manifest has no val_mse). snn-ae stays first: 04_ pairs
#: it with every other trained model.
TRAINED_MODELS = ("snn-ae", "lstm-ae", "gru-ae", "transformer-ae", "ann-ae")
REFERENCE_MODELS = ("mean", "pca")
#: The per-window inference energy of a trained model's rows (results_format 7), in nJ unless
#: noted. energy_nj is the paper's number: Lemaire et al. 2022's model (operations + address
#: arithmetic + memory accesses, each local SRAM priced at its own size) at Horowitz 2014's
#: 45 nm prices, = energy_ops_nj + energy_mem_nj. The rest are its checks: the same counts
#: with every access priced at one SRAM size, and the older arithmetic-only estimate
#: (4.6 pJ x MACs + 0.9 pJ x ACs), which no longer carries a claim. Meeting01Energy.hpp.
ENERGY_COLUMNS = ("energy_nj", "energy_ops_nj", "energy_mem_nj", "mem_accesses",
                  "energy_nj_sram8k", "energy_nj_sram32k", "energy_nj_sram1m", "arith_energy_nj")
#: The repricings a claim about energy must survive: (column, the words the paper uses).
ENERGY_CHECKS = (("energy_nj_sram8k", "with every memory access priced at an 8 KB SRAM"),
                 ("energy_nj_sram32k", "with every memory access priced at a 32 KB SRAM"),
                 ("energy_nj_sram1m", "with every memory access priced at a 1 MB SRAM"),
                 ("arith_energy_nj", "with arithmetic alone"))
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
    if isinstance(got, int) and not isinstance(got, bool) and got < RESULTS_FORMAT:
        why = _OLDER_FORMATS.get(got, "see the history above RESULTS_FORMAT in this module")
        raise ResultsError(
            f"{where} has results_format {got}, older than the {RESULTS_FORMAT} this script "
            f"reads: {why}. Its numbers cannot be compared with a current fold's; {_RERUN}.")
    raise ResultsError(
        f"{where} has results_format {got!r}, but this script reads {RESULTS_FORMAT}: the fold "
        "is newer than the scripts (or the manifest was edited). Update the post-processing "
        "scripts together with the binary (RESULTS_FORMAT in meeting01_results.py, "
        "kResultsFormat in Meeting01Output.hpp).")


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


# ------------------------------------------------------------- development folds

#: Folds whose TEST windows were looked at while the SNN-AE design was chosen: the
#: 2026-11-01 gap diagnosis compared design variants on fold 0 of FSDD and of eegmmidb
#: (.wiki/Experiments/Meeting01.md, "The diagnosis on EEG"). Every choice came out the same
#: on those folds' validation windows, but the test windows were seen, so the author decided
#: (2026-11-02) that the paper's headline numbers leave these folds out and report the same
#: comparison with them in as a sensitivity analysis. AudioMNIST and Siena were never looked
#: at. Change this only together with the paper's text about it.
DEVELOPMENT_FOLDS: dict[str, frozenset[int]] = {
    "fsdd": frozenset({0}),
    "eegmmidb": frozenset({0}),
}


def is_headline_fold(dataset: str, fold: int) -> bool:
    """False for a development fold (DEVELOPMENT_FOLDS), True for every other fold."""
    return fold not in DEVELOPMENT_FOLDS.get(dataset, frozenset())


# --------------------------------------------------------- the run's configuration

# Fields of a fold's recorded config that legitimately differ between the folds of ONE run:
# each array task runs one (dataset, fold) with --dataset/--cv-fold, which narrows
# evaluation.datasets and sets dataset.cv_fold. model.latent_dim is resolved per dataset.
_PER_PROCESS_FIELDS = (("dataset", "cv_fold"), ("evaluation", "datasets"), ("model", "latent_dim"))


def _without_per_process_fields(config: dict) -> dict:
    out = json.loads(json.dumps(config))
    for section, key in _PER_PROCESS_FIELDS:
        out.get(section, {}).pop(key, None)
    return out


def run_configuration(results_dir: pathlib.Path, run_tag: str) -> tuple[dict, dict]:
    """(config, {dataset: {"source": ..., "latent_dim": ...}}) of the run, from the "config" /
    "source" records every fold's split manifest carries since results_format 6. Refuses a
    run whose folds disagree on anything but the per-process fields: one paper table would
    then describe two configurations."""
    config = None
    first = None
    datasets: dict = {}
    for path, ds, fold, _ in fold_files(results_dir, run_tag, re.escape("split_manifest.json")):
        man = split_manifest(results_dir, run_tag, ds, fold)
        if "config" not in man or "source" not in man:
            raise ResultsError(f"{path.name} records no config/source: {_RERUN}.")
        common = _without_per_process_fields(man["config"])
        if config is None:
            config, first = common, path.name
        elif common != config:
            diff = sorted(f"{s}.{k}" for s in common for k in common[s]
                          if config.get(s, {}).get(k) != common[s][k])
            raise ResultsError(
                f"{path.name} and {first} were run with different configurations ({diff}): "
                "a redeploy changed the profile mid-run. Rerun the folds of one of them.")
        entry = {"source": man["source"], "latent_dim": man["config"]["model"]["latent_dim"]}
        if datasets.setdefault(ds, entry) != entry:
            raise ResultsError(f"{path.name}: {ds} was resolved differently in another fold.")
    if config is None:
        raise ResultsError(f"no {run_tag}_<dataset>_fold<f>_split_manifest.json under "
                           f"{results_dir}.")
    return config, datasets


def printed_configuration(path: pathlib.Path) -> tuple[dict, dict]:
    """The same pair from `meeting01 --print-resolved-config` output, for placeholder tables
    before any fold has run."""
    printed = json.loads(path.read_text(encoding="utf-8"))
    check_results_format(printed, path.name)
    return _without_per_process_fields(printed["config"]), printed["datasets"]


# ------------------------------------------------------------------- the paper's datasets

# "mitbih" replaced 2026-09-23 by eegmmidb/chbmit, then chbmit replaced the same day by
# siena (disk space), in the active LOSO grid; both are kept so their titles stay known.
DATASET_ORDER = ["fsdd", "audiomnist", "eegmmidb", "siena", "chbmit", "mitbih"]
# Short names for the side-by-side tables.
DATASET_SHORT = {"fsdd": "FSDD", "audiomnist": "AudioMNIST", "eegmmidb": "EEGMMIDB",
                 "siena": "Siena"}

def paper_datasets(config: dict) -> list[str]:
    """The run's datasets, in paper order: every dataset.sources entry (an array task narrows
    evaluation.datasets to its own, so that list cannot be used)."""
    names = [s["name"] for s in config["dataset"]["sources"]]
    return sorted(names, key=lambda d: (DATASET_ORDER.index(d) if d in DATASET_ORDER else 99, d))


def configured_families(config: dict) -> list[str]:
    """The trained families the configuration runs, in the paper's order (the SNN-AE, its
    matched control, then the searched sequence baselines): the binary's rule, the SNN arm
    on iff evaluation.snn_architectures is non-empty and every other family iff it is in
    evaluation.baselines. Every table, figure and macro of 02_ / 04_ lists these and only
    these. A family left out of the run must leave no "--" row, column or legend entry
    behind: on the page that is indistinguishable from a family whose folds have not run
    yet, so the paper would describe a comparison it never made."""
    ev = config["evaluation"]
    unknown = sorted(set(ev["baselines"]) - set(TRAINED_MODELS[1:]))
    if unknown:
        raise ResultsError(f"evaluation.baselines has {unknown}, which no table knows: add the "
                           "family to TRAINED_MODELS and to the label maps of 02_ / 04_.")
    families = ["snn-ae"] if ev["snn_architectures"] else []
    order = ("ann-ae", "lstm-ae", "gru-ae", "transformer-ae")
    return families + [m for m in order if m in ev["baselines"]]


def resolve_source(config: dict, name: str) -> dict:
    """{"source", "latent_dim"} of dataset `name` from the run's configuration, by the rules
    of Meeting01Config::Dataset::resolve and resolve_dataset_config: the dataset-level value,
    replaced by the dataset.sources entry's where that is set (> 0 / non-empty). Needed for a
    dataset no fold has run yet; for every other one write_paper_tables checks it against
    the fold's own record, so this copy of the rules cannot drift unnoticed."""
    d = config["dataset"]
    src = {"name": name, "root": d["dataset_root"], "window_size": d["window_size"],
           "cv_num_folds": d["cv_num_folds"], "sample_rate": 0,
           "max_windows_per_recording": d["max_windows_per_recording"],
           "loso_max_train_windows": d["loso_max_train_windows"],
           "loso_max_val_windows": d["loso_max_val_windows"],
           "loso_max_test_windows": d["loso_max_test_windows"], "latent_dim": d["latent_dim"]}
    for e in d["sources"]:
        if e["name"] != name:
            continue
        for k, v in e.items():
            if k != "name" and ((isinstance(v, str) and v) or (not isinstance(v, str) and v > 0)):
                src[k] = v
        break
    if src["latent_dim"] <= 0:
        raise ResultsError(f"{name} has no latent_dim of its own (dataset.sources[]."
                           "latent_dim): set it so the paper states the bottleneck that ran.")
    return {"source": src, "latent_dim": src["latent_dim"]}


def synthetic_run_config(fold: int, baselines: tuple = ("lstm-ae", "ann-ae")) -> dict:
    """Self-test fixture of 02_ and 04_: what a fold's split manifest records under "config"
    (resolved_config_json), reduced to the fields the scripts read. fsdd and siena are
    configured; the self-tests run fsdd only, on 3 folds (02_ refuses a dataset missing any
    configured fold, so the fixture configures exactly the 3 it writes). `baselines` must
    name exactly the families the test writes rows for: the scripts take the grid, the
    tables and the Holm families from it."""
    mechanics = {"population_size": 8, "generations": 5, "winner_seeds": 3,
                 "crossover_prob": 0.9, "mutation_prob": 0.2, "tournament_k": 2, "seed": 0}
    recurrent = {**mechanics, "min_hidden": 8, "max_hidden": 256, "min_layers": 1,
                 "max_layers": 3}
    source = {"root": "", "window_size": 0, "cv_num_folds": 0, "loso_max_train_windows": 0,
              "loso_max_val_windows": 0, "loso_max_test_windows": 0}
    return {
        "experiment": {"run_tag": "t", "seed": 42, "repeats": 2, "seed_deterministic": False,
                       "check_determinism": False},
        "dataset": {"dataset_root": "/data", "window_size": 256, "cv_fold": fold,
                    "cv_num_folds": 3, "max_windows_per_recording": 0,
                    "loso_max_train_windows": 1000, "loso_max_val_windows": 150,
                    "loso_max_test_windows": 1500, "latent_dim": 0,
                    "sources": [{**source, "name": "fsdd", "sample_rate": 8000,
                                 "max_windows_per_recording": 0, "latent_dim": 16},
                                {**source, "name": "siena", "sample_rate": 512,
                                 "max_windows_per_recording": 40, "latent_dim": 64}]},
        "training": {"epochs": 30, "early_stop_patience": 5,
                     "early_stop_min_skill": 0.10000000149011612, "samples_per_batch": 1,
                     "learning_rate": 0.0010000000474974513,
                     "learning_rate_biophysical": 9.999999747378752e-05},
        "model": {"latent_dim": 16, "lstm_frame_size": 8, "recurrent_time_steps": 1,
                  "time_steps": 16, "initializer": "xavier"},
        "evaluation": {"datasets": ["fsdd"], "snn_architectures": ["dense"],
                       "baselines": list(baselines), "ga": {
            "snn": {**mechanics, "min_layers": 1, "max_layers": 4, "min_width": 4,
                    "max_width": 128, "voltage_threshold_min": 0.5,
                    "voltage_threshold_max": 1.5, "alpha_min": 0.800000011920929,
                    "alpha_max": 0.9900000095367432, "time_steps_min": 2, "time_steps_max": 32},
            "lstm": recurrent, "gru": dict(recurrent),
            "transformer": {**mechanics, "min_d_model": 16, "max_d_model": 128,
                            "head_choices": [1, 2, 4, 8], "min_layers": 1, "max_layers": 4,
                            "min_d_ff": 32, "max_d_ff": 256}}},
    }




def folds_text(folds: list[int]) -> str:
    """[1, 2, 3, 4, 5] -> 1--5; a gap is listed: [0, 2, 3] -> 0, 2--3."""
    if not folds:
        return "--"
    out, start, prev = [], folds[0], folds[0]
    for f in folds[1:] + [None]:
        if f is not None and f == prev + 1:
            prev = f
            continue
        out.append(str(start) if start == prev else f"{start}--{prev}")
        if f is not None:
            start = prev = f
    return ", ".join(out)
