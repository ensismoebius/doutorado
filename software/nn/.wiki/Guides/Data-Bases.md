# Data Bases: where the thesis data comes from and how to rebuild it

## Why this page exists

The thesis binary reads two SQLite files that are **not in the repository**: `~/database.sqlite`
(Coretto *et al.*) and `~/biomex.sqlite` (BIOMEX-DB). They are large (gigabytes), and their
licences ask you to obtain them from the source. On a new machine nothing runs until both
exist, and a wrong file does not always fail loudly: a base built from other source files, or
by an older converter, still loads and trains. That is the silent failure this page guards
against. Every base therefore goes through the same four steps: **obtain → convert → verify →
stamp**. Chapter 8 of the thesis quotes the verification results, so the text depends on them
too.

```
 source files                  converter                     SQLite                 verifier → stamp
 ─────────────                 ─────────                     ──────                 ────────────────
 Coretto .mat (S01..S15)  ──►  import_mat_dataset_to_sqlite ─► ~/database.sqlite  ──► coretto_verify_mirror  ─► ~/database.sqlite.mirror-verified
 BIOMEX RAR (51 vols.)    ──►  biomex_download + biomex_to_sqlite ─► ~/biomex.sqlite ──► biomex_verify ─► ~/biomex.sqlite.verified
                                                                         │
                                       thesis_base_facts reads both bases + stamps
                                                                         ▼
                                                      results/thesis/base_facts.csv ─► \tn{base.*} in ch. 8
```

All commands run from `software/nn/`. The scripts live in `scripts/data/` (conversion and
verification) and `scripts/analysis/` (facts the thesis quotes).

## The two bases side by side

| | Coretto *et al.* 2017 | BIOMEX-DB (Moreno-Rodríguez *et al.* 2021) |
|---|---|---|
| Role in the thesis | Phase 00 and Phase 01 (all selection happens here) | confirmatory replication only: nothing is selected on it |
| Speakers | 15 (S01..S15), Spanish | 51 volunteers, 45 kept (strict rule below) |
| Speech | pronounced **and imagined** vowels + 6 words | pronounced digits only |
| EEG | 6 channels (F3 F4 C3 C4 P3 P4), 1024 Hz | Emotiv Epoc, 14 channels, 128 Hz, written at the Coretto rates |
| Voice | 44.1 kHz | 16 kHz, written at 44.1 kHz |
| Source | authors' `.mat` files (DOI 10.1117/12.2255697); EEG re-hosted on Zenodo, record 19502780 | Mendeley Data, DOI 10.17632/s7chktmb6x.1, CC BY 4.0 |
| SQLite schema | `subject`, `modalities`, `stimulus`, `trial` (9315), `eeg_samples` (9315), `audio_samples` (1974: pronounced trials only) | `speaker` (45), `stimulus` (10), `pair` (6297: one voice + one EEG epoch per spoken digit), `recording` (key/value provenance) |
| Read by | `ThesisDataset.cpp`, `dataset.source = "coretto2017"` | `ThesisDataset.cpp`, `dataset.source = "biomex2021"` |

The profile key `dataset.root` names the file (`~/database.sqlite` or `~/biomex.sqlite`).

## Coretto *et al.* (`~/database.sqlite`)

**Obtain.** The authors distribute 30 `.mat` files, `SXX_EEG.mat` and `SXX_Audio.mat` for
S01..S15, through the link in their paper (a Google Drive folder that now asks for a sign-in).
The EEG half is also on Zenodo (record 19502780, files renamed `sub-NN_eeg.mat`); the audio is
only at the authors'.

**Convert.**

```bash
python3 scripts/data/import_mat_dataset_to_sqlite.py <folder with S01..S15> ~/database.sqlite
```

Each `.mat` row is one trial: the six channels of 4096 samples each, concatenated, then three
labels (modality 1 = imagined / 2 = pronounced, stimulus 1–11, blink). The importer stores each
channel as a float32 BLOB.

**Verify** against the Zenodo EEG copy (download it to `~/coretto_zenodo` first):

```bash
python3 scripts/data/coretto_verify_mirror.py --db ~/database.sqlite --mirror ~/coretto_zenodo
```

It checks that every `.mat` row is in the SQLite exactly once, with equal channels, modality,
stimulus and blink label. Exit status 1 on any difference. On success it writes the stamp
`~/database.sqlite.mirror-verified`: line 1 is the sha256 of the SQLite, line 2 the counts as
JSON (the run of record: 9315 / 9315 trials equal). **The audio cannot be checked this way**: it
is not in the mirror. Thesis ch. 8 says so.

## BIOMEX-DB (`~/biomex.sqlite`)

```bash
python3 scripts/data/biomex_download.py  --dest ~/biomex                # 51 RAR, keeps EEG + Voice only
python3 scripts/data/biomex_inventory.py --src ~/biomex/extracted [--csv inventory.csv]
python3 scripts/data/biomex_to_sqlite.py --src ~/biomex/extracted --out ~/biomex.sqlite
python3 scripts/data/biomex_verify.py    --src ~/biomex/extracted --db ~/biomex.sqlite
```

| Step | What it does | Fails when |
|---|---|---|
| download | fetches each volunteer's archive, checks its SHA-256 against the listing, extracts `EEG/` and `Voice/`, deletes the archive (`--keep-archives` keeps it); resumes | a hash differs |
| inventory | lists, per volunteer, every EDF and voice file and whether it is the complete protocol | — (a report: read it before changing any rule) |
| convert | pairs each spoken digit's voice window with the EEG epoch recorded while it was spoken; applies the rules below | it cannot pair a digit for certain |
| verify | re-derives every pair by a second path and checks the EDF against the CSV export and the paper's Table 1 | any difference (exit 1); writes `~/biomex.sqlite.verified` |

**The rules of record (chosen on the whole base, before any result):**

- **Strict recording rule.** A volunteer is kept only with one complete `01G10` and one
  complete `01G04` session under their own name. No per-file exceptions. That excludes F007,
  M001, M007, M010, M018 and M024, which leaves 45 volunteers.
- **Edge pairs are dropped.** The EEG goes through the Coretto authors' own FIR filter. A pair
  whose EEG epoch lies within the filter's reach (788 samples) of a session edge cannot be
  filtered like the others, so it is dropped (M015 and M019, one digit each). Result:
  6297 pairs.
- Both signals are written at the Coretto rates (44.1 kHz voice, 1024 Hz EEG), so the same
  extractors read both bases.

Changing a rule means a new base: re-convert, re-verify, re-run the BIOMEX profiles, and
update ch. 8 (which states these rules) — never patch the SQLite by hand.

## The stamps and the facts the thesis quotes

The stamp files are how a later step knows the SQLite it reads is the one that was verified:

```
~/biomex.sqlite.verified
  line 1: 7210c936…   sha256 of ~/biomex.sqlite when it passed
  line 2: {"volunteers": 45, "excluded": [...], "pairs_equal": 6297, "differences": 0, ...}
```

`scripts/analysis/thesis_base_facts.py` reads both bases **and** both stamps:

```bash
python3 scripts/analysis/thesis_base_facts.py --biomex ~/biomex.sqlite --coretto ~/database.sqlite \
    --out results/thesis/base_facts.csv
```

It refuses a stamp whose sha256 is not the current file's, which is a **loud** failure. The
silent version would be a converted-again base with an old stamp, and the thesis would then
state verification results for a file nobody verified. The CSV feeds the `base.*` keys of
`thesis_build_text_numbers.py` ([Thesis Document](./Thesis-Document.md)). Its tests:
`scripts/testing/test_thesis_base_checks.py` and `test_biomex_convert.py`, both run by CI.

## Decoded-dataset caches

The first thesis run on a base decodes every trial and writes a binary sidecar next to it:
`<root>.pronounced.e05dscache` and `<root>.imagined-<seed>.e05dscache` (about 1.6 GB each for
Coretto). Later runs load it in about 1.5 s instead of about 27 s. The header stores each source
file's size and modification time, so a changed base decodes again by itself. Set
`THESIS_NO_DATASET_CACHE=1` to bypass it. Delete the sidecars freely: they are rebuilt on
demand. (A `<root>.e05dscache` without a condition in its name comes from before the
pronounced/imagined split and is no longer read.) Details:
[Running Thesis Profiles › Decoded-dataset cache](./Running-Thesis-Profiles.md#decoded-dataset-cache).

## On the cluster

`scripts/pipeline/meeting01/gridunesp_deploy.sh` (with the `thesis` target) sends
`~/database.sqlite` and its two caches to the same paths on the cluster, and `~/biomex.sqlite`
when it exists. It **refuses** to send a `biomex.sqlite` that has no stamp or whose sha256 is no
longer the stamped one, so an unverified base cannot reach a cluster run. See [Experiment05 › Running on GridUnesp](../Experiments/Thesis.md#running-on-gridunesp-2026-11-12).

## Other bases in the repository

The Meeting01 paper uses FSDD, AudioMNIST, EEGMMIDB and Siena, fetched by
`scripts/pipeline/meeting01/ensure_datasets.sh` (see [Meeting01](../Experiments/Meeting01.md)).
They are unrelated to the thesis.

## Related

- [Experiment05](../Experiments/Thesis.md) §11 — why a second base, and what it replicates
- [Imagined Speech and EEG](../Concepts/Imagined-Speech-and-EEG.md) — the Coretto protocol
- [DataLoaders](../Core/DataLoaders.md) — the C++ loaders
