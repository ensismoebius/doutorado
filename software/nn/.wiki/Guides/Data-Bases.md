# Data Bases: where the thesis data comes from and how to rebuild it

## Why this page exists

The thesis binary reads two SQLite files that are **not in the repository**: `database.sqlite`
(Coretto *et al.*) and `biomex.sqlite` (BIOMEX-DB), both in the databases directory (next section). They are large (gigabytes), and their
licences ask you to obtain them from the source. On a new machine nothing runs until both
exist, and a wrong file does not always fail loudly: a base built from other source files, or
by an older converter, still loads and trains. That is the silent failure this page guards
against. Every base therefore goes through the same four steps: **obtain → convert → verify →
stamp**. Chapter 8 of the thesis quotes the verification results, so the text depends on them
too.

```
 source files                  converter                     SQLite                 verifier → stamp
 ─────────────                 ─────────                     ──────                 ────────────────
 Coretto .mat (S01..S15)  ──►  import_mat_dataset_to_sqlite ─► database.sqlite  ──► coretto_verify_mirror  ─► database.sqlite.mirror-verified
 BIOMEX RAR (51 vols.)    ──►  biomex_download + biomex_to_sqlite ─► biomex.sqlite ──► biomex_verify ─► biomex.sqlite.verified
                                                                         │
                                       thesis_base_facts reads both bases + stamps
                                                                         ▼
                                                      results/thesis/base_facts.csv ─► \tn{base.*} in ch. 8
```

## Where everything lives: one directory

Every base, the thesis's and the Meeting01 paper's, is in one directory, written `<databases>` below:

```
<databases> = ~/Documentos/academico/UNESP/doutorado/databases

<databases>/
  database.sqlite                  Coretto et al.        (+ database.sqlite.mirror-verified, the stamp)
  database.sqlite.pronounced.e05dscache, .imagined-<seed>.e05dscache      decoded caches, 1.6 GB each
  coretto_zenodo/                  the Coretto EEG mirror, S01.zip ... S15.zip (verification only)
  biomex.sqlite                    BIOMEX-DB             (+ biomex.sqlite.verified, the stamp)
  biomex/extracted                 only while converting/verifying: the 51 volunteers, EEG + Voice (deleted afterwards)
  fsdDataset/  audioMNIST_8k/  eegmmidb/  siena/  mitbih/                  Meeting01 corpora
```

The thesis profiles name it in `dataset.root` (`~/Documentos/academico/UNESP/doutorado/databases/database.sqlite`, and
`.../biomex.sqlite`), and every script that downloads, converts or verifies a base uses it **by
default**, so none of the commands below needs a path. `~` is expanded on the machine that runs
the profile, so the cluster holds the same layout under its own home (the deploy creates it).
The directory is written in four places that `scripts/testing/test_databases_dir.py` keeps equal:
`scripts/data/databases_dir.py`, the profile generator, `gridunesp_deploy.sh` and
`ensure_datasets.sh`.

**`biomex/` is not kept once `biomex.sqlite` exists (decision 2026-11-14).** The download is a
way to *regenerate* the SQLite, not something the thesis reads: the thesis binary opens only
`biomex.sqlite`. If a rule changes, or the verifier has to run again, download the 51 archives
again (`biomex_download.py`, 8.1 GB, SHA-256 checked) and convert from them. Every source file
is verified by hash on download, so a re-download is the same input.

```
 biomex.sqlite, what it holds                       who reads it
 ──────────────────────────────────────────────     ───────────────────────────────
 speaker 1..45     excluded = 0   6294 pairs   ───► the thesis binary, base_facts, the verifier
 speaker 46..51    excluded = 1    560 pairs   ───► nobody (kept for the record)
```

**Why the six excluded volunteers are in the base but never read.** The strict rule keeps a
volunteer only if their EEG files can be matched to them with no guessing (see below). The six
it drops (F007, M001, M007, M010, M018, M024) still have data on disk, and the user's decision
was to *store* them so that nothing downloaded is lost, and to *not process* them so that the
rule chosen before any result stays the rule. Example, M024: the folder holds `F024-01G04-*.edf`
under a different volunteer's name, so which EEG belongs to M024's voice is a guess; the base
stores 140 pairs built on that guess, flagged `excluded = 1`, and no reader selects them.

| | `excluded = 0` (45 volunteers) | `excluded = 1` (6 volunteers) |
|---|---|---|
| ids | 1..45 (the order of the folders) | 46..51, after the kept ones |
| pairs | 6294 (ids 1..6294), verified against the originals | 560 (F007 140, M001 0, M007 0, M010 140, M018 140, M024 140), **not verified** |
| `excluded_reason` | `NULL` | why the rule dropped them, then `Stored: ...` what could still be paired (M001: two complete recordings, nothing says which the voice used, so no pairs for that session; M007: a digit pair 1.96 s apart, shorter than the 2 s epoch, so none) |
| read by | `ThesisDataset.cpp`, `thesis_base_facts.py`, `biomex_verify.py` | nobody: each reader's query has `WHERE speaker.excluded = 0`, with a comment saying why |

**Failure mode: silent.** A new query on `pair` without the `excluded = 0` join would still
run, and would add 560 pairs of unverified origin to the data (and 6 speakers to the group
count, which changes folds). The kept ids 1..45 and pair ids 1..6294 are deliberately
unchanged by the excluded rows (they come after, 6295..6854), so the excluded rows never move
a fold. (Leaving out the 3 WORLD-unanalysable pairs on 2026-11-16 did renumber the kept pairs;
the BIOMEX-DB replication was re-run on that base, array 5328574.) Test: `ExcludedSpeakersAreNeverRead` (C++, a truncated voice blob on an excluded
speaker would crash a reader that touched it) and `BaseFacts` in `test_thesis_base_checks.py`.

All commands run from `software/nn/`. The scripts live in `scripts/data/` (conversion and
verification) and `scripts/analysis/` (facts the thesis quotes).

## The two bases side by side

| | Coretto *et al.* 2017 | BIOMEX-DB (Moreno-Rodríguez *et al.* 2021) |
|---|---|---|
| Role in the thesis | Phase 00 and Phase 01 (all selection happens here) | confirmatory replication only: nothing is selected on it |
| Speakers | 15 (S01..S15), Spanish | 51 volunteers, 45 kept and 6 stored but never read (strict rule below) |
| Speech | pronounced **and imagined** vowels + 6 words | pronounced digits only |
| EEG | 6 channels (F3 F4 C3 C4 P3 P4), 1024 Hz | Emotiv Epoc, 14 channels, 128 Hz, written at the Coretto rates |
| Voice | 44.1 kHz | 16 kHz, written at 44.1 kHz |
| Source | authors' `.mat` files (DOI 10.1117/12.2255697); EEG re-hosted on Zenodo, record 19502780 | Mendeley Data, DOI 10.17632/s7chktmb6x.1, CC BY 4.0 |
| SQLite schema | `subject`, `modalities`, `stimulus`, `trial` (9315), `eeg_samples` (9315), `audio_samples` (1974: pronounced trials only) | `speaker` (51: 45 with `excluded = 0`, 6 flagged `excluded = 1`), `stimulus` (10), `pair` (6294 read + 560 stored for the excluded), `recording` (key/value provenance); format `thesis-paired-v2` |
| Read by | `ThesisDataset.cpp`, `dataset.source = "coretto2017"` | `ThesisDataset.cpp`, `dataset.source = "biomex2021"` |

The profile key `dataset.root` names the file (`<databases>/database.sqlite` or `<databases>/biomex.sqlite`).

## Coretto *et al.* (`<databases>/database.sqlite`)

**Obtain.** The authors distribute 30 `.mat` files, `SXX_EEG.mat` and `SXX_Audio.mat` for
S01..S15, through the link in their paper (a Google Drive folder that now asks for a sign-in).
The EEG half is also on Zenodo (record 19502780, files renamed `sub-NN_eeg.mat`); the audio is
only at the authors'.

**Convert.**

```bash
python3 scripts/data/import_mat_dataset_to_sqlite.py <folder with S01..S15> ~/Documentos/academico/UNESP/doutorado/databases/database.sqlite
```

Each `.mat` row is one trial: the six channels of 4096 samples each, concatenated, then three
labels (modality 1 = imagined / 2 = pronounced, stimulus 1–11, blink). The importer stores each
channel as a float32 BLOB.

**Verify** against the Zenodo EEG copy (download it to `<databases>/coretto_zenodo` first):

```bash
python3 scripts/data/coretto_verify_mirror.py        # --db and --mirror default to <databases>
```

It checks that every `.mat` row is in the SQLite exactly once, with equal channels, modality,
stimulus and blink label. Exit status 1 on any difference. On success it writes the stamp
`<databases>/database.sqlite.mirror-verified`: line 1 is the sha256 of the SQLite, line 2 the counts as
JSON (the run of record: 9315 / 9315 trials equal). **The audio cannot be checked this way**: it
is not in the mirror. Thesis ch. 8 says so.

## BIOMEX-DB (`<databases>/biomex.sqlite`)

```bash
python3 scripts/data/biomex_download.py                         # 51 RAR -> <databases>/biomex, keeps EEG + Voice only (needed only to regenerate)
python3 scripts/data/biomex_inventory.py [--csv inventory.csv]  # reads <databases>/biomex/extracted
python3 scripts/data/biomex_to_sqlite.py                        # -> <databases>/biomex.sqlite
python3 scripts/data/biomex_verify.py                           # checks it against <databases>/biomex/extracted
# then delete <databases>/biomex/: the SQLite and its stamp are what the thesis uses
```

| Step | What it does | Fails when |
|---|---|---|
| download | fetches each volunteer's archive, checks its SHA-256 against the listing, extracts `EEG/` and `Voice/`, deletes the archive (`--keep-archives` keeps it); resumes | a hash differs |
| inventory | lists, per volunteer, every EDF and voice file and whether it is the complete protocol | — (a report: read it before changing any rule) |
| convert | pairs each spoken digit's voice window with the EEG epoch recorded while it was spoken; applies the rules below | it cannot pair a digit for certain |
| verify | re-derives every pair by a second path and checks the EDF against the CSV export and the paper's Table 1 | any difference (exit 1); writes `<databases>/biomex.sqlite.verified` |

**The rules of record (chosen on the whole base, before any result):**

- **Strict recording rule.** A volunteer is kept only with one complete `01G10` and one
  complete `01G04` session under their own name. No per-file exceptions. That excludes F007,
  M001, M007, M010, M018 and M024, which leaves 45 volunteers. The excluded six are written to
  the base (`speaker.excluded = 1`) and never processed; see the table above.
- **Edge pairs are dropped.** The EEG goes through the Coretto authors' own FIR filter. A pair
  whose EEG epoch lies within the filter's reach (788 samples) of a session edge cannot be
  filtered like the others, so it is dropped (M015 and M019, one digit each).
- **Voices WORLD cannot analyse are dropped (2026-11-16).** Every WORLD dysphonia profile starts
  with Harvest's F0 contour; a voice with no voiced frame has no pitch to edit and is refused (no
  fallback). Three windows do that: M006 01G10 series 1 digit 1 (a faint hum, no word) and two
  very short M014 "seis". The user decided they leave the base: `UNANALYSABLE_VOICES` in
  `biomex_to_sqlite.py` (a stale entry is refused); `biomex_verify.py` rebuilds each from the
  originals and requires the WORLD path (`scripts/data/world_voicing.py`) to refuse it and to
  accept every stored voice. Result: 6294 pairs = 45 × 140 − 6 (1 unlabelled digit, 2 edge, 3
  unanalysable).
- Both signals are written at the Coretto rates (44.1 kHz voice, 1024 Hz EEG), so the same
  extractors read both bases.

Changing a rule means a new base: re-convert, re-verify, re-run the BIOMEX profiles, and
update ch. 8 (which states these rules) — never patch the SQLite by hand.

## The stamps and the facts the thesis quotes

The stamp files are how a later step knows the SQLite it reads is the one that was verified:

```
<databases>/biomex.sqlite.verified
  line 1: 2a4aaf29…   sha256 of biomex.sqlite when it passed
  line 2: {"volunteers": 45, "excluded": [...], "pairs_equal": 6294, "differences": 0,
           "edge_dropped": [...], "unanalysable": [3 windows], "world_refused_stored": 0, ...}
```

`scripts/analysis/thesis_base_facts.py` reads both bases **and** both stamps:

```bash
python3 scripts/analysis/thesis_base_facts.py --out results/thesis/base_facts.csv   # both bases from <databases>
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
`database.sqlite` and its two caches to the same paths under the cluster's home
(`~/Documentos/academico/UNESP/doutorado/databases/`, created if missing), and `biomex.sqlite`
when it exists. It **refuses** to send a `biomex.sqlite` that has no stamp or whose sha256 is no
longer the stamped one, so an unverified base cannot reach a cluster run. See [Experiment05 › Running on GridUnesp](../Experiments/Thesis.md#running-on-gridunesp-2026-11-12).

## Other bases in the repository

The Meeting01 paper uses FSDD, AudioMNIST, EEGMMIDB and Siena, fetched by
`scripts/pipeline/meeting01/ensure_datasets.sh` into the same `<databases>` directory (see
[Meeting01](../Experiments/Meeting01.md)).
They are unrelated to the thesis.

## Related

- [Experiment05](../Experiments/Thesis.md) §11 — why a second base, and what it replicates
- [Imagined Speech and EEG](../Concepts/Imagined-Speech-and-EEG.md) — the Coretto protocol
- [DataLoaders](../Core/DataLoaders.md) — the C++ loaders
