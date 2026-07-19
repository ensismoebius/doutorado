# Voice and EEG identification project

Research workspace for the PhD project: biometric authentication of severely dysphonic
speakers, combining phonated voice with EEG-captured imagined speech. The repository holds
the thesis manuscript, the produced articles, exploratory notebooks, and the main C++20
neural-network framework (`nn`) that runs the experiments.

## Start here

| If you want to… | Go to |
|---|---|
| Understand or build the C++ framework | **[`software/nn/.wiki/Home.md`](software/nn/.wiki/Home.md)** — the full wiki |
| Build and run your first test | [Getting Started tutorial](software/nn/.wiki/Tutorials/Getting-Started.md) |
| Re-run the experiments | [Re-run Runbook](software/nn/.wiki/Guides/Re-run-Runbook.md) |
| Read the thesis | `documentation/00-thesis/monography/monografia.tex` |
| Read the conference paper | `documentation/07-articlesProduced/conference71070Guaiaquil/paper.tex` |
| Know why the code looks like it does | [Engineering Fixes Log](software/nn/.wiki/Guides/Engineering-Fixes-Log.md) |

## Repository map

- `documentation/` — thesis, produced articles, reading backlog, research notes.
- `notebooks/` — exploratory EEG/signal-processing notebooks.
- `results/` — top-level experiment outputs.
- `scripts/` — repository-level helper scripts.
- `software/` — implementation projects.

### Documentation

- `documentation/00-thesis/monography/` — main thesis LaTeX sources (`monografia.tex`).
- `documentation/00-thesis/presentation/` — presentation slides.
- `documentation/00-thesis/stateOfTheArtReview/` — literature review material.
- `documentation/01-researchNotesAndFiles/` — research notes and supporting files.
- `documentation/03-articlesToRead/` — papers and reading backlog.
- `documentation/06-BooksToRead/` — books backlog.
- `documentation/07-articlesProduced/conference71070Guaiaquil/` — conference paper package
  (`paper.tex`, figures, generated data).
- `documentation/07-articlesProduced/article/`, `.../spm-featurearticle-latex/` — other
  article assets.

### Software

- `software/nn/` — primary C++20 neural-network framework (SNN, LSTM, autoencoders).
  - `include/` — backend-agnostic public headers. Include as `"tensor/Tensor.hpp"`,
    `"layers/dense/Linear.hpp"`, … (there is **no** `include/nn/` directory).
  - `src/core/` — core implementation. Included as `"core/training/Trainer.hpp"`, ….
  - `src/experiments/` — experiments 00–05; Experiment 05 is the thesis's primary one.
  - `src/demos/` — standalone demos, including Python ones under `pyDemos/`.
  - `scripts/` — pipeline, testing, and analysis scripts.
  - `.wiki/` — **the project's real documentation.** Start at `.wiki/Home.md`.
- `software/signalAquirer/` — signal acquisition project.

### Data, results, and exploration

- `notebooks/` — exploratory analysis notebooks.
- `results/` — top-level experiment outputs. The thesis experiment writes to
  `software/nn/results/thesis/{phase00,phase01}/` instead.

## Release notes

Before packaging a release, verify the repository ships source material rather than editor or
build residue. Generated artifacts that may appear locally but are gitignored:

- LaTeX build products beside manuscripts: `*.aux`, `*.bbl`, `*.blg`, `*.fdb_latexmk`, `*.fls`,
  `*.idx`, `*.lof`, `*.lot`, `*.toc`.
- Notebook checkpoints under `.ipynb_checkpoints/`.
- Local development directories inside `software/nn/`: `.venv/`, `out/`, `__pycache__/`.

Use `git ls-files` to verify what is actually tracked before tagging a release.
