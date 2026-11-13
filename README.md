# Voice and EEG biometrics — PhD workspace

Research workspace of the PhD thesis *Autenticação Biométrica de Locutores Drasticamente
Disfônicos Aprimorada pela Imagined Speech* (UNESP): can EEG of imagined or pronounced
speech help authenticate a speaker whose voice is badly degraded? (The dysphonia in the
experiments is simulated.) The repository holds the thesis text, the papers, and the C++20
framework (`software/nn`) that runs every experiment, with the results of record.

Everything needed to build, run, change or write any part of the project is documented here
and in the wiki. Start with the table below.

## I want to…

| Task | Read |
|---|---|
| build the framework and run a test | [Getting Started](software/nn/.wiki/Tutorials/Getting-Started.md) → [Build System](software/nn/.wiki/Guides/Build-System.md) |
| find **where** to change something | [How to Change Anything](software/nn/.wiki/Guides/How-To-Change-Anything.md) — one table, task → files → tests → docs |
| understand the research | [Research Context](software/nn/.wiki/Research-Context.md) → [Experiment05 (thesis)](software/nn/.wiki/Experiments/Thesis.md) |
| run the thesis experiment | [Running Thesis Profiles](software/nn/.wiki/Guides/Running-Thesis-Profiles.md) |
| build or verify the data bases | [Data Bases](software/nn/.wiki/Guides/Data-Bases.md) |
| edit the thesis text, tables or figures | [Thesis Document](software/nn/.wiki/Guides/Thesis-Document.md) |
| work on the conference paper (Meeting01) | [Meeting01](software/nn/.wiki/Experiments/Meeting01.md) → [GridUnesp Deployment](software/nn/.wiki/Guides/GridUnesp-Deployment.md) |
| use the cluster (GridUnesp) | [GridUnesp Deployment](software/nn/.wiki/Guides/GridUnesp-Deployment.md), [Experiment05 › Running on GridUnesp](software/nn/.wiki/Experiments/Thesis.md#running-on-gridunesp-2026-11-12) |
| add a layer to the framework | [Adding a Layer](software/nn/.wiki/Tutorials/Adding-a-Layer.md) |
| find a script | [`software/nn/scripts/README.md`](software/nn/scripts/README.md) |
| know why the code is the way it is | [Engineering Fixes Log](software/nn/.wiki/Guides/Engineering-Fixes-Log.md) |
| see everything | the wiki home, [`software/nn/.wiki/Home.md`](software/nn/.wiki/Home.md) |

## Repository map

```
README.md                  this file
AGENTS.md, CLAUDE.md       instructions for coding agents (the human docs are the wiki)
.claude/, .github/skills/  agent slash-commands, skills and hooks (synced by scripts/dev/sync_skills.sh)
.github/workflows/ci.yml   CI: build, cppcheck, clang-tidy, ctest, the Python test suites
documentation/             the thesis, papers, presentations, reading material  → documentation/README.md
notebooks/                 Jupyter prototypes (audio features, neuron maths)    → notebooks/README.md
software/
  nn/                      the C++20 framework, experiments, scripts, results, wiki  → software/nn/README.md
  experiment_microscope/   PySide6 GUI that recomputes any pipeline number from the raw signal
  efficient_nn_lab/        interactive demos for the BitNet/SNN lecture
```

Folder guides: [`documentation/`](documentation/README.md), [`notebooks/`](notebooks/README.md),
[`software/nn/`](software/nn/README.md), [`software/experiment_microscope/`](software/experiment_microscope/README.md),
[`software/efficient_nn_lab/`](software/efficient_nn_lab/README.md).

The experiment outputs live in `software/nn/results/` (the thesis's in
`software/nn/results/thesis/`, committed with the trained models). There is no top-level
`results/` folder.

## Conventions that hold everywhere

- **No defaults, no fallbacks, no silent skips.** A profile declares every setting; a script
  takes every argument explicitly; anything missing or unreadable is an error, never a
  guess. This is what keeps a wrong run from looking like a valid one.
- **No hand-typed result in a paper.** Tables and numbers are generated from the committed
  results ([Thesis Document](software/nn/.wiki/Guides/Thesis-Document.md)).
- **Code, commits and the wiki are in English**; the thesis text is in Portuguese.
- Long experiments (hours to weeks) are never started casually: each one says so at the top,
  and the Meeting01 grid refuses to start without `EXPERIMENT_CONFIRMED=1`.

## What git does not hold

`git ls-files` is the truth about what is tracked. These exist locally and are ignored:
LaTeX build products (`*.aux`, `*.bbl`, `*.log`, …), `.ipynb_checkpoints/`, `*.wav`, and in
`software/nn/`: `.venv/`, `out/` (build trees), `__pycache__/`, the generated
`include/layers/Layers.hpp`. The data bases (`~/database.sqlite`, `~/biomex.sqlite`) are
outside the repository; [Data Bases](software/nn/.wiki/Guides/Data-Bases.md) says how to
rebuild them.
