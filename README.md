# Voice and EEG identification project

Research workspace for the PhD project. This repository mixes manuscript sources, experiment outputs, hardware references, notebooks, and the main C++ neural-network codebase.

## Repository Map

- `documentation/` — manuscripts, notes, reading material, and produced articles.
- `hardware/` — circuit references and KiCad parts.
- `notebooks/` — exploratory notebooks and intermediate analysis outputs.
- `results/` — trained models and checkpoint artifacts.
- `software/` — implementation projects, including the main `nn` framework.

## Where To Find Things

### Documentation

- `documentation/00-dissertation/monography/` — main dissertation LaTeX sources.
- `documentation/00-dissertation/presentation/` — presentation slides.
- `documentation/00-dissertation/stateOfTheArtReview/` — literature review material.
- `documentation/01-researchNotesAndFiles/` — research notes and supporting files.
- `documentation/03-articlesToRead/` — papers and reading backlog.
- `documentation/06-BooksToRead/` — books backlog.
- `documentation/07-articlesProduced/article/` — article manuscript assets.
- `documentation/07-articlesProduced/conference71070Guaiaquil/` — conference paper package, including `paper.tex`, paper figures, and submission data.
- `documentation/07-articlesProduced/spm-featurearticle-latex/` — SPM feature article material.

### Software

- `software/nn/` — primary C++20 neural-network framework.
- `software/nn/include/` — public headers.
- `software/nn/src/` — core implementation, demos, experiments, and tests.
- `software/nn/scripts/` — automation and helper scripts.
- `software/nn/tools/` — support utilities.
- `software/sampleRateMeasurer/` — standalone Python tool.
- `software/signalAquirer/` — signal acquisition project.

### Data, Results, and Exploration

- `notebooks/` — exploratory analysis notebooks.
- `notebooks/outputs/` — notebook-generated outputs.
- `results/checkpoints/` — saved checkpoints.
- `results/guayaquil/models/` — exported or trained models.

### Hardware

- `hardware/INA128/` — instrumentation amplifier references.
- `hardware/kicadParts/` — reusable KiCad symbols and footprints.

## Quick Navigation

- Conference paper: `documentation/07-articlesProduced/conference71070Guaiaquil/paper.tex`
- Dissertation manuscript: `documentation/00-dissertation/monography/monografia.tex`
- Main C++ project: `software/nn/README.md`
- Main build entrypoint: `software/nn/CMakeLists.txt`
- Main results directory: `results/`

## Release Notes

Before packaging a release, review generated artifacts and local-only files so the repository ships source material rather than editor or build residue.

- LaTeX build products commonly appear beside manuscripts as `*.aux`, `*.bbl`, `*.blg`, `*.brf`, `*.fdb_latexmk`, `*.fls`, `*.idx`, `*.lof`, `*.lot`, and `*.toc`.
- Notebook checkpoint files may appear under `.ipynb_checkpoints/`.
- Local development directories inside `software/nn/` such as `.venv/`, `build-debug/`, `out/`, `__pycache__/`, and `.qtcreator/` should not be part of a release bundle.

Use `git ls-files` to verify what is actually tracked before tagging a release.