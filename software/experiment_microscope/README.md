# Experiment Microscope

Research-grade interactive scientific inspection GUI for the two research
pipelines in this repository:

- **meeting01** — `software/nn/src/experiments/meeting01/` (nested-LOSO SNN vs
  LSTM/GRU/Transformer autoencoders on FSDD / AudioMNIST / MIT-BIH)
- **thesis** — `software/nn/src/experiments/thesis/` (handcrafted wavelet +
  paraconsistent feature selection + DSNN authentication)

The full specification is `FIXME.md` in this directory.

## Running

```bash
./run.sh
```

This bootstraps a `--system-site-packages` venv (`.venv/`) and launches the app.

## The C++ binding

Live recomputation of pipeline intermediates ("why did this number become this
number?") goes through `nn_microscope`, a pybind11 module that links the actual
`meeting01` / `thesis` C++ libraries — never a second Python reimplementation.

Build it once:

```bash
cd ../nn
cmake --preset=python-bindings
cmake --build out/build/python-bindings --target nn_microscope
```

Without it the app still opens and browses persisted artifacts, but any view
that needs a recomputed intermediate raises with the build command above.

## Tests

```bash
./.venv/bin/pytest -q
```
