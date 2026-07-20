# Tutorial 1 — Getting Started

**Goal:** go from a fresh checkout to a compiled library, a passing test, and one real
experiment run — in about 20 minutes, most of it waiting on the compiler.

**You need:** a C++20 compiler (clang preferred), CMake ≥ 3.20, Ninja, and Python 3.
Everything else the build fetches or vendors itself.

Every command below is run from `software/nn/`, and every one has been executed as written.

---

## Step 1 — Configure the build

The project ships CMake *presets*, so you never have to remember compiler flags:

```bash
cd software/nn
cmake --preset=max-performance
```

`max-performance` is the CPU (XTensor) backend. It is the reference backend: the thesis and
the conference paper both report numbers from it, and it is the right default unless you are
specifically working on GPU code. Configuring takes a minute or two the first time.

> **Other presets exist** (`max-performance-opencl`, `max-performance-sycl`, …) but they
> require a working GPU stack, and the project deliberately refuses to configure them if that
> stack is missing rather than silently falling back to CPU. Stick with `max-performance`
> until you have a reason not to.

## Step 2 — Build

```bash
cmake --build --preset=max-performance -j$(nproc)
```

This is the slow step — a heavily templated C++20 codebase. Expect several minutes on a first
build; later builds are incremental and fast.

## Step 3 — Run your first test

```bash
ctest --test-dir out/build/max-performance -R TensorOpsTest --output-on-failure
```

Expected: **4 tests passed**.

### The one thing that will trip you up

`-R` filters on the **GoogleTest name** (`SuiteName.TestName`), *not* the CMake target name.
So this looks reasonable and silently does nothing:

```bash
ctest --test-dir out/build/max-performance -R tensor_gtest   # <- "No tests were found!!!"
```

`tensor_gtest` is the *binary*; `TensorOpsTest` is the *suite*. When in doubt, list what is
actually registered and grep it:

```bash
ctest --test-dir out/build/max-performance -N | grep -i tensor
#   Test #1015: TensorOpsTest.Creation      <- this is what -R wants
```

Two more gotchas worth knowing now:

- **`-R` is case-sensitive.** `-R lif` matches nothing; `-R Lif` matches 37 tests.
- **`ctest --preset=...` does not work here.** `CMakePresets.json` defines configure and build
  presets but no *test* presets, so always use `--test-dir out/build/<preset>`.

To run everything (a few minutes):

```bash
ctest --test-dir out/build/max-performance --output-on-failure -j4
```

## Step 4 — Run a real experiment

Experiment 05 is the thesis's primary experiment. Every run is driven by a JSON *profile* —
no command-line hyperparameters, so a run is always reproducible from a file you can commit.

Start with a **smoke** profile: same code paths as the real thing, tiny parameters (60
samples, 2 epochs), so it finishes in seconds instead of hours:

```bash
cmake --build --preset=max-performance --target thesis -j$(nproc)

./out/build/max-performance/src/experiments/thesis/thesis \
  --config src/experiments/thesis/profiles/smoke/phase00/p00_ae_ann_base_eeg.json
```

You should see something like:

```
[Thesis] Loaded 60 samples from 15 subjects, 11 stimuli.
[Thesis] Extracted 1 feature set(s).
[Thesis] Paraconsistent ranking:
  autoencoder-ann alpha=0.812447 beta=0.871317 D_truth=1.26045
[Thesis] classifier.enabled=false — Phase 00 run, stopping after ranking.
[Thesis] Done. Results written to results/thesis/smoke
```

Read that output as follows:

- **alpha (α)** — how *tightly grouped* each speaker's feature vectors are. Higher is better.
- **beta (β)** — how much *different speakers overlap*. Lower is better.
- **D_truth** — a single score combining the two: the distance to the ideal "Truth" corner of
  the paraconsistent plane. **Lower is better.**
- **`classifier.enabled=false`** — this is a Phase 00 run. It only *scores* the features and
  stops; it never trains a classifier. That is by design (see below).

> Smoke *numbers* are meaningless — 2 epochs on 60 samples. Only the plumbing is being tested.
> Clean up with `rm -rf results/thesis/smoke` when done.

## Step 5 — Understand what you just ran

Experiment 05 runs in **two phases**, and keeping them separate is the core design idea:

| Phase | Question it answers | Trains a classifier? |
|---|---|---|
| **Phase 00** | Which feature extractor best separates speakers? | No — scores features only |
| **Phase 01** | How well does authentication actually perform? | Yes — trains the DSNN |

Phase 00 sweeps 208 profiles (wavelets × frequency scales × autoencoders) and scores each with
the paraconsistent metric. Only the single winner is passed to Phase 01. Separating them means
the choice of features is never contaminated by how good the classifier happens to be.

The switch between phases is one field in the profile: `classifier.enabled`.

## Where to go next

| Next step | Page |
|---|---|
| Understand the tensor type everything is built on | [Tensor](../Core/Tensor.md) · [plain-language version](../Core/Plain/Tensor.md) |
| Add your own layer to the framework | [Tutorial 2 — Adding a Layer](./Adding-a-Layer.md) |
| Run the full experiment (hours, not seconds) | [Running Experiment05 Profiles](../Guides/Running-Thesis-Profiles.md) |
| Understand the α/β/D_truth scoring you just saw | [Paraconsistent](../Core/Plain/Paraconsistent.md) |
| Understand spiking neurons | [SNN and Surrogate Gradients](../Concepts/Plain/SNN-and-Surrogate-Gradients.md) |
| Build options, sanitisers, other presets | [Build System](../Guides/Build-System.md) |
