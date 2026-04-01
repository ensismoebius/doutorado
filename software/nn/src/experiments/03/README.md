# experiment03 - Autoencoder training runner

Overview
--------
`experiment03` trains ANN or SNN autoencoders against the 10.1117 imagined-speech EEG+Audio dataset.
It supports two layers of configuration:

- A launcher-level `default_config` in `src/experiments/03/experiment03.cpp`.
- A profile JSON loaded from `src/experiments/03/profiles/*.json`.
- Explicit CLI flags, which override both the launcher defaults and the selected profile.

Configuration precedence
------------------------
Runtime configuration is resolved in this order:

1. `default_config` in `src/experiments/03/experiment03.cpp`
2. `--profile <name>` or `--profile <path/to/file.json>`
3. Explicit CLI flags such as `--batch-size`, `--epochs`, or `--ae-latent-size`

This means the experiment is configurable enough for multi-batch and multi-epoch training already, but the previous documentation was incomplete and one CLI contract was misleading.

Use `--help` to inspect the full option list without starting the experiment or touching the logger pipeline.

Important behavior fixes
------------------------
- `--epochs N` controls how many full training passes are executed.
- `--batch-size N` controls how many samples are packed into each optimization step.
- `--max-batches N` limits batches per epoch.
- `--max-batches 0` means: do not cap the epoch, iterate the full dataset.

The runtime already supported `max_batches_per_epoch = 0`, but the CLI validator rejected `0`. That mismatch has now been fixed.

What you can tune
-----------------
Dataset selection and discovery:
- `--dataset-root <path>`: root folder containing subject directories.
- `--subject <regex>`: regex used to pick subject folders.
- `--dataset-type <protocol|eeg-window|audio-window|fused-window>`: which dataset representation to train on.
- `--input-mode <concatenated|eeg-only|audio-only>`: protocol-dataset modality selection.

Training loop controls:
- `--batch-size <N>`: samples per optimization step.
- `--epochs <N>`: number of epochs.
- `--max-batches <N>`: batches per epoch; use `0` for full-dataset epochs.
- `--lr <value>`: Adam learning rate.

Input pipeline controls:
- `--lookahead <N>`: background prefetch queue depth.
- `--prefetch-ram-cap-mb <N>`: RAM limit for prefetched batches.
- `--use-sqlite`: use the SQLite-backed batch source.

Sampling controls:
- `--shuffle` / `--no-shuffle`: legacy shuffle toggle.
- `--seed <N>`: deterministic shuffle seed.
- `--sampler-type <sequential|random|weighted|distributed>`: explicit sampler selection.
- `--sampler-weights <w1,w2,...>`: weights for weighted sampling.
- `--weighted-num-samples <N>`: samples drawn per epoch for weighted sampling.
- `--distributed-num-replicas <N>`: total distributed partitions.
- `--distributed-rank <N>`: current partition rank.
- `--distributed-shuffle` / `--distributed-no-shuffle`: distributed shuffling.
- `--distributed-drop-last` / `--distributed-no-drop-last`: divisibility behavior.

Autoencoder controls:
- `--autoencoder <token>`: ANN/SNN family to instantiate.
- `--ae-hidden-size <N>`: default dense hidden width.
- `--ae-latent-size <N>`: latent bottleneck width.
- `--ae-depth <N>`: encoder/decoder depth.
- `--ae-layer-sizes <a,b,c>`: explicit hidden-size schedule.
- `--ae-input-features <N>`: manual input feature override.
- `--ae-eeg-features <N>` and `--ae-audio-features <N>`: manual multimodal split overrides.
- `--ae-architecture <auto|residual-dense|dual-branch-fusion>`: architecture family.
- `--ae-branch-hidden-size <N>`: branch projection width.
- `--ae-fusion-hidden-size <N>`: shared fusion width.
- `--ae-residual-blocks <N>`: residual blocks per stage.
- `--ae-time-step <value>`: SNN time step.
- `--ae-resistance <value>`: SNN membrane resistance.
- `--ae-capacitance <value>`: SNN membrane capacitance.

Windowing controls:
- `--eeg-window-size <N>` and `--eeg-overlap <ratio>`: EEG window configuration.
- `--audio-window-size <N>` and `--audio-overlap <ratio>`: audio window configuration.

How to think about batches and epochs
-------------------------------------
If you want genuine training runs instead of capped smoke tests, the most important knobs are:

- `--epochs`: increase this above `1` for repeated training passes.
- `--batch-size`: controls optimization granularity and throughput.
- `--max-batches 0`: removes the artificial epoch cap and lets each epoch consume the full dataset.

Typical patterns:

Smoke test:
```bash
./src/experiments/03/experiment03 \
  --profile fused-window-ann-lightweight \
  --epochs 1 \
  --max-batches 2
```

Full training epoch over the dataset:
```bash
./src/experiments/03/experiment03 \
  --profile fused-window-ann-default \
  --batch-size 128 \
  --epochs 10 \
  --max-batches 0
```

Longer SNN training run:
```bash
./src/experiments/03/experiment03 \
  --profile fused-window-snn-default \
  --epochs 25 \
  --max-batches 0 \
  --ae-time-step 0.5 \
  --ae-resistance 1.0 \
  --ae-capacitance 1.0
```

Profiles
--------
Profiles live in `src/experiments/03/profiles/` and act as reusable starting points.

The profile loader accepts ordinary JSON and ignores unknown keys. That means you can document a profile inline by adding fields such as `_comment_profile` or `_comment_step_1`. This repo now includes a loadable example of that pattern in `sample-training-flow.json`.

Useful built-in examples:
- `default.json`: generic baseline defaults.
- `lightweight.json`: smaller/faster smoke-test baseline.
- `fused-window-ann-default.json`: fused ANN baseline.
- `fused-window-snn-default.json`: fused SNN baseline.
- `protocol-ann-default.json`: protocol ANN baseline.
- `protocol-snn-default.json`: protocol SNN baseline.
- `sample-training-flow.json`: commented, loadable example that shows how to move from smoke tests to full training.

Profiles may set:
- dataset type
- autoencoder type
- batch size
- epoch count
- learning rate
- window configs
- autoencoder widths/depth
- prefetch settings

CLI flags remain the final override, so a profile is a starting point, not a lock.

Sample commented profile flow
-----------------------------
Use `sample-training-flow.json` when you want a self-documented baseline.

Recommended progression:

1. Start with the profile as-is for a short smoke run.
2. Increase `--epochs` and set `--max-batches 0` once the pipeline is stable.
3. Increase `--batch-size` only after checking RAM and throughput.
4. Change dataset/autoencoder families after the training loop is already behaving correctly.

Example:
```bash
./src/experiments/03/experiment03 \
  --profile sample-training-flow \
  --epochs 10 \
  --max-batches 0 \
  --batch-size 128
```

What the launcher defaults currently imply
------------------------------------------
The launcher in `src/experiments/03/experiment03.cpp` currently defaults to:

- dataset root pointing at the local 10.1117 dataset path
- fused-window dataset mode
- fused-window ANN autoencoder
- batch size `100`
- epoch cap `100` batches
- SQLite-backed input pipeline enabled
- prefetch lookahead `20`

These defaults are reasonable for a local training workstation, but they are still only defaults. The intended way to run alternate studies is:

1. pick a profile close to the experiment you want
2. override only the few parameters you want to sweep on the CLI

Data layout expected by the runner
----------------------------------
The dataset root is expected to contain subject folders such as:

```text
dataset_root/
  S01/
    S01_EEG.mat
    S01_Audio.mat
  S02/
    S02_EEG.mat
    S02_Audio.mat
```

The protocol dataset path uses synchronized audio and EEG rows. Windowed dataset types derive sliding windows from those source signals using the configured EEG/audio window specs.

Operational notes
-----------------
- The prefetcher uses a single producer thread to avoid unsafe concurrent MAT I/O.
- Progress reporting uses the effective batch limit, so capped runs still report correct percentages.
- Run summaries are written as JSON and include profile, dataset type, autoencoder type, exit code, processed samples, seen batches, and epoch mean losses.

Troubleshooting
---------------
- If training ends too quickly, check whether `--max-batches` is still capped to a small number.
- If you expected multiple epochs, verify `--epochs` is greater than `1`.
- If no batches are produced, verify `--dataset-root`, `--subject`, and dataset integrity.
- If I/O becomes unstable during debugging, temporarily reduce `--lookahead` to `1`.

Code pointers
-------------
- Launcher defaults: `src/experiments/03/experiment03.cpp`
- CLI contract and `Config` fields: `src/experiments/03/lib/include/cli.hpp`
- CLI parsing and overrides: `src/experiments/03/lib/src/cli.cpp`
- Training loop: `src/experiments/03/lib/src/experiment03.cpp`
- Profile loading: `src/experiments/03/lib/src/ProfileLoader.cpp`
