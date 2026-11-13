# meeting01 — what does spiking cost an autoencoder? (was "Experiment04")

The conference paper's experiment. A spiking autoencoder (SNN-AE) is compared with its matched
ANN control (`ann-ae`: the same network with a leaky ReLU instead of the LIF neuron), PCA and the
mean frame, on FSDD, AudioMNIST, EEGMMIDB and Siena, under nested leave-one-group-out
cross-validation. Each family's architecture is chosen by a small GA inside the training groups.
The LSTM, GRU and Transformer autoencoders are still in the code, but were removed from the grid
on 2026-11-02.

**Everything else is in the wiki: [`.wiki/Experiments/Meeting01.md`](../../../.wiki/Experiments/Meeting01.md)**
(its first section is the current state). Cluster runs:
[GridUnesp Deployment](../../../.wiki/Guides/GridUnesp-Deployment.md).

## Build and run (from `software/nn`)

```bash
cmake --preset=max-performance
cmake --build --preset=max-performance --target meeting01 -j"$(nproc)"

# one (dataset, outer fold) slice
./out/build/max-performance/src/experiments/meeting01/meeting01 \
  --comparative-config src/experiments/meeting01/profiles/meeting01-loso.json \
  --dataset fsdd --cv-fold 0

# the whole grid + the paper's post-processing (weeks: run it on GridUnesp)
EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
```

`dataset.cv_fold` is required everywhere. The original pooled-and-shuffled split let one
speaker or recording land in both train and validation, the defect a reviewer flagged on
submission 71, and it no longer exists.

## Folder map

| Path | Role |
|---|---|
| `meeting01.cpp` | entry point |
| `lib/*/Meeting01Cli*`, `Meeting01Config.*` | command line and the profile schema |
| `lib/*/Meeting01Dataset*`, `Meeting01Eeg*`, `Meeting01EegBand.hpp`, `Meeting01MitBih.*` | the four bases (and MIT-BIH), windowing, group-disjoint splits |
| `lib/*/Meeting01Encoding.*` | spike encodings (poisson, latency, direct) |
| `lib/*/Meeting01Training.*`, `Meeting01Stopping.hpp`, `Meeting01EpochHistory.hpp`, `Meeting01BatchLossCollector.hpp` | training loop, the plateau-aware stopping rule |
| `lib/*/Meeting01Ga*`, `Meeting01RecurrentGa*`, `Meeting01TransformerGa*` | the per-family architecture search (NSGA-II core in `include/ga/Nsga2Core.hpp`) |
| `lib/*/Meeting01Evaluation.*`, `Meeting01Metrics.*`, `Meeting01PerWindow.hpp`, `Meeting01RunMetrics.hpp` | test metrics, per-window errors |
| `lib/*/Meeting01Energy.*` | the memory-aware energy model |
| `lib/*/Meeting01Output.*`, `Meeting01ResultRow.hpp`, `Meeting01Events*`, `Meeting01EventCallback.hpp`, `Meeting01EpochLogger.hpp` | result files and the live event stream the monitors read |
| `lib/*/Meeting01Runner.*`, `lib/src/Meeting01Experiment.cpp` | orchestration of one (dataset, fold) |
| `lib/*/Meeting01LstmAutoencoder.*` | the LSTM-AE wrapper (no longer in the grid) |
| `profiles/` | `meeting01-loso.json` (record), `meeting01-prevalidation.json`, `debug.json`, `lstm-*.json` |
| `tests/` | the GoogleTest suites below |
| `questionsAndTasks.md` | the working log of open questions (historical notes, not instructions) |

## Tests

Test executables: `meeting01_comparative_gtest`, `dat_writer_gtest`, `profile_audit_gtest`,
`loaders_gtest`, `event_sink_gtest`, `parameter_count_gtest`, `meeting01_encoding_gtest`,
`meeting01_split_audit_gtest`, `meeting01_ga_gtest`, `meeting01_recurrent_ga_gtest`,
`meeting01_transformer_ga_gtest` (`lstm_ops_microbench` is a benchmark, not a test). Build them
with `cmake --build --preset=max-performance -j"$(nproc)"` and run each binary from
`out/build/max-performance/src/experiments/meeting01/tests/`, or one suite through ctest:

```bash
ctest --test-dir out/build/max-performance -R 'SplitAudit' --output-on-failure
```

`ctest -R` filters by GoogleTest suite name, and the suites here have no common prefix
(`Meeting01*`, `SplitAudit`, `StoppingRule`, `ProfileAuditTest`, `MemoryAwareEnergy`, …), so
`-R Meeting01` silently skips most of them.
