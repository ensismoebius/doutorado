# Experiment04 — LSTM Autoencoder for 1-D Temporal Signals

## Purpose

Experiment04 implements a **LSTM-based Autoencoder** using the project's internal
neural-network framework. It is designed as a direct experimental peer to the SNN
autoencoders in Experiment03, with identical loss function, optimizer, and logging
conventions so that reconstruction quality can be compared across architectures.

---

## Architecture

```
Input  [T × D]
   │
   ▼
 ┌─────────────────────────────────────────────┐
 │  Encoder                                    │
 │  LSTM_enc × num_layers  [T × D] → [T × H]  │
 │  Take last hidden state          [1 × H]    │
 │  Linear projection               [1 × H→Z]  │
 │  tanh                            [1 × Z]    │
 └─────────────────────────────────────────────┘
   │ z  [1 × Z]
   ▼
 ┌─────────────────────────────────────────────┐
 │  Decoder                                    │
 │  Linear expansion        [1 × Z→H]          │
 │  Repeat T times          [T × H]            │
 │  LSTM_dec × num_layers   [T × H] → [T × H]  │
 │  Linear output proj      [T × H→D]          │
 └─────────────────────────────────────────────┘
   │
   ▼
Reconstruction  [T × D]

Loss: MSE(reconstruction, input)
```

### LSTM gate equations (per time step t)

```
i_t = σ(W_i · x_t + U_i · h_{t-1} + b_i)     input gate
f_t = σ(W_f · x_t + U_f · h_{t-1} + b_f)     forget gate  (bias init = 1)
o_t = σ(W_o · x_t + U_o · h_{t-1} + b_o)     output gate
g_t = tanh(W_g · x_t + U_g · h_{t-1} + b_g)  candidate cell

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

All four input weight matrices are **stacked** into one `W [4H × D]` and all four
recurrent matrices into `U [4H × H]` — reducing allocations in the time loop to a
single matmul per step.

---

## Files

| Path | Description |
|---|---|
| `lib/include/LSTMLayer.hpp` | Single LSTM cell with full BPTT; stacked gate weights |
| `lib/include/LSTMAutoencoder.hpp` | Encoder/decoder architecture declaration |
| `lib/src/LSTMAutoencoder.cpp` | Forward and BPTT backward implementation |
| `lib/include/Experiment04Config.hpp` | Flat runtime configuration struct |
| `lib/include/Trainer.hpp` | Training loop (Adam, MSE, gradient clipping, logging) |
| `experiment04.cpp` | Thin CLI launcher |
| `profiles/default.json` | Default JSON profile |
| `tests/LSTMAutoencoder_gtest.cpp` | Unit + smoke tests |

---

## Build

Experiment04 is wired into the parent `src/experiments/CMakeLists.txt`. Build with
the standard project workflow:

```bash
cmake --preset <your-preset> -S .
cmake --build build/ --target experiment04
```

The only mandatory dependencies are **Eigen3**, **util**, and **nlohmann_json** —
all already present in the project. Real dataset loaders (`dataLoaders_10_1117`,
`dataLoaders_10_1117_windowing`) are linked automatically when available and enable
loading from the 10.1117 dataset.

---

## Running

```bash
# With synthetic data (no dataset required):
./experiment04

# With a JSON profile:
./experiment04 --config profiles/default.json

# Override key parameters on the command line:
./experiment04 --epochs 50 --lr 5e-4 --hidden-size 256 --latent-size 32

# Point to real dataset:
./experiment04 --dataset-root /path/to/10.1117/data
```

A JSON result file is written to `results_dir` (default: working directory):

```json
{
  "run_tag": "experiment04-default",
  "model": "lstm-autoencoder",
  "loss": "mse",
  "arch": { "input_size": 64, "seq_len": 32, "hidden_size": 128, "latent_size": 16 },
  "train_losses": [...],
  "val_losses":   [...],
  "exit_code": 0
}
```

---

## Comparison with Experiment03

| Property | Experiment03 | Experiment04 |
|---|---|---|
| Model family | ANN / SNN autoencoder | LSTM autoencoder |
| Loss | MSE | MSE |
| Optimizer | Adam (configurable) | Adam |
| Logging | `epoch=N  train_loss=X  val_loss=Y` | identical format |
| Result artifact | JSON | JSON (same schema) |
| Dataset | 10.1117 windowed | 10.1117 windowed or synthetic |
| Temporal modelling | feedforward / surrogate BPTT | true BPTT through time |

---

## Extension Points

Following the same convention as Experiment03:

- **New dataset variant** — add a `DatasetBuilder04` routing class and hook it
  into `experiment04.cpp`; the `LSTMAutoencoder` is agnostic to data source.
- **Additional LSTM layers** — increase `num_layers` in the profile; the
  encoder and decoder stack scales automatically.
- **Bidirectional encoder** — add a reverse-pass `LSTMLayer` and concatenate
  hidden states before the projection.
- **New optimizer** — swap the `Adam` in `Trainer.hpp` for any `Optimizer`
  subclass; the interface is identical.
- **Save/load weights** — `state_dict()` / `load_state_dict()` are fully
  implemented on both `LSTMLayer` and `LSTMAutoencoder`.
- **Regression tests** — add cases to `tests/LSTMAutoencoder_gtest.cpp`.

---

## Design Notes

### Memory efficiency

The stacked `W [4H × D]` / `U [4H × H]` layout avoids four separate matrix
multiplications per time step. Gate pre-activations are sliced out with `block()`
after a single matmul, keeping hot-path allocation minimal.

### Numerical stability

- Forget-gate bias is initialised to 1.0 (prevents vanishing cell gradients
  in early training).
- Latent codes are bounded to `(-1, 1)` by the encoder `tanh`.
- Gradient clipping (L2 global norm, configurable via `grad_clip_norm`) is
  applied before the optimizer step to prevent BPTT gradient explosion.
- Per-sample `reset_state()` is mandatory and called automatically by `Trainer`.

### Fixed-length assumption

Each input sample is a 2-D tensor `[T × D]` where `T = seq_len` is fixed per
experiment run. This matches the windowed dataset convention from Experiment03.
Variable-length support can be added by passing `T` at call-time to `decode()`,
which already accepts it as a parameter.
