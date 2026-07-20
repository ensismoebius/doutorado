# LSTM Performance Guide

Profiling results and optimization history for `LSTMLayerImpl<Backend>`.

## Microbenchmark Setup

File: `src/experiments/guayaquil/tests/lstm_ops_microbench.cpp`
Config: B=1, D=128 (audio features), H=32 (hidden), T=256 (time steps), 100 reps.

Build and run:
```bash
cmake --build out/build/max-performance --target lstm_ops_microbench -j$(nproc)
./out/build/max-performance/src/experiments/guayaquil/tests/lstm_ops_microbench
```

## Profiling Results

### Before Optimizations

| Operation | Time (ms) | % of Full Step |
|---|---|---|
| `x_t @ W^T` matmul (1,128)×(128,128) | 0.036 | 50.6% |
| `h @ U^T` matmul (1,32)×(32,128) | 0.003 | 4.1% |
| `block()` + 4 gate activations (combined) | 0.031 | **43.7%** |
| Individual sigmoid or tanh on (1,32) | 0.0003 | 0.44% each |
| **Full timestep** | **0.1436** | 100% |

### After Optimization (fused block+activation)

| Operation | Time (ms) | % of Full Step |
|---|---|---|
| `x_t @ W^T` matmul | 0.036 | ~71% |
| 4 gates (fused) | 0.0016 | **1.1%** |
| **Full timestep** | **0.0504** | — |

**Speedup: 2.85× per timestep** from eliminating `block()` intermediate copies.

## Optimization: Fused Block+Activation

### Root Cause

`block()` on `XTensorBackend` creates a **new tensor** (allocates memory + copies data). Calling it 4 times per timestep (once per gate) costs 43.7% of the full step while the sigmoid/tanh math itself is only 4 × 0.44% = 1.76%.

### Fix Applied

Added `sigmoid_fast_block` and `tanh_fast_block` to `include/layers/activations/FastActivations.hpp`:

```cpp
// Fused: reads src[:,col_start:col_start+gate_size] directly, applies activation, no intermediate copy.
inline auto sigmoid_fast_block(const nn::Tensor& src, nn::Index col_start, nn::Index gate_size) -> nn::Tensor;
inline auto tanh_fast_block(const nn::Tensor& src, nn::Index col_start, nn::Index gate_size) -> nn::Tensor;
```

Forward pass changed from:
```cpp
Tensor i_g = sigmoid_fast_tensor(pre.block(0, 0 * H, B, H));   // alloc + copy + activate
Tensor f_g = sigmoid_fast_tensor(pre.block(0, 1 * H, B, H));
Tensor o_g = sigmoid_fast_tensor(pre.block(0, 2 * H, B, H));
Tensor g_g = tanh_fast_tensor(pre.block(0, 3 * H, B, H));
```
To:
```cpp
Tensor i_g = nn::activations::sigmoid_fast_block(pre, 0 * H, H);  // single pass, no intermediate
Tensor f_g = nn::activations::sigmoid_fast_block(pre, 1 * H, H);
Tensor o_g = nn::activations::sigmoid_fast_block(pre, 2 * H, H);
Tensor g_g = nn::activations::tanh_fast_block(pre, 3 * H, H);
```

## Fast Activation Approximations

Also applied: replaced `exp()`-based standard sigmoid/tanh with rational approximations (error < 0.01):

| Standard | Fast | FLOPs saved |
|---|---|---|
| `1 / (1 + exp(-x))` | `0.5 + x / (2*(1+\|x\|))` | avoids `exp()` (~50–100 cycles/call) |
| `(exp(2x)-1)/(exp(2x)+1)` | `x / (1 + \|x\|)` | avoids `exp()` |

Input clipped to [-8, 8] to prevent overflow without complex branching.

## Remaining Bottleneck

After fused-block fix, `x_t @ W^T` (BLAS `sgemm`) dominates at ~71% of full timestep. This is the theoretical floor for (1,128)×(128,128) GEMM at this batch size.

**Potential next optimization:** Concatenate `x_t` and `h` along the feature axis and perform one matmul against `[W | U]^T` instead of two separate calls:

```cpp
// Instead of:
auto pre = x_t.matmul_transposed(W_).add(h.matmul_transposed(U_));

// Single BLAS call (requires pre-stacking W_ and U_ horizontally at construction):
Tensor xh = concat_cols(x_t, h);   // (B, D+H)
Tensor WU = concat_rows(W_, U_);    // (4H, D+H)
auto pre = xh.matmul_transposed(WU);
```

Saves one BLAS dispatch overhead per timestep. Tradeoff: extra memory for `WU` (4H × (D+H)) and parameter update complexity. Not yet implemented.

## Experiment Benchmark

Profile `src/experiments/guayaquil/profiles/lstm-bench.json`:

| Setting | Value |
|---|---|
| Dataset | FSDD (32 train, 8 val samples) |
| Window size | 256 time steps |
| Encodings | direct only |
| Repeats | 1 |
| Epochs | 3 |
| Purpose | Isolate LSTM timing |

Run:
```bash
./out/build/max-performance/src/experiments/guayaquil/guayaquil \
  --profile src/experiments/guayaquil/profiles/lstm-bench.json
```

Wall time with fused-block optimizations: **~29s** (1 run, 3 epochs, 32 samples, T=256, 12 threads).

## Input Framing: The Largest Win (2026-07-18)

Every optimisation above shaves the cost of *one* timestep. Framing removes
timesteps outright, and it dwarfs all of them.

Guayaquil previously built the LSTM autoencoder with `input_size = 1` and
`seq_len = window_size`, so a 256-sample window was consumed **one scalar per
timestep**. The dominant cost per step is not the input term but the recurrent
one, $h \cdot U^\top$ with $U$ of shape $(4H, H)$ — 16 384 MACs for $H = 64$ —
and it was paid 256 times.

`model.lstm_frame_size` (default **8**) groups that many consecutive samples into
each timestep. No information is discarded; the window is merely re-blocked.

| | frame=1 (old) | frame=8 |
|---|---|---|
| sequential steps $T$ | 256 | 32 |
| input dim $D$ | 1 | 8 |
| total MACs | 8 523 840 | 1 184 256 |
| CPU LSTM train | 3 711 ms | **478 ms** |
| OpenCL LSTM train | 144 848 ms | **18 335 ms** |

~7× fewer MACs and ~8× less sequential depth, on **both** backends. On the
OpenCL backend it matters even more, because each timestep costs several kernel
enqueues at ~95 µs apiece.

`lstm_frame_size` must divide `dataset.window_size` (validated in
`GuayaquilConfig::validate`). Set it to `1` to reproduce the original behaviour.

### Framing is not a plain reshape

Storage is column-major, so reshaping `(256, 1)` to `(32, 8)` would place samples
$\{t, t+32, t+64, \dots\}$ in frame $t$ — a polyphase split, not framing.
`to_lstm_frames()` reshapes to `(frame, T)` and transposes, giving
`element(t,d) = sample[t*frame + d]`:

```cpp
// src/experiments/guayaquil/lib/src/GuayaquilEncoding.cpp
Tensor d_major = sample;
d_major.reshape({frame, steps});   // element (d,t) = sample[t*frame + d]
return d_major.transpose();        // -> (steps, frame)
```

### Scientific note

This changes the LSTM architecture and therefore the paper's LSTM-AE results. It
is arguably a *fairer* baseline: the SNN-AE sees all 256 samples at once through
`linear:64`, while the old LSTM saw one scalar at a time across 256 steps — a
regime where LSTMs are known to struggle with long-range dependencies
[Bengio, 1994].

## Rule: Never Use `block()` in Hot Loops

> `block()` on `XTensorBackend` always allocates and copies. Use fused alternatives from `FastActivations.hpp` when reading a column range and applying an element-wise function in the same step.

## See Also

- [LSTM-and-BPTT](../Concepts/LSTM-and-BPTT.md) — equations, BPTT derivation, pitfalls
- [Layers](../Core/Layers.md) — FastActivations API reference
- [PGO](PGO.md) — profile-guided optimization workflow
- [Experiment04](../Experiments/Guayaquil.md) — where `lstm_frame_size` is configured
- [OpenCL Debugging and Performance](./OpenCL-Debugging-And-Performance.md) — why per-timestep enqueues cost so much on the GPU backend

## References

[1] Y. Bengio, P. Simard, and P. Frasconi, "Learning long-term dependencies with gradient descent is difficult," *IEEE Trans. Neural Netw.*, vol. 5, no. 2, pp. 157–166, Mar. 1994. [Online]. Available: https://doi.org/10.1109/72.279181

[2] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997. [Online]. Available: https://doi.org/10.1162/neco.1997.9.8.1735
