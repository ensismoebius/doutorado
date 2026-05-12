# GPU Saturation

## How to check

```bash
# Real-time AMD GPU busy % (updates every 0.5s)
watch -n 0.5 'cat /sys/class/drm/card*/device/gpu_busy_percent'

# Full breakdown: shader/memory/bus/VRAM (requires radeontop)
radeontop -d -
```

OpenCL kernel duty-cycle (already wired in codebase via `OpenCLProfiling.hpp`):
```
duty_cycle = sum(kernel_execution_ns) / wall_time_ns
```
If duty cycle < 30% the GPU is starved; if < 10% the bottleneck is CPU sync overhead.

---

## Why Renoir APU is unsaturated during training

| Cause | Numbers |
|---|---|
| Matmul kernel execution (batch=32, 256→64) | ~1 µs |
| `clFinish` + next kernel setup | ~10 µs |
| Resulting duty cycle (before fix) | **~9%** |

Each layer (`LinearImpl`, `ReLUImpl`, etc.) creates its own `BatchScope`. Because
`BatchScope` uses a flat bool (`s_batching`) with no nesting support, every layer
destructor fires `clFinish` — even layers inside a larger forward pass. A 6-layer
network stalls the GPU 6 times per sample.

---

## The fix: reference-counted BatchScope

Change `s_batching: bool` → `s_batch_depth: int`. `begin_batch` increments,
`end_batch` decrements — only the outermost scope fires `clFinish`.

Layer-level `BatchScope` instances (already in `LinearImpl::forward`) keep working
unchanged. Wrapping the full network forward pass in an additional `BatchScope`
suppresses all inner syncs automatically.

**Before (6 clFinish per forward pass):**
```
fc1::forward() → begin_batch → matmul, bias → end_batch → [SYNC]
relu::forward() → begin_batch → relu → end_batch             → [SYNC]
fc2::forward() → begin_batch → matmul, bias → end_batch → [SYNC]
```

**After (1 clFinish per forward pass):**
```
NetworkBatchScope → begin_batch (depth=1)
  fc1::forward()  → begin_batch (depth=2) → matmul, bias → end_batch (depth=1, no sync)
  relu::forward() → begin_batch (depth=2) → relu         → end_batch (depth=1, no sync)
  fc2::forward()  → begin_batch (depth=2) → matmul, bias → end_batch (depth=1, no sync)
NetworkBatchScope destructor → end_batch (depth=0) → [ONE SYNC]
```

### Implementation

`OpenCLContext`: change `s_batching: bool` → `s_batch_depth: int` (thread_local or
atomic for thread safety). `begin_batch()` → `++depth`. `end_batch()` →
`if (--depth == 0) flush()`. `is_batching()` → `depth > 0`.

`Trainer`: wrap `forward_pass()` and `backward_pass()` each in a `BatchScope`.

### Confirmed gain (measured 2026-05-12)

Fix is fully wired end-to-end:
- `OpenCLContext`: `s_batching: bool` → `s_batch_depth: int` (ref-counted)
- `Trainer::fit_loop` + `fit_loop_supervised`: each mini-batch wrapped in `BatchScope`
- Build guard: `#if defined(NN_BACKEND_OPENCL)` — CPU builds unaffected

User-reported: **"made a huge difference"** after first run with the fix.

Theoretical estimate: 6-layer network → 6 × ~10 µs = 60 µs sync overhead per sample
reduced to 1 × ~10 µs. Net savings ~50 µs/sample; ~1.6 ms/batch at batch=32.
GPU busy % expected: ~9% → ~40–60% (residual limit is compute time, not sync stalls).

---

## Residual bottlenecks after the fix

Even with ref-counted BatchScope, saturation is bounded by:

1. **Host→Device data copy per sample** — `clEnqueueWriteBuffer` before each kernel
   for non-GPU-resident tensors. Fix: keep weights GPU-resident (already implemented
   via `set_gpu_resident(true)`); keep activations in the GPU buffer pool.

2. **Small work size per matmul** — 32×256 matrix → 8192 floats, 2048 work-items.
   Renoir has 7 CUs × 64 SIMD lanes × 4 SIMD width = 1792 concurrent FP32 ops.
   Occupancy is ~100% per CU for these sizes, but only 1–2 CUs are needed.
   Fix: larger batch size (128–256) or wider layers to engage all 7 CUs.

3. **LIF neuron kernel** — `lif_step_kernel` is stateful and sequential over time steps.
   Not fusible. The SNN time-step loop is inherently serial.

---

## Further reading

- [GPU Kernel Fusion](GPU-Kernel-Fusion.md) — fusing matmul+activation into one kernel
  (reduces kernel count; complementary to this fix)
- [LSTM Performance](LSTM-Performance.md) — LSTM-specific bottlenecks
