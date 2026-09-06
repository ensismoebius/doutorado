---
name: memory-constrained-design
description: "Constrain model and pipeline choices for low-memory targets (e.g., Raspberry Pi class devices)."
---

# memory-constrained-design

Goal
- Keep experiments inside hard RAM budgets.

Rules
- RULE: HARD_BUDGET
  DO: Treat memory budget as a hard constraint.
  AVOID: Optimizing first and budgeting later.
- RULE: SMALL_DEFAULTS
  DO: Use `batch_size <= 8`, `latent <= 32`, `depth <= 2` unless justified.
  AVOID: Large default model shapes.
- RULE: DTYPE_DISCIPLINE
  DO: Prefer `float32` for training/inference unless required otherwise.
  AVOID: Higher precision defaults on constrained devices.
- RULE: TENSOR_CAP
  DO: Keep single tensor allocations under practical limits (target: <10MB).
  AVOID: Silent large allocation spikes.

Validation
- Config declares budget-sensitive parameters explicitly.
- Memory-heavy operations are justified and measured.

Project Context (nn framework)
**Target hardware:** AMD Renoir APU — 7 compute units, 64 KiB LDS per CU, 4 GB RAM shared between CPU and GPU. No discrete VRAM; all OpenCL buffers come from this pool.

**SNN memory cost:** `O(T × B × F)` per history buffer (`v_mem_history`, `spike_history`). For T=10, B=32, F=64: 20,480 floats = 80 KB per layer.

**Batch size choice:** `batch_size=32` chosen to keep all layer buffers + weights within GPU buffer pool. Larger batches (128+) engage more CUs but may exhaust the 4 GB shared pool.

**`set_gpu_resident(true)`** — keeps weight tensors in the GPU buffer pool between batches. Avoids repeated host→device copies. Set for all Linear layers in Exp04 by default.
