---
description: "Validate backward pass tensor shapes, cache consistency, and gradient finiteness before optimizer step."
---

# gradient-flow-validator

Catch silent backward pass failures: shape mismatches, stale caches, and NaN/Inf gradients that corrupt weight updates.

## Project Context (nn framework)

**Layer cache patterns** — check these are populated before `backward()`:
- `CrossEntropyLoss`: caches softmax output from `forward()`
- `LeakyBPTT`: `spike_history` length must equal `time_steps`; `v_mem_history` same
- `LinearImpl`: caches input tensor for weight gradient computation

**Time-major grad invariant:** Gradient entering `LeakyBPTT::backward()` must be `(T*B, F)` — same shape as forward input. If BPTT history length ≠ `time_steps`, the off-by-one means backward reads past the allocated history.

**Consequence of wrong history length:** Silent gradient corruption — spike gradients from wrong timestep are applied, loss appears to decrease but model diverges on longer sequences.

**Wiki & knowledge graph:**
- Documentation at `.wiki/` — theory, guides, experiment pages, concept definitions
- Graph output at `.wiki/graphify-out/` — 1926 nodes, 4987 edges, 203 communities
- Find any symbol/concept:
```bash
python3 -c "
import json,sys
with open('.wiki/graphify-out/graph.json') as f: g=json.load(f)
q=sys.argv[1].lower()
for n in g['nodes']:
    if q in n['id'].lower() or q in n.get('label','').lower():
        print(n['id'],'|',n.get('source_file',''),'|',n.get('source_location',''))
" <QUERY>
```
- Workflow: `GRAPH_REPORT.md` → community → node → `source_file` → read → follow edges

## Rules

- **SHAPE_MATCH**: Gradient input to `backward()` must have the same shape as the output of the corresponding `forward()`. Assert or log mismatch explicitly. No silent shape mismatches.
- **CACHE_VALIDITY**: Caches used in `backward()` (softmax output, spike history, activation masks) must be validated as non-empty and matching current batch dimensions. No use of stale cache from a previous forward call.
- **FINITE_BEFORE_STEP**: Check all gradient tensors for NaN/Inf after `backward()` and before `optimizer.step()`. Log which layer produced the first non-finite gradient. No updating weights with invalid gradients.
- **NORM_LOGGING**: When `clip_grad_norm` is applied, log the pre-clip gradient norm per layer in DEBUG mode. No silent clipping that hides exploding gradients.
- **BPTT_HISTORY_LENGTH**: For BPTT/SNN backward passes, assert that history length equals the number of unrolled time steps declared in the config. No off-by-one unrolling.
- **FORWARD_BEFORE_BACKWARD**: Assert that `forward()` was called before `backward()` on each layer. No calling `backward()` on an un-initialized cache.

## Key Files to Audit

- [include/nn/layers/losses/CrossEntropyLoss.hpp](include/nn/layers/losses/CrossEntropyLoss.hpp) — `last_targets` cache vs batch size
- [include/nn/layers/spiking/LeakyBPTT.hpp](include/nn/layers/spiking/LeakyBPTT.hpp) — `v_post_history`, `v_mem_history` consistency
- [include/nn/layers/activations/LeakyReLU.hpp](include/nn/layers/activations/LeakyReLU.hpp) — gradient mask shape
- [src/core/training/Trainer.hpp](src/core/training/Trainer.hpp) — `clip_grad_norm` call site

## Checklist

1. For each layer with a `backward()`, confirm input shape == stored forward output shape.
2. Confirm all caches are cleared after optimizer step (prevent stale use across batches).
3. Add `NN_LOG_DEBUG` gradient norm output at the Trainer level.
4. Run a single training step and verify no NaN appears in any gradient tensor.

## Validation

- No shape mismatch between `backward()` input and `forward()` output in any layer.
- Gradient NaN/Inf triggers a log error and skips the optimizer step (or aborts training).
- BPTT history length assertion fires on misconfigured `time_steps`.
