---
name: checkpoint-versioning-enforcer
description: "Enforce checkpoint format versioning, architecture metadata, and load-time compatibility validation."
---

# checkpoint-versioning-enforcer

Ensure model checkpoints are self-describing: they carry the format version and architecture shape needed to detect and reject incompatible loads.

## Rules

- **VERSION_HEADER**: Every checkpoint must begin with a format version string (e.g., `"nn_checkpoint_v1"`). No loading a checkpoint without verifying this header first.
- **ARCHITECTURE_METADATA**: Each checkpoint must include layer names, shapes (in, out), and dtypes. No loading weights into a model whose architecture was not validated against the saved metadata.
- **FAIL_ON_MISMATCH**: When loading, if architecture metadata does not match the current model, fail loudly with a diagnostic. No silent loading of incompatible weights.
- **PROJECT_VERSION_TAG**: Include the project CMake version (`0.2.0` or current) in the checkpoint metadata. Warn when checkpoint version predates current code.
- **ATOMIC_WRITE**: Write checkpoints atomically (write to `<path>.tmp`, then rename). No partial checkpoint files that can corrupt a training run.
- **COMPANION_JSON**: Alongside the binary weight file, always write a `<name>.meta.json` with version, architecture, seed, and timestamp. No binary-only checkpoints.

## Expected Checkpoint Layout

```
checkpoints/<experiment_id>/<timestamp>/
├── model.bin           ← weight tensors
└── model.meta.json     ← { "format_version": "nn_checkpoint_v1",
                             "project_version": "0.2.0",
                             "timestamp": "...",
                             "random_seed": 42,
                             "layers": [{"name": "fc1", "shape": [128, 64], "dtype": "float32"}] }
```

## Key Files to Update

- [include/serialization/NnSaver.hpp](include/serialization/NnSaver.hpp) — add version + metadata write
- [include/serialization/NetworkSerializer.hpp](include/serialization/NetworkSerializer.hpp) — add load-time compatibility check

## Validation

- Loading a checkpoint from a different architecture version prints an error and does not silently proceed.
- `model.meta.json` is present alongside every `model.bin`.
- Atomic write: no partial `.bin` file left on crash.
