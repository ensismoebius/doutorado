# saver

Purpose
- High-level save/restore helpers for model checkpoints and experiment artifacts.

Usage
- Utilities in this folder work with `serialization` helpers to persist model and optimizer `state_dict`s.

Integration
- Typical flow: call `save_state_dict()` from `serialization` to a file, and `load_state_dict()` during restore.

Tests
- See `src/core/serialization/tests/` for concrete examples.
