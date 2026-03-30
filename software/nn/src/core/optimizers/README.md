# optimizers

Purpose
- Optimizer implementations and helper utilities (e.g. Adam, SGD conventions, parameter state management).

Usage
- Attach optimizers to model parameter spans, call `zero_grad()` and `step()` in your training loop.
- Use provided `state_dict()` / `load_state_dict()` helpers to checkpoint and restore optimizer state.

CMake Target
- `optimizers`

Tests
- See `src/core/optimizers/tests/` for examples of saving/loading state and expected numeric behavior.
