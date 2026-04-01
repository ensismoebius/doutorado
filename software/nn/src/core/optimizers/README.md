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

Recent updates
- Optimizer thread-safety tests now use RAII (`std::unique_ptr`) for temporary parameter tensors while passing non-owning raw pointers to optimizer APIs.

Optimization techniques and references
- RAII-based temporary ownership in tests: deterministic cleanup and lower leak risk in exceptional paths, while preserving optimizer API contract on non-owning parameter spans (see [1], [2]).

Bibliographic references
- [1] Bjarne Stroustrup and Herb Sutter (eds.). C++ Core Guidelines (Resource Management and Ownership rules), ongoing.
- [2] Scott Meyers. Effective Modern C++. O'Reilly, 2014.
