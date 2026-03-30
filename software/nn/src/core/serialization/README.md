# serialization

Purpose
- Binary and structured serialization helpers for saving and loading `StateDict`s (models and optimizer state).

Key APIs
- `save_state_dict()` / `load_state_dict()` implemented in `include/nn/serialization/StateIO.hpp`.

Usage
- Use these helpers to persist `std::map<std::string, nn::Tensor>` style state dictionaries.

Tests
- See `src/core/serialization/tests/` and `src/core/optimizers/tests/state_io_gtest.cpp` for examples.
