# initializers

Purpose
- Weight and parameter initialization helpers used when constructing models and layers.

Usage
- Use functions provided in this folder to initialize tensors before training (e.g. Xavier, He, constant initializers).

Integration
- Typically called from model constructors in `layers/` or experiment setup code.

Tests
- See `src/core/initializers/tests/initializers_gtest.cpp` for usage examples.
