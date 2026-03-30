# tensor

Purpose
- Core tensor abstraction and Eigen-backed implementations used across the codebase.

Usage
- Use `nn::Tensor` for numeric data and call high-level operations from the `tensor` API. Prefer high-level helpers instead of dealing with raw Eigen matrices directly.

Integration
- The tensor module is a foundational dependency for layers, optimizers, and serialization. Link against the `tensor`/`linearAlgebra` targets as required.

Tests
- See `src/core/tensor/tests/` for construction, operations, and gradient examples.
