# tensor

Purpose
- Core tensor abstraction and Eigen-backed implementations used across the codebase.

Usage
- Use `nn::Tensor` for numeric data and call high-level operations from the `tensor` API. Prefer high-level helpers instead of dealing with raw Eigen matrices directly.

Integration
- The tensor module is a foundational dependency for layers, optimizers, and serialization. Link against the `tensor`/`linearAlgebra` targets as required.

Tests
- See `src/core/tensor/tests/` for construction, operations, and gradient examples.

Recent updates
- Added backend primitive for rowwise addition of a `(cols, 1)` bias vector across all rows (`add_col_vector_to_rows_inplace`), exposed through `nn::Tensor`.

Optimization techniques and references
- Backend-level fused rowwise update: move broadcast-add into the Eigen-backed kernel path to leverage vectorized matrix expressions and avoid per-element scalar loops at call sites (see [1], [2]).

Bibliographic references
- [1] Gene H. Golub and Charles F. Van Loan. Matrix Computations (4th ed.). Johns Hopkins University Press, 2013.
- [2] Intel 64 and IA-32 Architectures Optimization Reference Manual. Intel Corporation, current edition.
