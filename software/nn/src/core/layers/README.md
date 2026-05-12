# layers

Purpose
- Neural network layer implementations (convolutions, activations, and helpers) and small layer utilities.

Layout
- Public headers are now organized by category under `include/nn/layers/`:
	- `base/`: module/container primitives (`Module`, `Sequential`)
	- `dense/`: dense layers (`Linear`)
	- `activations/`: activation layers (`ReLU`, `LeakyReLU`)
	- `convolution/`: convolution/pooling layers (`Conv2d`, `Conv2d_utils`, `MaxPool2d`)
	- `losses/`: objective layers (`MSELoss`, `MAELoss`, `CrossEntropyLoss`, `SpikeCountLoss`)
	- `regularization/`: regularization helpers (`Regularization`)
	- `residual/`: residual architecture blocks (`ResidualBlock`, `ResNetBlock`, `SimpleResNet`)
	- `spiking/`: spiking neuron and surrogate-gradient layers (`Lif`, `LifIntegrator`, `LifBPTT`, `SurrogateGradient`)
- Layer implementation TUs are organized by category under `src/core/layers/`:
	- `convolution/Conv2d_impl.cpp`
	- `convolution/Conv2d_utils.cpp`
	- `misc/dummy.cpp`

Usage
- Include layer headers from `include/nn/layers` and construct layer objects as part of your `Module`/`Sequential` models.

CMake Target
- `layers`

Tests
- See `src/core/layers/tests/` for examples of layer construction and forward-pass expectations.

Recent updates
- `Linear::forward` now performs bias addition through a backend-level rowwise vectorized path instead of nested scalar loops.
- `Conv2d` crash fixes: aligned Eigen/OpenMP compile definitions for the `layers` library with its test target using `configure_eigen_parallel_target(layers)` to prevent cross-target Eigen ABI mismatches that caused invalid frees in Conv2d tests.
- `Conv2d::im2col_optimized` now writes via tensor accessors instead of manual raw-pointer indexing, reducing risk of storage-layout-dependent corruption.
- SNN neuron safety: `Lif` and `LifIntegrator` now evaluate decay using positive-clamped effective `R`/`C` values to keep `tau`/`beta` numerically stable when raw parameters approach non-positive regions.
- `LifBPTT` updates: capacitance is now trainable and exposed in `params()`, and backward pass gradients were aligned with forward semantics (readout-mode recurrence consistency plus explicit threshold/reset-path contributions).
- Surrogate gradient constructors now validate hyperparameters and reject non-positive `sharpness`/`window` values.

Optimization techniques and references
- GEMM + vectorized bias epilogue: preserve matrix-multiply fast path and apply broadcast bias with backend rowwise operation to reduce scalar-loop overhead and improve SIMD utilization (see [1], [2]).

Bibliographic references
- [1] Kazushige Goto and Robert A. van de Geijn. Anatomy of High-Performance Matrix Multiplication. ACM TOMS, 34(3), 2008.
- [2] Intel 64 and IA-32 Architectures Optimization Reference Manual. Intel Corporation, current edition.
