# src/core — the framework's implementation

One folder per module. Each folder's `README.md` gives the location of its public headers
(almost always `include/<module>/`), its CMake target, its tests and how to run them. It also
links the module's wiki page in [`.wiki/Core/`](../../.wiki/Core/).

| Module | What |
|---|---|
| [`tensor/`](tensor/README.md) | `nn::Tensor` and the backends |
| [`layers/`](layers/README.md) | layers, losses, spiking neurons |
| [`models/`](models/README.md) | ready-made autoencoders (dense, spiking, LSTM, GRU, Transformer) |
| [`optimizers/`](optimizers/README.md) | Adam, SGD, …, `OptimizerFactory` |
| [`training/`](training/README.md) | `Trainer` and its callbacks |
| [`initializers/`](initializers/README.md) | weight initialisation |
| [`data_loaders/`](data_loaders/README.md) | datasets, samplers, `DataLoader`, per-base loaders |
| [`statistics/`](statistics/README.md) | metrics, EER, k-fold, inference tests |
| [`paraconsistent/`](paraconsistent/README.md) | the paraconsistent feature score |
| [`wavelet/`](wavelet/README.md), [`wave/`](wave/README.md) | wavelet transforms; WAV and audio features |
| [`linear_algebra/`](linear_algebra/README.md) | BLAS/LAPACK helpers |
| [`serialization/`](serialization/README.md) | saving and loading models |
| [`progress/`](progress/README.md), [`utility/`](utility/README.md) | progress bars; shared helpers and transforms |

Include public headers relative to `include/` with no prefix (`#include "tensor/Tensor.hpp"`).
Headers that live here take a `core/` prefix (`#include "core/training/Trainer.hpp"`).

Build everything and run all tests (from `software/nn`):

```bash
cmake --preset=max-performance
cmake --build --preset=max-performance -j$(nproc)
ctest --test-dir out/build/max-performance --output-on-failure -j4
```

The `src/core/*/tests/*_gtest.cpp` files are also the best usage examples of each API.
