# layers

Neural-network layers (`Module<Backend>` subclasses): dense, activations, convolution, recurrent (LSTM, GRU), attention, normalisation, residual blocks, losses, spiking neurons (LIF, LIF-BPTT) and surrogate gradients. Almost all layers are header-only templates; only the convolutions have `.cpp` files here.

**Full documentation:** [`.wiki/Core/Layers.md`](../../../.wiki/Core/Layers.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/layers`](../../../include/layers) |
| Implementation (`.cpp`) | 3 file(s) in this folder |
| CMake library target | `layers` |
| Test executables | `independent_refs_gtest`, `layers_gtest`, `lif_basic_behavior_gtest`, `micro_network_parity_gtest`, `spiking_mechanisms_gtest` |

## Header map

- `include/layers/activations/`: `FastActivations.hpp`, `LeakyReLU.hpp`, `ReLU.hpp`, `Sigmoid.hpp`, `Tanh.hpp`
- `include/layers/attention/`: `MultiHeadAttention.hpp`, `PositionalEncoding.hpp`, `TransformerEncoderBlock.hpp`
- `include/layers/base/`: `Module.hpp`, `Sequential.hpp`
- `include/layers/convolution/`: `Conv1d.hpp`, `Conv2d.hpp`, `Conv2d_utils.hpp`, `MaxPool1d.hpp`, `MaxPool2d.hpp`, `PatchIndices.hpp`
- `include/layers/dense/`: `Linear.hpp`
- `include/layers/gru/`: `GRULayer.hpp`
- `include/layers/losses/`: `CrossEntropyLoss.hpp`, `MAELoss.hpp`, `MSELoss.hpp`, `SpikeCountLoss.hpp`, `SpikeTimeLoss.hpp`
- `include/layers/lstm/`: `LSTMLayer.hpp`
- `include/layers/normalization/`: `LayerNorm.hpp`
- `include/layers/regularization/`: `IRegularization.hpp`, `L1Regularization.hpp`, `L2Regularization.hpp`, `Regularization.hpp`
- `include/layers/residual/`: `ResNetBlock.hpp`, `ResidualBlock.hpp`, `SimpleResNet.hpp`
- `include/layers/spiking/`: `ArcTanSurrogate.hpp`, `BoxcarSurrogate.hpp`, `ExponentialSurrogate.hpp`, `ISurrogateGradient.hpp`, `Lif.hpp`, `LifBPTT.hpp`, `LifIntegrator.hpp`, `PoissonLatentLayer.hpp`, `SurrogateGradient.hpp`, `ThresholdDependentBatchNorm.hpp`, `TimeMeanPool.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target independent_refs_gtest
  ctest --test-dir out/build/max-performance -R 'IndependentCrossCheck' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R independent_refs_gtest` silently matches nothing.

  Suites here: `LSTMGateTest`, `LSTMLayerTest`, `ResidualBlockTest`, `ResNetBlockTest`, `SimpleResNetTest`, `Conv1dTest`, `MaxPoolTest`, `SigmoidTest`, `TanhTest`, `ReLUMechanismTest`, `LeakyReLUMechanismTest`, `LinearMechanismTest` (and 40 more).
- After a behaviour change, update [`.wiki/Core/Layers.md`](../../../.wiki/Core/Layers.md) (and `.wiki/References.md` if a paper is involved).
