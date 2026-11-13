# models

Ready-made models built from the layers: dense and spiking window autoencoders, LSTM, GRU and Transformer autoencoders, and their builders.

**Full documentation:** [`.wiki/Core/Models.md`](../../../.wiki/Core/Models.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/models`](../../../include/models) |
| Implementation (`.cpp`) | 12 file(s) in this folder |
| CMake library target | `autoencoder_models`, `gru_models`, `lstm_models`, `transformer_models` |
| Test executables | `autoencoder_config_gtest`, `gru_autoencoder_gtest`, `lstm_ae_pytorch_parity_gtest`, `lstm_autoencoder_gtest`, `lstm_batch_gtest`, `transformer_autoencoder_gtest` |

## Header map

- `include/models/autoencoder/`: `AudioWindowAutoencoder.hpp`, `AudioWindowSpikingAutoencoder.hpp`, `AutoencoderBuilders.hpp`, `AutoencoderConfig.hpp`, `AutoencoderLayerSpecParsers.hpp`, `AutoencoderStageBuilders.hpp`, `AutoencoderTensorUtils.hpp`, `EegWindowAutoencoder.hpp`, `EegWindowSpikingAutoencoder.hpp`, `EncoderDecoderAutoencoder.hpp`, `FusedWindowAutoencoder.hpp`, `FusedWindowSpikingAutoencoder.hpp`, `ProtocolAutoencoder.hpp`, `ProtocolSpikingAutoencoder.hpp`
- `include/models/gru/`: `GRUAutoencoder.hpp`, `GRUAutoencoderConfig.hpp`
- `include/models/lstm/`: `LSTMAutoencoder.hpp`, `LSTMAutoencoderConfig.hpp`
- `include/models/transformer/`: `TransformerAutoencoder.hpp`, `TransformerAutoencoderConfig.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target autoencoder_config_gtest
  ctest --test-dir out/build/max-performance -R 'AutoencoderConfigTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R autoencoder_config_gtest` silently matches nothing.

  Suites here: `AutoencoderConfigTest`, `GRUAutoencoderConfig`, `GRUAutoencoder`, `GRUAutoencoderInitSeed`, `LstmAutoencoderTorchParity`, `LSTMAutoencoderConfigTest`, `CoreLSTMLayerTest`, `CoreLSTMAutoencoderTest`, `CoreLSTMAutoencoderBatch`, `LSTMAutoencoderInitSeed`, `LSTMBatchTest`, `TransformerAutoencoder` (and 1 more).
- After a behaviour change, update [`.wiki/Core/Models.md`](../../../.wiki/Core/Models.md) (and `.wiki/References.md` if a paper is involved).
