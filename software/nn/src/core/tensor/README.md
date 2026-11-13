# tensor

The `nn::Tensor` type every other module computes with. `Tensor` is an alias chosen in `include/Backend.hpp`; the default backend is XTensor (CPU, BLAS). `DeviceTensorBackend` is a documented skeleton for adding a new backend: it delegates all maths to XTensor. A CMake guard (`cmake/BackendImplementationGuard.cmake`) fails the build if code outside `include/Backend.hpp` and the backend folders names a concrete backend.

**Full documentation:** [`.wiki/Core/Tensor.md`](../../../.wiki/Core/Tensor.md). This README only says where things are and how to change them.

| What | Where |
|---|---|
| Public headers | [`include/tensor`](../../../include/tensor), [`include/Backend.hpp`](../../../include/Backend.hpp), [`include/device`](../../../include/device) |
| Implementation (`.cpp`) | 1 file(s) in this folder |
| CMake library target | `tensor` |
| Test executables | `device_backend_gtest`, `pytorch_parity_gtest`, `tensor_all_backends_gtest`, `tensor_backend_switchability_gtest`, `tensor_gtest`, `tensor_mock_gtest`, `tensor_perf_bench` |

## Header map

- `include/`: `Backend.hpp`
- `include/device/`: `Device.hpp`, `DeviceRuntime.hpp`, `DeviceType.hpp`
- `include/tensor/`: `DeviceTensorBackend.hpp`, `Tensor.hpp`
- `include/tensor/xtensor/`: `XTensorBackend.hpp`

## Changing it

- A new header in `include/...` needs nothing else unless it has a `.cpp`; a new `.cpp` must be added to this folder's `CMakeLists.txt`.
- A new test goes in `tests/`, named `*_gtest.cpp`, and is registered in `tests/CMakeLists.txt` (`add_executable` + `target_link_libraries(... google_test ...)` + `gtest_discover_tests(<exe>)`; copy a neighbour) ([Build System](../../../.wiki/Guides/Build-System.md#running-tests)).
- Build and run (from `software/nn`):

  ```bash
  cmake --build --preset=max-performance --target device_backend_gtest
  ctest --test-dir out/build/max-performance -R 'DeviceBackendTest' --output-on-failure
  ```

  `ctest -R` matches the GoogleTest **suite** name, not the executable: `-R device_backend_gtest` silently matches nothing.

  Suites here: `DeviceBackendTest`, `TensorBackendSwitchability`, `TensorTest`.
- After a behaviour change, update [`.wiki/Core/Tensor.md`](../../../.wiki/Core/Tensor.md) (and `.wiki/References.md` if a paper is involved).
