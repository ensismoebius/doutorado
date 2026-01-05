# Test Status Report

## Summary

Compilation successful. 184 tests discovered. Many tests passing, but significant memory corruption issues remain.

## Test Results (Last Run)

### ✅ Passing Tests (54 tests)

- SimpleSignalOperationsTest.\*
- FiltersOperationsTest.\*
- WavFileTest.\*
- AudioFeatureExtractionTest.\* (all variants)
- NetworkSerializerTest.\*
- LeakyReLUTest.ForwardAndBackward
- SpikeCountLossTest.ForwardAndBackward
- SurrogateGradientTest.\* (Exponential, Boxcar)
- Conv2dTest.\* (all 19 Conv2d tests pass!)
- LayerExceptionTest.\* (all 4 validation tests pass!)

### ❌ Failing Tests (Multiple - "double free or corruption")

- MSELossTest.ForwardAndBackward (#9)
- SequentialTest.ForwardAndBackward (#10)
- LinearLayerTest.ForwardSimple (#11)
- LeakyLayerTest.ForwardSpikeAndReset (#12)
- LeakyLayerTest.ForwardSpikeNoResetZero (#13)
- L1RegularizationTest.Forward (#28)
- L1RegularizationTest.Backward (#29)
- L2RegularizationTest.Forward (#30)
- L2RegularizationTest.Backward (#31)
- SimpleResNetTest.ForwardAndBackward (#32)
- SimpleResNetTest.ForwardAndBackwardEdgeCases (#33)
- LayerMemoryStressTest.LargeLinearLayer (#38)
- (More tests likely affected)

## Root Cause Analysis

### Issue: Memory Corruption ("double free or corruption (out)")

**Pattern**: All failing tests use either:

1. Linear layer (with input caching: `input_cache = input`)
2. MSELoss (with input/target caching)
3. Tensor copy operations

**Potential Causes**:

1. ~~Tensor copy assignment operator~~ (reviewed - looks correct)
2. ~~EigenTensorBackend::clone()~~ (reviewed - looks correct)
3. ~~Tensor move semantics~~ (using = default with unique_ptr, should be fine)
4. **MSELoss::backward() modifying cached last_target** ✅ FIXED

### Recent Fix Applied

**File**: `src/nn/layers/MSELoss.hpp`

**Issue**: The `backward()` method was calling `last_target.multiply_scalar(-1.0f)` which modifies `last_target` in-place (since `multiply_scalar` returns `Tensor&`). This could corrupt the cached target data.

**Fix**: Create a copy of `last_target` before negating it:

```cpp
nn::Tensor negated_target = last_target;
negated_target.multiply_scalar(-1.0f);
auto diff = last_input.add(negated_target);
```

## Next Steps

1. **Rebuild the project**:

   ```bash
   cd build && cmake --build . -j$(nproc)
   ```

2. **Run tests** to verify fix:

   ```bash
   ctest --output-on-failure -j4
   ```

3. **If tests still fail**, investigate:
   - Tensor lifecycle management (copy/move/destroy)
   - EigenTensorBackend recursive gradient cloning
   - Potential buffer overflows in Eigen matrix operations
   - AddressSanitizer reports (may provide more details)

4. **Consider adding**:
   - More comprehensive Tensor unit tests
   - Memory leak detection tests
   - Explicit tests for copy/move semantics

## Build Configuration

- **Compiler**: Clang (C++20)
- **Build Type**: Debug
- **Sanitizers**: AddressSanitizer + UndefinedBehaviorSanitizer (enabled)
- **Optimization**: -O0 (no optimization in Debug)
- **OpenMP**: Enabled

## Code Changes Made

1. **src/nn/layers/Linear.hpp**: Added input dimension validation
2. **src/core/layers/Conv2d_impl.cpp**: Added 4D tensor and size validation
3. **src/nn/layers/Sequential.hpp**: Added empty layers validation
4. **src/nn/layers/MSELoss.hpp**: Added target_set validation + fixed backward()
5. **src/core/layers/tests/layers_gtest.cpp**: Fixed Tensor construction from Eigen matrices
6. Multiple files: Fixed type mismatches, unused includes, signed/unsigned comparisons

## Status: NEEDS TESTING

The MSELoss fix has been applied but not yet tested. Tests need to be run to verify if the memory corruption is resolved.
