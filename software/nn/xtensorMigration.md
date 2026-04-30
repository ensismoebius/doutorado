# xtensor Migration Guide
## Replace EigenTensorBackend → XTensorBackend, Add LSTM 3D Support

**Status:** Reference only — do not execute until instructed.  
**Scope:** Full removal of Eigen from codebase, wiki, comments. Replace with xtensor. Redesign LSTM forward to accept `(B, T, D)` 3D tensors natively.

---

## Table of Contents

1. [Why xtensor](#1-why-xtensor)
2. [What Changes, What Stays](#2-what-changes-what-stays)
3. [Complete File Inventory](#3-complete-file-inventory)
4. [Step 0 — Understand the API Mapping](#step-0--understand-the-api-mapping)
5. [Step 1 — Add xtensor to CMake](#step-1--add-xtensor-to-cmake)
6. [Step 2 — Create XTensorBackend](#step-2--create-xtensorbackend)
7. [Step 3 — Update Backend.hpp](#step-3--update-backendhpp)
8. [Step 4 — Update DeviceTensorBackend.hpp](#step-4--update-devicetensorbackendhpp)
9. [Step 5 — Replace VendorEigenParallel → VendorXtensorParallel](#step-5--replace-vendoreigenparallel--vendorxtensorparallel)
10. [Step 6 — Replace EigenParallel.hpp → XtensorParallel.hpp](#step-6--replace-eigenparallelhpp--xtensorparallelhpp)
11. [Step 7 — Rewrite vectorizationCheck.cpp](#step-7--rewrite-vectorizationcheckcpp)
12. [Step 8 — Fix mat_file_utils.cpp](#step-8--fix-mat_file_utilscpp)
13. [Step 9 — Remove EigenBan Infrastructure](#step-9--remove-eigenban-infrastructure)
14. [Step 10 — Update All CMakeLists.txt Files](#step-10--update-all-cmakeliststxt-files)
15. [Step 11 — Remove Eigen from PackageChecking.cmake](#step-11--remove-eigen-from-packagecheckingcmake)
16. [Step 12 — Scan and Purge Residual Eigen References](#step-12--scan-and-purge-residual-eigen-references)
17. [Step 13 — Redesign LSTM for 3D Support](#step-13--redesign-lstm-for-3d-support)
18. [Step 14 — Delete Deprecated Files and Directories](#step-14--delete-deprecated-files-and-directories)
19. [Step 15 — Update Wiki](#step-15--update-wiki)
20. [Step 16 — Build and Verify](#step-16--build-and-verify)
21. [Troubleshooting](#troubleshooting)

---

## 1. Why xtensor

| | Eigen | xtensor |
|---|---|---|
| Storage | 2D `MatrixXf`; higher dims faked via flattening | Native N-D strided array (`xarray<float>`) |
| N-D matmul | Error — must pre-flatten | `xt::linalg::dot` on any 2D slice; batch loops natural |
| Broadcasting | Manual `.replicate()` | Automatic (NumPy rules) |
| Views | `.block()`, `.row()` return copies | `xt::view(...)` zero-copy lazy views |
| Reshape | Data copy + shape hack | `xt::reshape_view(...)` zero-copy, or in-place |
| 3D LSTM input | Impossible in one `forward()` call | `forward(Tensor{B,T,D})` works natively |
| Header-only | Yes | Yes (xtensor + xtl); xtensor-blas needs BLAS |

---

## 2. What Changes, What Stays

**Changes:**
- `include/nn/tensor/eigen/EigenTensorBackend.hpp` → `include/nn/tensor/xtensor/XTensorBackend.hpp`
- `include/nn/Backend.hpp` — alias updated
- `include/nn/tensor/DeviceTensorBackend.hpp` — remove Eigen constructor
- `include/nn/utility/EigenParallel.hpp` → `include/nn/utility/XtensorParallel.hpp`
- `cmake/PackageChecking.cmake` — remove `find_package(Eigen3)`
- `cmake/VendorEigenParallel.cmake` — replaced by `cmake/VendorXtensorParallel.cmake`
- `cmake/EigenBan.cmake` + `cmake/EigenBan.hpp` — deleted
- `cmake/VendorIncludes.cmake` — update includes
- 17 CMakeLists.txt files — `Eigen3::Eigen` → `xtensor`, `configure_eigen_parallel_target` → new helper
- `src/core/utility/vectorizationCheck.cpp` — remove Eigen SIMD report
- `src/core/dataLoaders/io/mat_file_utils.cpp` — remove Eigen matrix intermediary
- `include/nn/layers/lstm/LSTMLayer.hpp` — 3D-aware forward/backward
- Wiki: 11 `.md` files

**Stays unchanged:**
- `include/nn/tensor/Tensor.hpp` — public API unchanged; Backend concept fulfilled by XTensorBackend
- All layer headers (`Linear`, `Conv1d`, `ReLU`, etc.) — they never include Eigen directly
- Optimizer, loss, saver headers — not Eigen-aware
- OpenCL backend — separate, not touched by this migration
- All experiment files that already use `nn::Backend` alias — zero changes needed there

---

## 3. Complete File Inventory

### Files to CREATE
| File | Purpose |
|---|---|
| `include/nn/tensor/xtensor/XTensorBackend.hpp` | New CPU backend |
| `cmake/VendorXtensor.cmake` | FetchContent for xtl + xtensor + xtensor-blas |
| `cmake/VendorXtensorParallel.cmake` | OpenMP + BLAS wiring (replaces VendorEigenParallel) |
| `include/nn/utility/XtensorParallel.hpp` | Thread-count init helper (replaces EigenParallel.hpp) |

### Files to EDIT
| File | What changes |
|---|---|
| `include/nn/Backend.hpp` | `EigenTensorBackend` → `XTensorBackend` |
| `include/nn/tensor/DeviceTensorBackend.hpp` | Remove Eigen ctors; delegate to XTensorBackend |
| `cmake/PackageChecking.cmake` | Remove `find_package(Eigen3 REQUIRED NO_MODULE)` |
| `cmake/VendorIncludes.cmake` | Remove `VendorEigenParallel`; add `VendorXtensor` + `VendorXtensorParallel` |
| `CMakeLists.txt` (top-level) | Remove `include(cmake/EigenBan.cmake)` |
| `cmake/DevAndAnalysisTargets.cmake` | Remove `check_eigen_leaks` target |
| `src/core/layers/CMakeLists.txt` | `Eigen3::Eigen` → `xtensor`; update `configure_eigen_parallel_target` call |
| `src/core/tensor/CMakeLists.txt` | Same |
| `src/core/tensor/tests/CMakeLists.txt` | Same (3 test targets) |
| `src/core/layers/tests/CMakeLists.txt` | Same |
| `src/core/dataLoaders/CMakeLists.txt` | Same |
| `src/core/dataLoaders/10.1117/CMakeLists.txt` | Same (2 targets) |
| `src/core/dataLoaders/10.1117/tests/CMakeLists.txt` | Same |
| `src/core/dataLoaders/10.1117/tests/fuzz/CMakeLists.txt` | Same |
| `src/core/dataLoaders/tests/CMakeLists.txt` | Same |
| `src/core/dataLoaders/tests/MatTestUtils/CMakeLists.txt` | Same |
| `src/core/dataLoaders/10.1117/tests/windowing/CMakeLists.txt` | Same |
| `src/core/initializers/tests/CMakeLists.txt` | Same |
| `src/core/optimizers/tests/CMakeLists.txt` | Same |
| `src/core/statistics/CMakeLists.txt` | Same |
| `src/core/utility/CMakeLists.txt` | Same |
| `src/core/utility/tests/CMakeLists.txt` | Same |
| `src/core/wave/CMakeLists.txt` | Same |
| `src/core/utility/vectorizationCheck.cpp` | Remove Eigen SIMD report |
| `src/core/dataLoaders/io/mat_file_utils.cpp` | Remove Eigen matrix intermediary |
| `include/nn/layers/lstm/LSTMLayer.hpp` | 3D-aware LSTM |
| `.wiki/Core/Tensor.md` | Update backend description |
| `.wiki/Core/Device.md` | Update mermaid diagram |
| `.wiki/Home.md` | Remove Eigen from deps list |
| `.wiki/Architecture.md` | Update diagrams |
| `.wiki/Core/Training.md` | Update code examples |
| `.wiki/Core/Layers.md` | Update examples |
| `.wiki/Concepts/LSTM-and-BPTT.md` | Update class examples |
| `.wiki/Concepts/Residual-Blocks.md` | Update class examples |
| `.wiki/Core/LinearAlgebra.md` | Update backend description |
| `.wiki/Guides/Build-System.md` | Remove EigenBan entries, add xtensor |

### Files to DELETE
| File | Reason |
|---|---|
| `include/nn/tensor/eigen/EigenTensorBackend.hpp` | Replaced by XTensorBackend |
| `include/nn/tensor/eigen/` directory | Empty after above deletion |
| `include/nn/utility/EigenParallel.hpp` | Replaced by XtensorParallel.hpp |
| `cmake/EigenBan.cmake` | Eigen no longer present |
| `cmake/EigenBan.hpp` | Same |
| `cmake/VendorEigenParallel.cmake` | Replaced |
| `scripts/check_eigen_leaks.py` | Eigen no longer present |

---

## Step 0 — Understand the API Mapping

Every method in `EigenTensorBackend` maps to an xtensor equivalent. Use this table
as the authoritative reference when writing `XTensorBackend`.

### Storage
| Eigen | xtensor |
|---|---|
| `Eigen::MatrixXf m_data` | `xt::xarray<float> m_data` |
| `m_data.rows()` | `m_data.shape(0)` |
| `m_data.cols()` | `m_data.shape(1)` |
| `m_data.size()` | `m_data.size()` |
| `m_data.data()` | `m_data.data()` (same — contiguous row-major) |

### Construction
| Eigen pattern | xtensor replacement |
|---|---|
| `Eigen::MatrixXf::Zero(r, c)` | `xt::zeros<float>({r, c})` |
| `m_data.resize(r, c); m_data.setZero()` | `m_data = xt::zeros<float>({r, c})` |
| `Eigen::MatrixXf::NullaryExpr(r, c, [&]{return dist(rng);})` | Manual fill loop: `m_data = xt::empty<float>({r,c}); for(auto& v: m_data) v = dist(rng);` |

### Mutation
| Eigen | xtensor |
|---|---|
| `m_data.setZero()` | `m_data.fill(0.0f)` |
| `m_data.setOnes()` | `m_data.fill(1.0f)` |
| `m_data.setConstant(v)` | `m_data.fill(v)` |

### Element access
| Eigen | xtensor |
|---|---|
| `m_data(row, col)` | `m_data(row, col)` (identical syntax!) |
| `m_data(flat_idx)` — 1D flat | `*(m_data.data() + flat_idx)` |
| Flat index from (d1,d2,d3): `m_data(d1, d2*shape[2]+d3)` | `m_data(d1, d2, d3)` — direct! |

> **Key insight**: xtensor `m_data(d1, d2, d3)` works directly when `m_data` has shape
> `{d1_max, d2_max, d3_max}`. No manual index flattening needed. This alone is a major
> simplification and the main reason for this migration.

### Arithmetic (in-place)
| Eigen | xtensor |
|---|---|
| `m_data += other.m_data` | `m_data += other.m_data` (identical!) |
| `m_data -= other.m_data` | `m_data -= other.m_data` (identical!) |
| `m_data.array() *= other.m_data.array()` | `m_data *= other.m_data` (identical!) |
| `m_data.array() /= other.m_data.array()` | `m_data /= other.m_data` (identical!) |
| `m_data.array() += scalar` | `m_data += scalar` (identical!) |
| `m_data *= scalar` | `m_data *= scalar` (identical!) |
| `m_data /= scalar` | `m_data /= scalar` (identical!) |
| `m_data = m_data.array().sqrt()` | `m_data = xt::sqrt(m_data)` |
| `m_data = m_data.array().square()` | `m_data = xt::square(m_data)` |

### Arithmetic (functional / returning new tensor)
| Eigen | xtensor |
|---|---|
| `m_data + other.m_data` | `m_data + other.m_data` (identical!) |
| `m_data - other.m_data` | `m_data - other.m_data` (identical!) |
| `m_data.cwiseProduct(other.m_data)` | `m_data * other.m_data` |
| `m_data.array() / other.m_data.array()` | `m_data / other.m_data` |
| `m_data.array() + scalar` | `m_data + scalar` |
| `m_data * scalar` | `m_data * scalar` |
| `m_data / scalar` | `m_data / scalar` |
| `m_data.array().exp()` | `xt::exp(m_data)` |
| `m_data.array().sqrt()` | `xt::sqrt(m_data)` |
| `m_data.array().square()` | `xt::square(m_data)` |
| `m_data.array().abs()` | `xt::abs(m_data)` |
| `m_data.cwiseMax(0.0f)` | `xt::maximum(m_data, 0.0f)` |
| `(m_data.array()>0).select(m_data, alpha*m_data)` | `xt::where(m_data > 0.0f, m_data, alpha * m_data)` |
| `m_data.cwiseMax(min).cwiseMin(max)` | `xt::clip(m_data, min, max)` |

### Matrix operations
| Eigen | xtensor |
|---|---|
| `m_data * other.m_data` — matmul | `xt::linalg::dot(m_data, other.m_data)` |
| `m_data * other.m_data.transpose()` | `xt::linalg::dot(m_data, xt::transpose(other.m_data))` |
| `m_data.transpose()` | `xt::eval(xt::transpose(m_data))` |
| `m_data.rowwise() += col.transpose()` | `m_data += xt::reshape_view(xt::view(col, xt::all(), 0), std::vector<std::size_t>{1, col.shape(0)})` |

> **Note on `xt::eval`**: `xt::transpose` returns a lazy view. Wrap with `xt::eval()` to
> get an evaluated copy when you need to store it as a standalone `xarray<float>`.

### Reductions
| Eigen | xtensor |
|---|---|
| `m_data.sum()` | `xt::sum(m_data)()` — `()` extracts scalar from 0-D xarray |
| `m_data.mean()` | `xt::mean(m_data)()` |
| `m_data.norm()` | `std::sqrt(xt::sum(xt::square(m_data))())` |
| `m_data.squaredNorm()` | `xt::sum(xt::square(m_data))()` |
| `(a-b).squaredNorm() / n` | `xt::sum(xt::square(m_data - target.m_data))() / static_cast<float>(m_data.size())` |
| `m_data.rowwise().sum()` | `xt::eval(xt::reshape_view(xt::sum(m_data, {1}), std::vector<std::size_t>{m_data.shape(0), 1}))` |
| `m_data.colwise().sum()` | `xt::eval(xt::reshape_view(xt::sum(m_data, {0}), std::vector<std::size_t>{1, m_data.shape(1)}))` |
| `m_data.hasNaN()` | `xt::any(xt::isnan(m_data))()` |
| `m_data.isApprox(other.m_data)` | `xt::allclose(m_data, other.m_data)` |

> **Critical**: `xt::sum(arr, {1})` returns shape `{N}` (1-D), NOT `{N, 1}`.
> Existing code that wraps `rowwise().sum()` in a `(rows, 1)` tensor expects `{N, 1}`.
> Always reshape as shown above.

### Comparisons (element-wise → float 0/1)
| Eigen | xtensor |
|---|---|
| `(a < b).cast<float>()` | `xt::cast<float>(m_data < other.m_data)` |
| `(a <= b).cast<float>()` | `xt::cast<float>(m_data <= other.m_data)` |
| `(a > b).cast<float>()` | `xt::cast<float>(m_data > other.m_data)` |
| `(a >= b).cast<float>()` | `xt::cast<float>(m_data >= other.m_data)` |
| `(a == b).cast<float>()` | `xt::cast<float>(m_data == other.m_data)` |
| `(a < scalar).cast<float>()` | `xt::cast<float>(m_data < scalar)` |
| `a.replicate(n, 1)` for broadcast | **Delete — not needed.** xtensor auto-broadcasts. |

> **Broadcasting**: xtensor follows NumPy broadcasting rules automatically.
> `{1, C}` vs `{R, C}` — xtensor handles this without `replicate`. The entire
> `compare_lt / compare_le / compare_eq / compare_gt / compare_ge` broadcasting logic
> in EigenTensorBackend collapses to just the expression on same-shape or
> already-broadcastable tensors. The `out_rows = max(rows_a, rows_b)` logic disappears.

### Slicing and views
| Eigen | xtensor |
|---|---|
| `m_data.row(i)` | `xt::eval(xt::view(m_data, i, xt::all()))` → shape `{cols}` (1-D) |
| `m_data.col(j)` | `xt::eval(xt::view(m_data, xt::all(), j))` → shape `{rows}` (1-D) |
| `m_data.leftCols(n)` | `xt::eval(xt::view(m_data, xt::all(), xt::range(0, n)))` |
| `m_data.topRows(n)` | `xt::eval(xt::view(m_data, xt::range(0, n), xt::all()))` |
| `m_data.block(r, c, rows, cols)` | `xt::eval(xt::view(m_data, xt::range(r, r+rows), xt::range(c, c+cols)))` |
| `m_data.block(...) = other.m_data` | `xt::view(m_data, xt::range(r, r+rows), xt::range(c, c+cols)) = other.m_data` |
| `slice(span<int> indices)` — row gather | Loop: `for(size_t i=0; i<n; ++i) xt::view(result, i, xt::all()) = xt::view(m_data, idx[i], xt::all());` |

> **Shape note for row()**: Eigen's `m_data.row(i)` returns a `{1, cols}` matrix.
> xtensor's `xt::view(m_data, i, xt::all())` returns a 1-D tensor `{cols}`.
> The `XTensorBackend::row()` method must wrap the result back into `{1, cols}` shape
> so callers that expect a 2D "row tensor" don't break. Use:
> ```cpp
> xt::eval(xt::reshape_view(xt::view(m_data, i, xt::all()),
>                           std::vector<std::size_t>{1, m_data.shape(1)}))
> ```

### Gradient storage
No change in concept — keep `std::unique_ptr<XTensorBackend> m_grad_backend`.
Replace `make_like(Eigen::MatrixXf data)` with `make_like(xt::xarray<float> data)`:
```cpp
XTensorBackend make_like(xt::xarray<float> data) const {
    XTensorBackend result;
    result.m_data = std::move(data);
    result.m_shape = m_shape;
    return result;
}
```

---

## Step 1 — Add xtensor to CMake

### 1a. Create `cmake/VendorXtensor.cmake`

```cmake
# cmake/VendorXtensor.cmake
# Downloads xtl (required by xtensor), xtensor (N-D array), and xtensor-blas (matmul).
# All are header-only except xtensor-blas which requires BLAS (already found via PackageChecking).

include(FetchContent)

# xtl — type utilities required by xtensor
FetchContent_Declare(xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    GIT_TAG        0.7.7
    GIT_SHALLOW    TRUE)

# xtensor — core N-D array library
FetchContent_Declare(xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    GIT_TAG        0.25.0
    GIT_SHALLOW    TRUE)

# xtensor-blas — BLAS-backed linalg (dot, norm, solve)
FetchContent_Declare(xtensor-blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas
    GIT_TAG        0.21.0
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(xtl xtensor xtensor-blas)

# xtensor-blas needs BLAS — already required in PackageChecking.cmake via find_package(BLAS).
# Link xtensor-blas to targets that need linalg::dot via:
#   target_link_libraries(<target> PUBLIC xtensor xtensor-blas)
```

### 1b. Create `cmake/VendorXtensorParallel.cmake`

```cmake
# cmake/VendorXtensorParallel.cmake
# Provides configure_xtensor_parallel_target(<target>).
# Links OpenMP + BLAS/LAPACK. Replaces VendorEigenParallel.cmake.

function(configure_xtensor_parallel_target target)
    target_link_libraries(${target} PRIVATE
        OpenMP::OpenMP_CXX
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        xtensor
        xtensor-blas)

    target_compile_definitions(${target} PRIVATE
        XTENSOR_USE_OPENMP=1
        XTENSOR_USE_XSIMD=1)
endfunction()
```

### 1c. Edit `cmake/VendorIncludes.cmake`

Remove this line:
```cmake
include(cmake/VendorEigenParallel.cmake)
```

Add these lines in its place:
```cmake
include(cmake/VendorXtensor.cmake)
include(cmake/VendorXtensorParallel.cmake)
```

### 1d. Edit `cmake/PackageChecking.cmake`

Remove this line exactly:
```cmake
find_package(Eigen3 REQUIRED NO_MODULE)
```

Do not add anything in its place. xtensor is fetched via FetchContent in VendorXtensor.cmake.

### 1e. Edit top-level `CMakeLists.txt`

Remove this line:
```cmake
include(cmake/EigenBan.cmake)
```

Do not add anything in its place.

---

## Step 2 — Create XTensorBackend

Create file `include/nn/tensor/xtensor/XTensorBackend.hpp`. This is the largest step.
The complete implementation follows. Copy it exactly.

```cpp
#pragma once
// XTensorBackend — xtensor-backed N-D tensor storage.
// Replaces EigenTensorBackend. Uses xt::xarray<float> for true N-D support.
// All shapes are logical (not flattened): a (B,T,D) tensor stores B*T*D elements
// and is indexed as m_data(b, t, d) — no manual stride arithmetic.

#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>    // xt::zeros, xt::ones
#include <xtensor/xeval.hpp>       // xt::eval
#include <xtensor/xmath.hpp>       // xt::exp, sqrt, square, abs, sum, mean, clip
#include <xtensor/xmanipulation.hpp> // xt::reshape_view, xt::transpose
#include <xtensor/xview.hpp>       // xt::view, xt::range, xt::all
#include <xtensor/xio.hpp>         // operator<< for debug
#include <xtensor-blas/xlinalg.hpp> // xt::linalg::dot, norm

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "nn/logging/Logger.hpp"

namespace nn
{

using Index = std::size_t;

class XTensorBackend
{
   public:
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    XTensorBackend() = default;

    explicit XTensorBackend(Index rows, Index cols)
        : m_data(xt::zeros<float>({rows, cols})), m_shape({rows, cols})
    {}

    explicit XTensorBackend(Index d1, Index d2, Index d3)
        : m_data(xt::zeros<float>({d1, d2, d3})), m_shape({d1, d2, d3})
    {}

    explicit XTensorBackend(Index d1, Index d2, Index d3, Index d4)
        : m_data(xt::zeros<float>({d1, d2, d3, d4})), m_shape({d1, d2, d3, d4})
    {}

    explicit XTensorBackend(const std::vector<Index>& shape) : m_shape(shape)
    {
        xt::dynamic_shape<Index> xshape(shape.begin(), shape.end());
        m_data = xt::zeros<float>(xshape);
    }

    // Construct directly from an xt::xarray (used internally by make_like and operations).
    explicit XTensorBackend(xt::xarray<float> data)
        : m_data(std::move(data))
    {
        m_shape = std::vector<Index>(m_data.shape().begin(), m_data.shape().end());
    }

    // Copy — deep copy including gradient.
    XTensorBackend(const XTensorBackend& other)
        : m_data(other.m_data), m_shape(other.m_shape)
    {
        if (other.m_grad_backend)
            m_grad_backend = std::make_unique<XTensorBackend>(*other.m_grad_backend);
    }

    XTensorBackend(XTensorBackend&& other) noexcept = default;

    XTensorBackend& operator=(const XTensorBackend& other)
    {
        if (this != &other)
        {
            m_data  = other.m_data;
            m_shape = other.m_shape;
            if (other.m_grad_backend)
                m_grad_backend = std::make_unique<XTensorBackend>(*other.m_grad_backend);
            else
                m_grad_backend.reset();
        }
        return *this;
    }

    XTensorBackend& operator=(XTensorBackend&& other) noexcept = default;

    // ------------------------------------------------------------------
    // Static factories
    // ------------------------------------------------------------------

    static XTensorBackend zeros(Index rows, Index cols)
    {
        return XTensorBackend(xt::zeros<float>({rows, cols}));
    }

    static XTensorBackend ones(Index rows, Index cols)
    {
        return XTensorBackend(xt::ones<float>({rows, cols}));
    }

    static XTensorBackend random(Index rows, Index cols)
    {
        XTensorBackend t(rows, cols);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : t.m_data) v = dist(rng);
        return t;
    }

    static XTensorBackend random(Index rows, Index cols, std::mt19937& rng)
    {
        XTensorBackend t(rows, cols);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : t.m_data) v = dist(rng);
        return t;
    }

    static XTensorBackend random(Index d1, Index d2, Index d3)
    {
        XTensorBackend t(d1, d2, d3);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : t.m_data) v = dist(rng);
        return t;
    }

    static XTensorBackend random(Index d1, Index d2, Index d3, std::mt19937& rng)
    {
        XTensorBackend t(d1, d2, d3);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : t.m_data) v = dist(rng);
        return t;
    }

    // ------------------------------------------------------------------
    // Shape
    // ------------------------------------------------------------------

    const std::vector<Index>& shape() const { return m_shape; }

    void reshape(const std::vector<Index>& new_shape)
    {
        const Index new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                               Index{1}, std::multiplies<Index>{});
        if (m_data.size() != new_size)
            throw std::invalid_argument("Reshape total size mismatch");

        xt::dynamic_shape<Index> xshape(new_shape.begin(), new_shape.end());
        m_data.reshape(xshape);
        m_shape = new_shape;
    }

    Index rows() const { return m_shape.empty() ? 0 : m_shape[0]; }
    Index cols() const { return m_shape.size() < 2 ? 1 : m_shape[1]; }
    Index size() const { return m_data.size(); }

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    float& at(Index i)
    {
        if (i >= size()) throw std::out_of_range("Index out of range");
        return *(m_data.data() + i);
    }
    const float& at(Index i) const
    {
        if (i >= size()) throw std::out_of_range("Index out of range");
        return *(m_data.data() + i);
    }

    float& at(Index row, Index col)
    {
        if (m_shape.size() != 2)
            throw std::invalid_argument("at(row, col) valid only for 2D tensors");
        if (row >= m_shape[0] || col >= m_shape[1])
            throw std::out_of_range("Index out of range");
        return m_data(row, col);
    }
    const float& at(Index row, Index col) const
    {
        if (m_shape.size() != 2)
            throw std::invalid_argument("at(row, col) valid only for 2D tensors");
        if (row >= m_shape[0] || col >= m_shape[1])
            throw std::out_of_range("Index out of range");
        return m_data(row, col);
    }

    float& at(Index d1, Index d2, Index d3)
    {
        if (m_shape.size() != 3)
            throw std::invalid_argument("at(d1,d2,d3) valid only for 3D tensors");
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2])
            throw std::out_of_range("Index out of range");
        return m_data(d1, d2, d3);   // no manual flattening needed!
    }
    const float& at(Index d1, Index d2, Index d3) const
    {
        if (m_shape.size() != 3)
            throw std::invalid_argument("at(d1,d2,d3) valid only for 3D tensors");
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2])
            throw std::out_of_range("Index out of range");
        return m_data(d1, d2, d3);
    }

    float& at(Index d1, Index d2, Index d3, Index d4)
    {
        if (m_shape.size() != 4)
            throw std::invalid_argument("at(d1,d2,d3,d4) valid only for 4D tensors");
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2] || d4 >= m_shape[3])
            throw std::out_of_range("Index out of range");
        return m_data(d1, d2, d3, d4);
    }
    const float& at(Index d1, Index d2, Index d3, Index d4) const
    {
        if (m_shape.size() != 4)
            throw std::invalid_argument("at(d1,d2,d3,d4) valid only for 4D tensors");
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2] || d4 >= m_shape[3])
            throw std::out_of_range("Index out of range");
        return m_data(d1, d2, d3, d4);
    }

    float& at(const std::vector<Index>& indices)
    {
        if (indices.size() != m_shape.size())
            throw std::invalid_argument("Indices dimension mismatch");
        if (indices.size() == 1) return at(indices[0]);
        if (indices.size() == 2) return at(indices[0], indices[1]);
        if (indices.size() == 3) return at(indices[0], indices[1], indices[2]);
        if (indices.size() == 4) return at(indices[0], indices[1], indices[2], indices[3]);
        // 5D+: flat linear
        Index flat = 0, stride = 1;
        for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
        {
            if (indices[i] >= m_shape[i]) throw std::out_of_range("Index out of range");
            flat += indices[i] * stride;
            stride *= m_shape[i];
        }
        return *(m_data.data() + flat);
    }
    const float& at(const std::vector<Index>& indices) const
    {
        if (indices.size() != m_shape.size())
            throw std::invalid_argument("Indices dimension mismatch");
        if (indices.size() == 1) return at(indices[0]);
        if (indices.size() == 2) return at(indices[0], indices[1]);
        if (indices.size() == 3) return at(indices[0], indices[1], indices[2]);
        if (indices.size() == 4) return at(indices[0], indices[1], indices[2], indices[3]);
        Index flat = 0, stride = 1;
        for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
        {
            if (indices[i] >= m_shape[i]) throw std::out_of_range("Index out of range");
            flat += indices[i] * stride;
            stride *= m_shape[i];
        }
        return *(m_data.data() + flat);
    }

    // ------------------------------------------------------------------
    // In-place arithmetic
    // ------------------------------------------------------------------

    void add_inplace(const XTensorBackend& other)
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for add_inplace");
        m_data += other.m_data;
    }
    void subtract_inplace(const XTensorBackend& other)
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for subtract_inplace");
        m_data -= other.m_data;
    }
    void multiply_inplace(const XTensorBackend& other)
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for multiply_inplace");
        m_data *= other.m_data;
    }
    void divide_inplace(const XTensorBackend& other)
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for divide_inplace");
        m_data /= other.m_data;
    }
    void add_scalar_inplace(float val)    { m_data += val; }
    void multiply_scalar_inplace(float val) { m_data *= val; }
    void divide_scalar_inplace(float val)   { m_data /= val; }
    void sqrt_inplace()  { m_data = xt::sqrt(m_data); }
    void square_inplace() { m_data = xt::square(m_data); }

    // ------------------------------------------------------------------
    // Functional arithmetic (return new backend)
    // ------------------------------------------------------------------

    XTensorBackend exp() const { return make_like(xt::eval(xt::exp(m_data))); }

    XTensorBackend rowwise_sum() const
    {
        // Returns shape {rows, 1} to match Eigen convention.
        auto s = xt::eval(xt::sum(m_data, {1}));
        s.reshape({m_data.shape(0), 1});
        return XTensorBackend(s);
    }

    XTensorBackend add(const XTensorBackend& other) const
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for add");
        return make_like(xt::eval(m_data + other.m_data));
    }

    void add_col_vector_to_rows_inplace(const XTensorBackend& col_vector)
    {
        // col_vector shape: {F, 1}. Add bias[f] to all m_data[:, f].
        if (m_shape.size() != 2 || col_vector.m_shape.size() != 2)
            throw std::invalid_argument("add_col_vector_to_rows_inplace valid only for 2D");
        if (col_vector.cols() != 1 || col_vector.rows() != cols())
            throw std::invalid_argument("Bias shape must be (cols, 1)");
        // Reshape col_vector from {F,1} to {1,F} so xtensor broadcasts over rows.
        auto bias_row = xt::eval(xt::reshape_view(
            xt::view(col_vector.m_data, xt::all(), 0),
            std::vector<Index>{1, col_vector.m_data.shape(0)}));
        m_data += bias_row;
    }

    XTensorBackend subtract(const XTensorBackend& other) const
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for subtract");
        return make_like(xt::eval(m_data - other.m_data));
    }

    XTensorBackend multiply(const XTensorBackend& other) const
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for multiply");
        return make_like(xt::eval(m_data * other.m_data));
    }

    XTensorBackend matmul_transposed(const XTensorBackend& other) const
    {
        if (m_shape.size() != 2 || other.m_shape.size() != 2)
            throw std::invalid_argument("matmul_transposed valid only for 2D");
        if (cols() != other.cols())
            throw std::invalid_argument("Dimension mismatch for matmul_transposed");
        return XTensorBackend(xt::linalg::dot(m_data, xt::transpose(other.m_data)));
    }

    XTensorBackend matmul(const XTensorBackend& other) const
    {
        if (m_shape.size() != 2 || other.m_shape.size() != 2)
            throw std::invalid_argument("matmul valid only for 2D");
        if (cols() != other.rows())
            throw std::invalid_argument("Dimension mismatch for matmul");
        return XTensorBackend(xt::linalg::dot(m_data, other.m_data));
    }

    XTensorBackend transpose() const
    {
        if (m_shape.size() != 2)
            throw std::invalid_argument("transpose valid only for 2D");
        return XTensorBackend(xt::eval(xt::transpose(m_data)));
    }

    XTensorBackend add_scalar(float val) const { return make_like(xt::eval(m_data + val)); }
    XTensorBackend multiply_scalar(float val) const { return make_like(xt::eval(m_data * val)); }
    XTensorBackend divide_scalar(float val) const { return make_like(xt::eval(m_data / val)); }

    XTensorBackend divide(const XTensorBackend& other) const
    {
        return make_like(xt::eval(m_data / other.m_data));
    }

    // ------------------------------------------------------------------
    // Comparisons (return float tensor of 0/1)
    // ------------------------------------------------------------------
    // xtensor auto-broadcasts — no replicate() needed.

    XTensorBackend compare_lt(const XTensorBackend& other) const
    {
        return make_like(xt::eval(xt::cast<float>(m_data < other.m_data)));
    }
    XTensorBackend compare_gt(const XTensorBackend& other) const
    {
        return make_like(xt::eval(xt::cast<float>(m_data > other.m_data)));
    }
    XTensorBackend compare_le(const XTensorBackend& other) const
    {
        return make_like(xt::eval(xt::cast<float>(m_data <= other.m_data)));
    }
    XTensorBackend compare_ge(const XTensorBackend& other) const
    {
        return make_like(xt::eval(xt::cast<float>(m_data >= other.m_data)));
    }
    XTensorBackend compare_eq(const XTensorBackend& other) const
    {
        return make_like(xt::eval(xt::cast<float>(m_data == other.m_data)));
    }
    XTensorBackend compare_lt_scalar(float v) const
    { return make_like(xt::eval(xt::cast<float>(m_data < v))); }
    XTensorBackend compare_gt_scalar(float v) const
    { return make_like(xt::eval(xt::cast<float>(m_data > v))); }
    XTensorBackend compare_le_scalar(float v) const
    { return make_like(xt::eval(xt::cast<float>(m_data <= v))); }
    XTensorBackend compare_ge_scalar(float v) const
    { return make_like(xt::eval(xt::cast<float>(m_data >= v))); }
    XTensorBackend compare_eq_scalar(float v) const
    { return make_like(xt::eval(xt::cast<float>(m_data == v))); }

    // ------------------------------------------------------------------
    // Math
    // ------------------------------------------------------------------

    XTensorBackend sqrt() const { return make_like(xt::eval(xt::sqrt(m_data))); }
    XTensorBackend square() const { return make_like(xt::eval(xt::square(m_data))); }
    XTensorBackend abs() const { return make_like(xt::eval(xt::abs(m_data))); }
    XTensorBackend relu() const { return make_like(xt::eval(xt::maximum(m_data, 0.0f))); }
    XTensorBackend leaky_relu(float alpha) const
    {
        return make_like(xt::eval(xt::where(m_data > 0.0f, m_data, alpha * m_data)));
    }
    XTensorBackend clamp(float min_val, float max_val) const
    {
        return make_like(xt::eval(xt::clip(m_data, min_val, max_val)));
    }
    void clamp_inplace(float min_val, float max_val)
    {
        m_data = xt::clip(m_data, min_val, max_val);
    }

    // ------------------------------------------------------------------
    // Reductions
    // ------------------------------------------------------------------

    float mean_squared_error(const XTensorBackend& target) const
    {
        if (m_shape != target.m_shape)
        {
            std::ostringstream oss;
            oss << "Shape mismatch in mean_squared_error: "
                << rows() << "x" << cols() << " vs "
                << target.rows() << "x" << target.cols();
            NN_LOG_ERROR(oss.str());
            throw std::invalid_argument("Shape mismatch for mean_squared_error");
        }
        return xt::sum(xt::square(m_data - target.m_data))() /
               static_cast<float>(m_data.size());
    }

    float norm() const
    {
        return std::sqrt(static_cast<float>(xt::sum(xt::square(m_data))()));
    }

    float sum() const { return static_cast<float>(xt::sum(m_data)()); }
    float mean() const
    {
        if (m_data.size() == 0) return 0.0f;
        return static_cast<float>(xt::mean(m_data)());
    }

    XTensorBackend sum_rows() const
    {
        // Returns {rows, 1}.
        auto s = xt::eval(xt::sum(m_data, {1}));
        s.reshape({m_data.shape(0), 1});
        return XTensorBackend(s);
    }

    XTensorBackend sum_cols() const
    {
        // Returns {1, cols}.
        auto s = xt::eval(xt::sum(m_data, {0}));
        s.reshape({1, m_data.shape(1)});
        return XTensorBackend(s);
    }

    bool hasNaN() const
    {
        return static_cast<bool>(xt::any(xt::isnan(m_data))());
    }

    bool operator==(const XTensorBackend& other) const
    {
        return xt::allclose(m_data, other.m_data);
    }
    bool operator!=(const XTensorBackend& other) const { return !(*this == other); }

    // ------------------------------------------------------------------
    // Slicing and views
    // ------------------------------------------------------------------

    XTensorBackend row(Index i) const
    {
        if (i >= rows()) throw std::out_of_range("Index out of range");
        // Return shape {1, cols} to match Eigen convention.
        auto r = xt::eval(xt::view(m_data, i, xt::all()));
        r.reshape({1, m_data.shape(1)});
        return XTensorBackend(r);
    }

    XTensorBackend col(Index j) const
    {
        if (j >= cols()) throw std::out_of_range("Index out of range");
        auto c = xt::eval(xt::view(m_data, xt::all(), j));
        c.reshape({m_data.shape(0), 1});
        return XTensorBackend(c);
    }

    XTensorBackend leftCols(Index n) const
    {
        return XTensorBackend(
            xt::eval(xt::view(m_data, xt::all(), xt::range(Index{0}, n))));
    }

    XTensorBackend topRows(Index n) const
    {
        return XTensorBackend(
            xt::eval(xt::view(m_data, xt::range(Index{0}, n), xt::all())));
    }

    XTensorBackend block(Index r, Index c, Index block_rows, Index block_cols) const
    {
        if (m_shape.size() != 2) throw std::invalid_argument("block valid only for 2D");
        if (r + block_rows > m_shape[0] || c + block_cols > m_shape[1])
            throw std::out_of_range("Block indices out of range");
        return XTensorBackend(xt::eval(
            xt::view(m_data,
                     xt::range(r, r + block_rows),
                     xt::range(c, c + block_cols))));
    }

    void setBlock(Index r, Index c, const XTensorBackend& other)
    {
        if (m_shape.size() != 2 || other.m_shape.size() != 2)
            throw std::invalid_argument("setBlock valid only for 2D");
        if (r + other.rows() > m_shape[0] || c + other.cols() > m_shape[1])
            throw std::invalid_argument("Block indices out of range");
        xt::view(m_data,
                 xt::range(r, r + other.rows()),
                 xt::range(c, c + other.cols())) = other.m_data;
    }

    XTensorBackend slice(std::span<const int> indices) const
    {
        const Index n = indices.size();
        xt::xarray<float> result = xt::zeros<float>({n, m_data.shape(1)});
        for (Index i = 0; i < n; ++i)
        {
            if (indices[i] < 0 || static_cast<Index>(indices[i]) >= rows())
                throw std::out_of_range("Index out of range");
            xt::view(result, i, xt::all()) =
                xt::view(m_data, static_cast<Index>(indices[i]), xt::all());
        }
        return XTensorBackend(std::move(result));
    }

    // ------------------------------------------------------------------
    // Mutators
    // ------------------------------------------------------------------

    void fill(float v)     { m_data.fill(v); }
    void set_zero()        { m_data.fill(0.0f); }
    void set_ones()        { m_data.fill(1.0f); }

    const float* data_ptr() const noexcept { return m_data.data(); }
    float* mutable_data_ptr() noexcept     { return m_data.data(); }

    // ------------------------------------------------------------------
    // Gradient
    // ------------------------------------------------------------------

    XTensorBackend get_grad() const
    {
        if (m_grad_backend) return *m_grad_backend;
        return XTensorBackend(m_shape);
    }

    void set_grad(const XTensorBackend& other)
    {
        if (!m_grad_backend)
            m_grad_backend = std::make_unique<XTensorBackend>(m_shape);
        m_grad_backend->m_data = other.m_data;
    }

    void zero_grad()
    {
        if (m_grad_backend) m_grad_backend->m_data.fill(0.0f);
    }

    XTensorBackend& grad_ref()
    {
        if (!m_grad_backend)
            m_grad_backend = std::make_unique<XTensorBackend>(m_shape);
        return *m_grad_backend;
    }

   private:
    xt::xarray<float>  m_data;
    std::vector<Index> m_shape;
    mutable std::unique_ptr<XTensorBackend> m_grad_backend;

    XTensorBackend make_like(xt::xarray<float> data) const
    {
        XTensorBackend result;
        result.m_data  = std::move(data);
        result.m_shape = m_shape;
        return result;
    }
};

} // namespace nn
```

---

## Step 3 — Update Backend.hpp

File: `include/nn/Backend.hpp`

Old content:
```cpp
#pragma once
#include "nn/tensor/eigen/EigenTensorBackend.hpp"
namespace nn {
    using Backend = EigenTensorBackend;
}
```

New content:
```cpp
#pragma once
#include "nn/tensor/xtensor/XTensorBackend.hpp"
namespace nn {
    using Backend = XTensorBackend;
}
```

---

## Step 4 — Update DeviceTensorBackend.hpp

File: `include/nn/tensor/DeviceTensorBackend.hpp`

1. Remove the line `#include "nn/tensor/eigen/EigenTensorBackend.hpp"`.
2. Add the line `#include "nn/tensor/xtensor/XTensorBackend.hpp"`.
3. Find all constructors that take `const Eigen::MatrixXf&` or `Eigen::MatrixXf&&` — delete them.
4. Find the member `EigenTensorBackend m_host` — rename to `XTensorBackend m_host`.
5. All delegate calls `m_host(...)` stay identical — XTensorBackend has the same method names.

Specific lines to find and act on:

**Line:** `explicit DeviceTensorBackend(const Eigen::MatrixXf& data) : m_host(data) {}`
**Action:** Delete this constructor.

**Line:** `explicit DeviceTensorBackend(Eigen::MatrixXf&& data) : m_host(std::move(data)) {}`
**Action:** Delete this constructor.

**All occurrences of `EigenTensorBackend`** in this file:
**Action:** Replace with `XTensorBackend`.

---

## Step 5 — Replace VendorEigenParallel → VendorXtensorParallel

`cmake/VendorXtensorParallel.cmake` was already created in Step 1b.

Do NOT delete `cmake/VendorEigenParallel.cmake` yet — that is done in Step 14.
`cmake/VendorIncludes.cmake` was already updated in Step 1c.

---

## Step 6 — Replace EigenParallel.hpp → XtensorParallel.hpp

Create `include/nn/utility/XtensorParallel.hpp`:

```cpp
#pragma once
// Thread-count initializer for xtensor + OpenMP.
// Replaces EigenParallel.hpp.

#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace util
{
inline void initializeXtensorParallel(int numThreads = 0)
{
    if (numThreads <= 0)
    {
        unsigned hc = std::thread::hardware_concurrency();
        numThreads = (hc > 0) ? static_cast<int>(hc) : 1;
    }
#ifdef _OPENMP
    omp_set_num_threads(numThreads);
#if _OPENMP >= 200805
    omp_set_max_active_levels(1);
#else
    omp_set_nested(0);
#endif
#endif
}
} // namespace util
```

Then find every file that `#include`s `nn/utility/EigenParallel.hpp` and calls
`util::initializeEigenParallel(...)`:

```bash
grep -rn "EigenParallel\|initializeEigenParallel" /path/to/project --include="*.hpp" --include="*.cpp"
```

For each hit:
- Replace `#include "nn/utility/EigenParallel.hpp"` → `#include "nn/utility/XtensorParallel.hpp"`
- Replace `util::initializeEigenParallel(` → `util::initializeXtensorParallel(`

---

## Step 7 — Rewrite vectorizationCheck.cpp

File: `src/core/utility/vectorizationCheck.cpp`

Remove the `#include <Eigen/Dense>` line and all `EIGEN_VECTORIZE_*` ifdef blocks.
Replace the body of `printVectorizationSupport()` with:

```cpp
void printVectorizationSupport()
{
    std::string info = "xtensor SIMD: ";
#ifdef XSIMD_VERSION_MAJOR
    info += "xsimd " + std::to_string(XSIMD_VERSION_MAJOR) + "."
          + std::to_string(XSIMD_VERSION_MINOR) + " ";
#endif
#ifdef _OPENMP
    info += "OpenMP " + std::to_string(_OPENMP) + " ";
#endif
    NN_LOG_INFO(info);
}
```

Also update `include/nn/utility/vectorizationCheck.hpp` — remove any Eigen includes there too.

---

## Step 8 — Fix mat_file_utils.cpp

File: `src/core/dataLoaders/io/mat_file_utils.cpp`

This file uses `matioCpp::to_eigen()` as an intermediary to copy MATLAB data into
`nn::Tensor`. After removing Eigen, use matioCpp's raw data pointer instead.

Remove:
```cpp
#include <matioCpp/EigenConversions.h>
```

Replace the private helper `to_tensor_from_eigen_matrix` and all uses of it:

Old (remove this):
```cpp
auto to_tensor_from_eigen_matrix(const Eigen::MatrixXf& eigen_matrix) -> nn::Tensor
{
    nn::Tensor result(
        static_cast<size_t>(eigen_matrix.rows()),
        static_cast<size_t>(eigen_matrix.cols()));
    std::copy_n(eigen_matrix.data(), ...);
    return result;
}
```

New helper (add this in the anonymous namespace):
```cpp
template <typename T>
auto to_tensor_from_raw(const T* data, size_t rows, size_t cols) -> nn::Tensor
{
    nn::Tensor result(rows, cols);
    float* dst = result.mutable_data_ptr();
    for (size_t i = 0; i < rows * cols; ++i)
        dst[i] = static_cast<float>(data[i]);
    return result;
}
```

Replace `to_tensor_from_multi`:
```cpp
template <typename T>
auto to_tensor_from_multi(const matioCpp::Variable& variable) -> std::optional<nn::Tensor>
{
    auto multi = variable.template asMultiDimensionalArray<T>();
    auto dims = multi.dimensions();
    size_t rows = (dims.size() > 0) ? dims[0] : 1;
    size_t cols = (dims.size() > 1) ? dims[1] : 1;
    return to_tensor_from_raw(multi.data(), rows, cols);
}
```

Replace `to_tensor_from_vector`:
```cpp
template <typename T>
auto to_tensor_from_vector(const matioCpp::Variable& variable) -> std::optional<nn::Tensor>
{
    auto vec = variable.template asVector<T>();
    return to_tensor_from_raw(vec.data(), vec.size(), 1);
}
```

---

## Step 9 — Remove EigenBan Infrastructure

### 9a. Edit top-level `CMakeLists.txt`

Find: `include(cmake/EigenBan.cmake)`  
Action: Delete this line.

### 9b. Edit `cmake/DevAndAnalysisTargets.cmake`

Find the block starting with:
```cmake
add_custom_target(check_eigen_leaks ...
```
Delete the entire `if(Python3_FOUND) ... check_eigen_leaks ... endif()` block.

Also remove any description comment that mentions `check_eigen_leaks`.

### 9c. Find and remove all calls to `nn_allow_eigen()` and `nn_disallow_eigen()`

Run:
```bash
grep -rn "nn_allow_eigen\|nn_disallow_eigen\|nn_eigen_guard\|NN_ALLOW_EIGEN" \
    /path/to/project --include="CMakeLists.txt" --include="*.cmake"
```

Delete every line found. These calls are scattered in:
- `src/core/layers/CMakeLists.txt`
- `src/core/tensor/CMakeLists.txt`
- Any other CMakeLists.txt that had Eigen-specific targets

---

## Step 10 — Update All CMakeLists.txt Files

For **every** CMakeLists.txt that contains `Eigen3::Eigen`:

### Pattern to apply:

**Find:** `Eigen3::Eigen`  
**Replace with:** `xtensor xtensor-blas`

**Find:** `configure_eigen_parallel_target(<target_name>)`  
**Replace with:** `configure_xtensor_parallel_target(<target_name>)`

**Find:** `${EIGEN3_INCLUDE_DIR}`  
**Delete** this line entirely (xtensor is found via FetchContent; no manual include dir needed).

### Complete list of files to edit (apply pattern above to each):

1. `src/core/layers/CMakeLists.txt`
2. `src/core/layers/tests/CMakeLists.txt`
3. `src/core/tensor/CMakeLists.txt`
4. `src/core/tensor/tests/CMakeLists.txt` (applies to all 3 test targets in this file)
5. `src/core/dataLoaders/CMakeLists.txt`
6. `src/core/dataLoaders/10.1117/CMakeLists.txt` (two targets: both occurrences)
7. `src/core/dataLoaders/10.1117/tests/CMakeLists.txt`
8. `src/core/dataLoaders/10.1117/tests/fuzz/CMakeLists.txt`
9. `src/core/dataLoaders/tests/CMakeLists.txt`
10. `src/core/dataLoaders/tests/MatTestUtils/CMakeLists.txt`
11. `src/core/dataLoaders/10.1117/tests/windowing/CMakeLists.txt`
12. `src/core/initializers/tests/CMakeLists.txt`
13. `src/core/optimizers/tests/CMakeLists.txt`
14. `src/core/statistics/CMakeLists.txt`
15. `src/core/utility/CMakeLists.txt`
16. `src/core/utility/tests/CMakeLists.txt`
17. `src/core/wave/CMakeLists.txt`

---

## Step 11 — Remove Eigen from PackageChecking.cmake

File: `cmake/PackageChecking.cmake`

Delete **only** this line (already done in Step 1d, verify it is gone):
```cmake
find_package(Eigen3 REQUIRED NO_MODULE)
```

Also delete the comment block above it if it specifically mentions Eigen:
```cmake
# Find Eigen
```

Do not touch `find_package(BLAS)` or `find_package(LAPACK)` — xtensor-blas needs them.

---

## Step 12 — Scan and Purge Residual Eigen References

Run this scan command. Every match is a file that needs editing:
```bash
grep -rn "Eigen\|eigen" \
    include/ src/ cmake/ \
    --include="*.hpp" --include="*.cpp" --include="*.cmake" --include="CMakeLists.txt" \
    | grep -v "xtensorMigration.md" \
    | grep -v "^Binary"
```

For each match, apply the appropriate fix:

### In source/header files (`.hpp`, `.cpp`)

| Pattern found | Action |
|---|---|
| `#include <Eigen/Dense>` | Delete line |
| `#include <Eigen/...>` | Delete line |
| `#include "nn/tensor/eigen/EigenTensorBackend.hpp"` | Replace with `#include "nn/tensor/xtensor/XTensorBackend.hpp"` |
| `#include "nn/utility/EigenParallel.hpp"` | Replace with `#include "nn/utility/XtensorParallel.hpp"` |
| `nn::EigenTensorBackend` | Replace with `nn::XTensorBackend` |
| `EigenTensorBackend` (bare) | Replace with `XTensorBackend` |
| `Eigen::MatrixXf` | Replace with `xt::xarray<float>` (only if the variable is being used as a tensor; if it's a true intermediary for matioCpp, use Step 8 instead) |
| `Eigen::Index` | Replace with `nn::Index` |
| `initializeEigenParallel` | Replace with `initializeXtensorParallel` |
| `EIGEN_VECTORIZE_AVX` etc. in comments | Delete the comment block |

### In CMake files (`.cmake`, `CMakeLists.txt`)

| Pattern found | Action |
|---|---|
| `find_package(Eigen3` | Delete line |
| `Eigen3::Eigen` | Replace with `xtensor xtensor-blas` |
| `${EIGEN3_INCLUDE_DIR}` | Delete line |
| `configure_eigen_parallel_target(` | Replace with `configure_xtensor_parallel_target(` |
| `EIGEN_USE_BLAS` etc. in `target_compile_definitions` | Delete (handled by VendorXtensorParallel) |
| `nn_allow_eigen(` | Delete line |
| `nn_disallow_eigen(` | Delete line |
| `include(cmake/EigenBan.cmake)` | Delete line |
| `include(cmake/VendorEigenParallel.cmake)` | Delete line (already replaced in Step 1c) |

### In comments and docstrings

For every `.hpp` and `.cpp` file, replace in comments:
- `"Eigen CPU"` → `"xtensor CPU"`
- `"Eigen-backed"` → `"xtensor-backed"`
- `"Eigen::MatrixXf"` in comments → `"xt::xarray<float>"`
- `"EigenTensorBackend"` in comments → `"XTensorBackend"`
- File header `@brief Eigen-backed host storage` → `@brief xtensor-backed host storage`

---

## Step 13 — Redesign LSTM for 3D Support

File: `include/nn/layers/lstm/LSTMLayer.hpp`

### Design

After this step, `LSTMLayerImpl::forward()` accepts:
- **2D input `(T, D)`** — single sequence, backward-compatible. Returns `(T, H)`.
- **3D input `(B, T, D)`** — batch of B sequences. Returns `(B, T, H)`.

Detection: `input.get_shape().size() == 3` → batch mode.

Hidden state:
- 2D mode: `h0_` shape `(1, H)`, `c0_` shape `(1, H)` — unchanged.
- 3D mode: on entry, if `h0_.rows() != B`, expand to `(B, H)` zeros.

The `Module<Backend>::forward(const Tensor&)` signature is unchanged — both cases pass a `Tensor`.

### Complete rewrite of forward() and backward()

Replace the `forward()` method body in `LSTMLayerImpl` with:

```cpp
auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
{
    requires_grad_ = requires_grad;
    const auto& shape = input.get_shape();

    if (shape.size() == 3)
        return forward_3d(input, requires_grad);
    else
        return forward_2d(input, requires_grad);
}

auto backward(const Tensor& grad_output) -> Tensor override
{
    if (grad_output.get_shape().size() == 3)
        return backward_3d(grad_output);
    else
        return Tensor(_bptt_apply(cache_, nn::Tensor(grad_output), false));
}
```

Add these private helpers inside `LSTMLayerImpl`:

```cpp
private:

// 2D single-sample forward — existing logic, unchanged.
auto forward_2d(const Tensor& input, bool requires_grad) -> Tensor
{
    const int T = static_cast<int>(input.rows());
    const int D = static_cast<int>(input.cols());
    if (D != input_size_)
        throw std::invalid_argument("LSTMLayerImpl::forward: input cols=" +
                                    std::to_string(D) + " != input_size=" +
                                    std::to_string(input_size_));
    if (requires_grad) { cache_.clear(); cache_.reserve(T); }

    nn::Tensor h = h0_;
    nn::Tensor c = c0_;
    nn::Tensor all_h(static_cast<nn::Index>(T),
                     static_cast<nn::Index>(hidden_size_));

    for (int t = 0; t < T; ++t)
    {
        nn::Tensor x_t = nn::Tensor(input).row(static_cast<nn::Index>(t));
        nn::Tensor pre  = x_t.matmul(W_.transpose())
                             .add(h.matmul(U_.transpose()))
                             .add(b_.transpose());

        nn::Tensor i_g = lstm_sigmoid(pre.block(0, 0,                1, hidden_size_));
        nn::Tensor f_g = lstm_sigmoid(pre.block(0, 1*hidden_size_,   1, hidden_size_));
        nn::Tensor o_g = lstm_sigmoid(pre.block(0, 2*hidden_size_,   1, hidden_size_));
        nn::Tensor g_g = lstm_tanh(  pre.block(0, 3*hidden_size_,   1, hidden_size_));

        nn::Tensor c_new = (f_g * c).add(i_g * g_g);
        nn::Tensor tc    = lstm_tanh(c_new);
        nn::Tensor h_new = o_g * tc;

        if (requires_grad)
            cache_.push_back({x_t, h, c, i_g, f_g, o_g, g_g, c_new, tc, h_new});

        all_h.setBlock(static_cast<nn::Index>(t), 0, h_new);
        h = h_new;
        c = c_new;
    }
    h0_ = h; c0_ = c;
    return Tensor(all_h);
}

// 3D batch forward — input shape (B, T, D), output shape (B, T, H).
auto forward_3d(const Tensor& input, bool requires_grad) -> Tensor
{
    const auto& shape = input.get_shape();
    const int B = static_cast<int>(shape[0]);
    const int T = static_cast<int>(shape[1]);
    const int D = static_cast<int>(shape[2]);

    if (D != input_size_)
        throw std::invalid_argument("LSTMLayerImpl::forward_3d: input D=" +
                                    std::to_string(D) + " != input_size=" +
                                    std::to_string(input_size_));

    if (requires_grad)
    {
        batch_caches_.clear();
        batch_caches_.resize(B);
        for (auto& bc : batch_caches_) { bc.clear(); bc.reserve(T); }
    }

    // Output: (B, T, H)
    nn::Tensor all_out = nn::Tensor::zeros(
        static_cast<nn::Index>(B),
        static_cast<nn::Index>(T),
        static_cast<nn::Index>(hidden_size_));

    for (int b = 0; b < B; ++b)
    {
        // Extract sample b: view (T, D) from (B, T, D)
        nn::Tensor sample(static_cast<nn::Index>(T),
                          static_cast<nn::Index>(D));
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D; ++d)
                sample.at(static_cast<nn::Index>(t),
                          static_cast<nn::Index>(d)) =
                    input.at(static_cast<nn::Index>(b),
                             static_cast<nn::Index>(t),
                             static_cast<nn::Index>(d));

        // Run single-sample forward with its own h0/c0 = zeros.
        h0_.set_zero(); c0_.set_zero();
        nn::Tensor h = h0_;
        nn::Tensor c = c0_;

        for (int t = 0; t < T; ++t)
        {
            nn::Tensor x_t = sample.row(static_cast<nn::Index>(t));
            nn::Tensor pre = x_t.matmul(W_.transpose())
                                .add(h.matmul(U_.transpose()))
                                .add(b_.transpose());

            nn::Tensor i_g = lstm_sigmoid(pre.block(0, 0,              1, hidden_size_));
            nn::Tensor f_g = lstm_sigmoid(pre.block(0, 1*hidden_size_, 1, hidden_size_));
            nn::Tensor o_g = lstm_sigmoid(pre.block(0, 2*hidden_size_, 1, hidden_size_));
            nn::Tensor g_g = lstm_tanh(  pre.block(0, 3*hidden_size_, 1, hidden_size_));

            nn::Tensor c_new = (f_g * c).add(i_g * g_g);
            nn::Tensor tc    = lstm_tanh(c_new);
            nn::Tensor h_new = o_g * tc;

            if (requires_grad)
                batch_caches_[b].push_back(
                    {x_t, h, c, i_g, f_g, o_g, g_g, c_new, tc, h_new});

            // Write h_new into all_out[b, t, :]
            for (int hh = 0; hh < hidden_size_; ++hh)
                all_out.at(static_cast<nn::Index>(b),
                           static_cast<nn::Index>(t),
                           static_cast<nn::Index>(hh)) = h_new.at(0, static_cast<nn::Index>(hh));

            h = h_new; c = c_new;
        }
    }
    return Tensor(all_out);
}

// 3D backward: grad_output shape (B, T, H), returns grad_input shape (B, T, D).
auto backward_3d(const Tensor& grad_output) -> Tensor
{
    const auto& shape = grad_output.get_shape();
    const int B = static_cast<int>(shape[0]);
    const int T = static_cast<int>(shape[1]);

    if (static_cast<int>(batch_caches_.size()) != B)
        throw std::runtime_error("backward_3d: batch size mismatch with forward cache");

    nn::Tensor dW_accum = nn::Tensor::zeros(dW_.rows(), dW_.cols());
    nn::Tensor dU_accum = nn::Tensor::zeros(dU_.rows(), dU_.cols());
    nn::Tensor db_accum = nn::Tensor::zeros(db_.rows(), db_.cols());

    nn::Tensor dx_all = nn::Tensor::zeros(
        static_cast<nn::Index>(B),
        static_cast<nn::Index>(T),
        static_cast<nn::Index>(input_size_));

    for (int b = 0; b < B; ++b)
    {
        // Extract grad slice for sample b: (T, H)
        nn::Tensor grad_b(static_cast<nn::Index>(T),
                          static_cast<nn::Index>(hidden_size_));
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < hidden_size_; ++h)
                grad_b.at(static_cast<nn::Index>(t), static_cast<nn::Index>(h)) =
                    grad_output.at(static_cast<nn::Index>(b),
                                   static_cast<nn::Index>(t),
                                   static_cast<nn::Index>(h));

        auto [dW_b, dU_b, db_b, dx_b] = _bptt_pure(batch_caches_[b], grad_b);

        for (nn::Index k = 0; k < static_cast<nn::Index>(dW_accum.size()); ++k)
            dW_accum.at(k) += dW_b.at(k);
        for (nn::Index k = 0; k < static_cast<nn::Index>(dU_accum.size()); ++k)
            dU_accum.at(k) += dU_b.at(k);
        for (nn::Index k = 0; k < static_cast<nn::Index>(db_accum.size()); ++k)
            db_accum.at(k) += db_b.at(k);

        // Write dx_b into dx_all[b, :, :]
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < input_size_; ++d)
                dx_all.at(static_cast<nn::Index>(b),
                           static_cast<nn::Index>(t),
                           static_cast<nn::Index>(d)) =
                    dx_b.at(static_cast<nn::Index>(t), static_cast<nn::Index>(d));
    }

    W_.set_grad(dW_accum); U_.set_grad(dU_accum); b_.set_grad(db_accum);
    dW_ = dW_accum; dU_ = dU_accum; db_ = db_accum;

    return Tensor(dx_all);
}
```

### Member additions

Add `batch_caches_` for 3D mode (if not already present). It stores per-sample caches
when `forward_3d` is called:
```cpp
std::vector<std::vector<LSTMStepCache>> batch_caches_;
```

### New shape contract (add to docstring)

```
forward(Tensor{T, D})    → Tensor{T, H}    — single sequence, 2D
forward(Tensor{B, T, D}) → Tensor{B, T, H} — batched, 3D
backward(Tensor{T, H})   → Tensor{T, D}    — single
backward(Tensor{B, T, H})→ Tensor{B, T, D} — batch; grads W/U/b accumulated over B
```

---

## Step 14 — Delete Deprecated Files and Directories

Only delete after the build passes (Step 16). Then:

```bash
# Delete EigenTensorBackend
rm include/nn/tensor/eigen/EigenTensorBackend.hpp
rmdir include/nn/tensor/eigen/

# Delete EigenParallel helper
rm include/nn/utility/EigenParallel.hpp

# Delete EigenBan system
rm cmake/EigenBan.cmake
rm cmake/EigenBan.hpp
rm cmake/VendorEigenParallel.cmake

# Delete check_eigen_leaks script (if present)
rm scripts/check_eigen_leaks.py
```

---

## Step 15 — Update Wiki

Edit each file below. Replace every Eigen reference with the xtensor equivalent.
Only the specific changes needed are listed — do not rewrite files from scratch.

### `.wiki/Home.md`

Find:
```
- Eigen 3.4+
```
Replace with:
```
- xtensor 0.25+ (header-only, via FetchContent)
- xtensor-blas 0.21+ (BLAS-backed matmul)
```

Find:
```
Multiple Backend Support: Eigen (CPU) and OpenCL (GPU) tensor backends
```
Replace with:
```
Multiple Backend Support: xtensor (CPU) and OpenCL (GPU) tensor backends
```

### `.wiki/Architecture.md`

In the mermaid diagram, find:
```
Eigen[Eigen CPU]
...
Tensor --> Eigen
```
Replace with:
```
xtensor[xtensor CPU]
...
Tensor --> xtensor
```

Find in text:
```
Template-based Backend Selection: Layers use template parameters (`<Backend>`) to select between Eigen and OpenCL at compile time.
```
Replace with:
```
Template-based Backend Selection: Layers use template parameters (`<Backend>`) to select between xtensor and OpenCL at compile time.
```

### `.wiki/Core/Tensor.md`

Find:
```
std::vector<float> data_;        // host data (Eigen backend)
```
Replace with:
```
xt::xarray<float> data_;         // host data (xtensor backend)
```

Find:
```
nn::EigenTensorBackend - CPU operations
```
Replace with:
```
nn::XTensorBackend - CPU operations (xtensor-backed)
```

In the mermaid diagram:
```
CPU[Eigen ops]
```
→
```
CPU[xtensor ops]
```

### `.wiki/Core/Device.md`

In the mermaid diagram:
```
eigen[Eigen CPU]
...
init -->|CPU| eigen
```
Replace with:
```
xtensor[xtensor CPU]
...
init -->|CPU| xtensor
```

### `.wiki/Core/Training.md`

Find:
```cpp
typename LossType = MSELossImpl<nn::EigenTensorBackend>>
```
Replace with:
```cpp
typename LossType = MSELossImpl<nn::Backend>>
```

Find:
```
LossType template (default MSELossImpl<EigenTensorBackend>)
```
Replace with:
```
LossType template (default MSELossImpl<nn::Backend>)
```

### `.wiki/Core/Layers.md`

Find:
```cpp
class LSTMLayer : public Module<nn::EigenTensorBackend>
```
Replace with:
```cpp
class LSTMLayer : public Module<nn::Backend>
```

Find:
```cpp
// File: include/nn/layers/eigen/Layers.hpp
nn::layers::Linear<nn::EigenTensorBackend> fc1(128, 64);
nn::layers::Linear<nn::EigenTensorBackend> fc2(64, 32);
```
Replace with:
```cpp
// File: include/nn/layers/Layers.hpp
nn::Linear fc1(128, 64);
nn::Linear fc2(64, 32);
```

Add a note under the LSTM entry:
```
**3D support**: `forward(Tensor{B,T,D})` returns `Tensor{B,T,H}`. Single-sequence
`forward(Tensor{T,D})` → `Tensor{T,H}` remains backward-compatible.
```

### `.wiki/Concepts/LSTM-and-BPTT.md`

Find:
```cpp
class LeakyBPTT : public Module<EigenTensorBackend>
```
Replace with:
```cpp
class LeakyBPTT : public Module<nn::Backend>
```

### `.wiki/Concepts/Residual-Blocks.md`

Find:
```cpp
class ResNetBlock : public Module<EigenTensorBackend>
```
Replace with:
```cpp
class ResNetBlock : public Module<nn::Backend>
```

### `.wiki/Core/LinearAlgebra.md`

Find:
```
These operations are compute-intensive and benefit from optimized libraries (Eigen, OpenCL BLAS).
```
Replace with:
```
These operations are compute-intensive and benefit from optimized libraries (xtensor-blas + BLAS, OpenCL).
```

Find:
```
// File: src/core/tensor/EigenTensorBackend.cpp
```
Replace with:
```
// File: include/nn/tensor/xtensor/XTensorBackend.hpp
```

### `.wiki/Guides/Build-System.md`

Find and replace each of these lines:

| Old | New |
|---|---|
| `cmake/EigenBan.cmake \| Compile-time poison to prevent direct Eigen include leaks` | Delete row |
| `cmake/PackageChecking.cmake \| System deps: Eigen3, OpenMP, ...` | `cmake/PackageChecking.cmake \| System deps: OpenMP, SDL2, BLAS/LAPACK, OpenCL` |
| `VendorEigenParallel.cmake \| Eigen parallelization settings` | `VendorXtensorParallel.cmake \| xtensor OpenMP/BLAS wiring` |
| `Eigen3 — linear algebra` | `xtensor 0.25 — N-D tensor library (FetchContent)` |
| `cmake --build ... --target check_eigen_leaks` | Delete line |
| `# Check for Eigen include leaks` | Delete line and the command below |

---

## Step 16 — Build and Verify

Run these commands **in order**. Do not skip steps.

```bash
# 1. Reconfigure (FetchContent downloads xtensor on first run)
cmake --preset=max-performance

# 2. Build tensor target first (isolate backend issues)
cmake --build out/build/max-performance --target tensor -j$(nproc)

# 3. Build all core tests
cmake --build out/build/max-performance --target core_gtest trainer_gtest -j$(nproc)

# 4. Build experiment targets
cmake --build out/build/max-performance --target experiment04_lib experiment04 -j$(nproc)

# 5. Run all tests
ctest --test-dir out/build/max-performance --output-on-failure -j4

# 6. Verify zero Eigen references remain (expected: zero output)
grep -rn "Eigen\|eigen" include/ src/ cmake/ \
    --include="*.hpp" --include="*.cpp" --include="*.cmake" --include="CMakeLists.txt" \
    | grep -v "xtensorMigration.md" \
    | grep -v "#.*Eigen" \
    | grep -v "^Binary"

# 7. Verify XTensorBackend is the active backend
grep -n "XTensorBackend\|Backend" include/nn/Backend.hpp
```

Expected final state:
- Step 6 produces zero matches.
- All previously passing tests still pass (no regressions).
- New 3D LSTM tests: `LSTMBatchTest.BatchEquivalence_GradsAreTwoTimeSingle`,
  `LSTMBatchTest.OutputShape` — should pass with 3D input tensors now.

---

## Troubleshooting

### `xt::linalg::dot` linker error: undefined reference to `cblas_sgemm`

**Cause**: xtensor-blas links to BLAS but the target doesn't link `${BLAS_LIBRARIES}`.  
**Fix**: Ensure `configure_xtensor_parallel_target(<target>)` is called, or manually add:
```cmake
target_link_libraries(<target> PRIVATE ${BLAS_LIBRARIES} xtensor-blas)
```

### `xt::sum(arr, {1})()` compile error: no `operator()()`

**Cause**: `xt::sum` returns an `xexpression`, not `xarray`. Must evaluate first.  
**Fix**: Use `xt::eval(xt::sum(arr, {1}))` before calling `()`.

### `m_data.reshape(...)` does not compile

**Cause**: `xt::xarray::reshape` takes `xt::dynamic_shape<size_t>`, not `std::vector<size_t>`.  
**Fix**:
```cpp
xt::dynamic_shape<Index> xshape(new_shape.begin(), new_shape.end());
m_data.reshape(xshape);
```

### `xt::view(m_data, i, xt::all())` returns 1-D but caller expects 2-D

**Cause**: xtensor view reduces dimensions when indexing with a scalar.  
**Fix**: Use `xt::range(i, i+1)` to keep dimension:
```cpp
xt::view(m_data, xt::range(i, i+1), xt::all())  // shape {1, cols}
```
Or reshape after eval:
```cpp
auto r = xt::eval(xt::view(m_data, i, xt::all()));
r.reshape({1, m_data.shape(1)});
```

### `xt::allclose` not found

**Cause**: `xt::allclose` is in `<xtensor/xmath.hpp>` or `<xtensor/xarray.hpp>` depending on version.  
**Fix**: Add `#include <xtensor/xmath.hpp>`. If still not found, implement manually:
```cpp
bool approx_equal(const xt::xarray<float>& a, const xt::xarray<float>& b,
                  float tol = 1e-5f)
{
    return static_cast<bool>(xt::all(xt::abs(a - b) <= tol)());
}
```

### DeviceTensorBackend fails to compile

**Cause**: Constructor `DeviceTensorBackend(const Eigen::MatrixXf&)` was removed but something still calls it.  
**Fix**: Search all callers: `grep -rn "DeviceTensorBackend.*MatrixXf\|DeviceTensorBackend.*Eigen"`. Any remaining call site must be updated to pass an `XTensorBackend` or `xt::xarray<float>` instead.

### matioCpp conversion fails at runtime

**Cause**: `matioCpp::MultiDimensionalArray<T>::data()` pointer order (column-major from MATLAB)
differs from expected row-major order.  
**Fix**: Add transpose in `to_tensor_from_raw` for multi-dimensional arrays:
```cpp
// MATLAB stores column-major; transpose to row-major.
nn::Tensor result(cols, rows);  // swapped
for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c)
        result.at(c, r) = static_cast<float>(data[r + c * rows]);
```
Then call `result.transpose()` if needed. Verify with existing mat-file tests
(`dataLoaders_gtest`) after migration.

### FetchContent fails to download xtensor

**Cause**: No internet access in build environment.  
**Fix**: Clone manually and use `FetchContent_Declare` with `SOURCE_DIR`:
```cmake
FetchContent_Declare(xtensor SOURCE_DIR /path/to/local/xtensor)
```
Or install via package manager:
```bash
# Arch/Manjaro:
sudo pacman -S xtensor
# Then in CMake:
find_package(xtensor REQUIRED)
```
If using system install, replace FetchContent with `find_package(xtensor REQUIRED)` in
`cmake/VendorXtensor.cmake`.

---

*End of guide. Wait for instructions before executing any step.*
