# OpenCL lhs-transposed backward benchmark (2026-05-02)

## Scope

This note records the benchmark evidence for the tuned OpenCL lhs-transposed
kernel used by Linear backward grad-weight computation.

## Runtime context

- Platform: rusticl
- Device: AMD Radeon Graphics (radeonsi, renoir, ACO, DRM 3.64, 6.19.14-arch1-1)
- Benchmark executable: out/build/max-performance/src/core/tensor/tests/tensor_perf_bench
- Iterations: NN_TENSOR_BENCH_ITERS=20

## Key rows (per_iter_ms)

Sample A:

- opencl,linear_backward_chain_1024x256x512: 19.985
- opencl,grad_weight_matmul_512x1024x256: 5.662
- opencl,grad_weight_matmul_via_transpose_probe_512x1024x256: 10.653
- opencl,rowwise_sum_512x1024: 3.546

Sample B:

- opencl,linear_backward_chain_1024x256x512: 19.981
- opencl,grad_weight_matmul_512x1024x256: 5.858
- opencl,grad_weight_matmul_via_transpose_probe_512x1024x256: 10.290
- opencl,rowwise_sum_512x1024: 3.921

## Derived comparison

- grad-weight new path vs transpose probe speedup:
  - Sample A: 10.653 / 5.662 = 1.88x
  - Sample B: 10.290 / 5.858 = 1.76x

## Notes

- The tuned lhs-transposed path was validated by targeted OpenCL and Linear tests.
- A prior non-tiled direct kernel variant regressed performance; this note
  captures the post-tiling stabilized behavior only.
