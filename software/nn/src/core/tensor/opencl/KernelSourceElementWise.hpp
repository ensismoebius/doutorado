// Embedded OpenCL C source for the element-wise kernel program (unary/binary ops, in-place ops,
// compares, LIF step).
#pragma once

namespace nn::opencl
{

// clang-format off
inline constexpr const char* KERNEL_SOURCE_ELEMENT_WISE = R"(
__kernel void add_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] + B[idx];
}

__kernel void multiply_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] * B[idx];
}

__kernel void subtract_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] - B[idx];
}

__kernel void divide_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] / (B[idx] + 1e-8f);
}

__kernel void exp_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = exp(input[idx]);
}

__kernel void sqrt_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = sqrt(max(input[idx], 0.0f));
}

__kernel void square_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] * input[idx];
}

__kernel void add_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] + scalar;
}

__kernel void multiply_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] * scalar;
}

__kernel void divide_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] / (scalar + 1e-8f);
}

__kernel void add_inplace_kernel(
    __global float* A,
    __global const float* B,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    A[idx] = A[idx] + B[idx];
}

__kernel void subtract_inplace_kernel(
    __global float* A,
    __global const float* B,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    A[idx] = A[idx] - B[idx];
}

__kernel void multiply_inplace_kernel(
    __global float* A,
    __global const float* B,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    A[idx] = A[idx] * B[idx];
}

__kernel void divide_inplace_kernel(
    __global float* A,
    __global const float* B,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    A[idx] = A[idx] / (B[idx] + 1e-8f);
}

__kernel void add_scalar_inplace_kernel(
    __global float* data,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = data[idx] + scalar;
}

__kernel void multiply_scalar_inplace_kernel(
    __global float* data,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = data[idx] * scalar;
}

__kernel void divide_scalar_inplace_kernel(
    __global float* data,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = data[idx] / (scalar + 1e-8f);
}

__kernel void sqrt_inplace_kernel(
    __global float* data,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = sqrt(max(data[idx], 0.0f));
}

__kernel void square_inplace_kernel(
    __global float* data,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = data[idx] * data[idx];
}

__kernel void fill_kernel(
    __global float* data,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = value;
}

__kernel void add_col_vector_to_rows_kernel(
    __global float* data,
    __global const float* col_vector,
    const uint rows,
    const uint cols
) {
    const uint idx = get_global_id(0);
    if (idx >= rows * cols) return;
    const uint col = idx / rows;
    data[idx] = data[idx] + col_vector[col];
}

__kernel void compare_lt_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] < B[idx] ? 1.0f : 0.0f;
}

__kernel void compare_gt_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] > B[idx] ? 1.0f : 0.0f;
}

__kernel void compare_le_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] <= B[idx] ? 1.0f : 0.0f;
}

__kernel void compare_ge_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] >= B[idx] ? 1.0f : 0.0f;
}

__kernel void compare_eq_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = fabs(A[idx] - B[idx]) < 1e-6f ? 1.0f : 0.0f;
}

__kernel void compare_lt_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] < value ? 1.0f : 0.0f;
}

__kernel void compare_gt_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] > value ? 1.0f : 0.0f;
}

__kernel void compare_le_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] <= value ? 1.0f : 0.0f;
}

__kernel void compare_ge_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] >= value ? 1.0f : 0.0f;
}

__kernel void compare_eq_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = fabs(input[idx] - value) < 1e-6f ? 1.0f : 0.0f;
}

__kernel void abs_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = fabs(input[idx]);
}

__kernel void abs_inplace_kernel(
    __global float* data,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = fabs(data[idx]);
}

__kernel void clamp_kernel(
    __global const float* input,
    __global float* output,
    const float min_val,
    const float max_val,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    float v = input[idx];
    output[idx] = v < min_val ? min_val : (v > max_val ? max_val : v);
}

__kernel void clamp_inplace_kernel(
    __global float* data,
    const float min_val,
    const float max_val,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    float v = data[idx];
    data[idx] = v < min_val ? min_val : (v > max_val ? max_val : v);
}
)";
// clang-format on

} // namespace nn::opencl
