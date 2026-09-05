// Embedded OpenCL C source for the fused kernel program (matmul+bias+activation combos).
#pragma once

namespace nn::opencl
{

// clang-format off
inline constexpr const char* KERNEL_SOURCE_FUSED = R"(
__kernel void mul_add_kernel(
    __global const float* a,
    __global const float* b,
    __global const float* c,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = a[idx] * b[idx] + c[idx];
}

__kernel void mul_add_sigmoid_kernel(
    __global const float* a,
    __global const float* b,
    __global const float* c,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    float val = a[idx] * b[idx] + c[idx];
    output[idx] = 1.0f / (1.0f + exp(-val));
}

__kernel void relu_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = fmax(input[idx], 0.0f);
}

__kernel void relu_inplace_kernel(
    __global float* data,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = fmax(data[idx], 0.0f);
}

__kernel void leaky_relu_kernel(
    __global const float* input,
    __global float* output,
    const float alpha,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] > 0.0f ? input[idx] : alpha * input[idx];
}

__kernel void add_bias_kernel(
    __global const float* input,
    __global const float* bias,
    __global float* output,
    const uint rows,
    const uint cols
) {
    const uint idx = get_global_id(0);
    if (idx >= rows * cols) return;
    const uint row = idx % rows;
    output[idx] = input[idx] + bias[row];
}

__kernel void matmul_add_bias_kernel(
    __global const float* a,
    __global const float* b,
    __global const float* bias,
    __global float* output,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0) / N;
    const uint col = get_global_id(0) % N;
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        sum += a[row * K + i] * b[i * N + col];
    }
    output[row * N + col] = sum + bias[col];
}

__kernel void gelu_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    float x = input[idx];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
}

__kernel void mse_kernel(
    __global const float* input,
    __global const float* target,
    __global float* partial_sums,
    const uint size
) {
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint lsize = get_local_size(0);

    __local float local_buf[256];
    float diff = (gid < size) ? input[gid] - target[gid] : 0.0f;
    local_buf[lid] = diff * diff;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1)
    {
        if (lid < stride)
        {
            local_buf[lid] += local_buf[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        partial_sums[get_group_id(0)] = local_buf[0];
    }
}

__kernel void lif_step_kernel(
    __global float* v_mem,
    __global const float* input,
    __global float* output,
    __global float* adapt_a,
    const float beta,
    const float threshold,
    const float reset_potential,
    const int reset_zero,
    const float adapt_decay,
    const float adapt_coupling,
    const int use_adaptation,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;

    float v = v_mem[idx];
    v = v * beta + input[idx];

    float eff_thresh = threshold;
    float a = 0.0f;
    if (use_adaptation)
    {
        a = adapt_a[idx] * adapt_decay;
        eff_thresh += a;
    }

    float spike = v > eff_thresh ? 1.0f : 0.0f;
    output[idx] = spike;

    if (spike > 0.5f)
    {
        v = reset_zero ? reset_potential : v - threshold;
        if (use_adaptation) a += adapt_coupling;
    }

    v_mem[idx] = v;
    if (use_adaptation) adapt_a[idx] = a;
}

__kernel void lif_grad_kernel(
    __global const float* v_pre,
    __global float* output,
    const float threshold,
    const float sharpness,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    float diff = v_pre[idx] - threshold;
    float abs_diff = fabs(diff);
    output[idx] = (1.0f / sharpness) * exp(-abs_diff / sharpness);
}

__kernel void lif_grad_boxcar_kernel(
    __global const float* v_pre,
    __global float* output,
    const float threshold,
    const float half_window,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    float diff = fabs(v_pre[idx] - threshold);
    output[idx] = (diff < half_window) ? 1.0f : 0.0f;
}

__kernel void matmul_rhs_transposed_bias_sigmoid_kernel(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        sum += A[row + i * M] * B[col + i * N];
    }
    float val = sum + bias[col];
    C[row + col * M] = 1.0f / (1.0f + exp(-val));
}

__kernel void matmul_rhs_transposed_bias_tanh_kernel(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        sum += A[row + i * M] * B[col + i * N];
    }
    float val = sum + bias[col];
    C[row + col * M] = tanh(val);
}

__kernel void matmul_rhs_transposed_bias_relu_kernel(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        // A is (M x K) column-major: A(row,i) -> row + i*M
        // B is (N x K) column-major; access B(col,i) -> col + i*N (RHS transposed)
        sum += A[row + i * M] * B[col + i * N];
    }
    float val = sum + bias[col];
    C[row + col * M] = val > 0.0f ? val : 0.0f;
}

__kernel void matmul_rhs_transposed_bias_leaky_relu_kernel(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const uint M,
    const uint N,
    const uint K,
    const float alpha
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        sum += A[row + i * M] * B[col + i * N];
    }
    float val = sum + bias[col];
    C[row + col * M] = val > 0.0f ? val : alpha * val;
}

__kernel void adam_step_kernel(
    __global float* param,
    __global float* moment1,
    __global float* moment2,
    __global const float* grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float bias_correction1,
    const float bias_correction2,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;

    float g = grad[idx];
    float m = beta1 * moment1[idx] + (1.0f - beta1) * g;
    float v = beta2 * moment2[idx] + (1.0f - beta2) * g * g;

    moment1[idx] = m;
    moment2[idx] = v;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    param[idx] = param[idx] - lr * m_hat / (sqrt(v_hat) + epsilon);
}
)";
// clang-format on

} // namespace nn::opencl
