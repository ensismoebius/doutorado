// Embedded OpenCL C source for the reductions kernel program (sum, mean-squared-error).
#pragma once

namespace nn::opencl
{

// clang-format off
inline constexpr const char* KERNEL_SOURCE_REDUCTIONS = R"(
__kernel void rowwise_sum_kernel(
    __global const float* input,
    __global float* output,
    const uint rows,
    const uint cols
) {
    const uint row = get_global_id(0);

    if (row >= rows) return;

    float sum = 0.0f;
    for (uint col = 0; col < cols; ++col) {
        sum += input[row + col * rows];
    }

    output[row] = sum;
}

__kernel void sum_kernel(
    __global const float* input,
    __global float* partial_sums,
    const uint size
) {
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint lsize = get_local_size(0);

    __local float local_buf[256];
    local_buf[lid] = (gid < size) ? input[gid] : 0.0f;
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
)";
// clang-format on

} // namespace nn::opencl
