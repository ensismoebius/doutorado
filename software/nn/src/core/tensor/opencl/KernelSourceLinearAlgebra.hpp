// Embedded OpenCL C source for the linear-algebra kernel program (matmul family, transpose,
// rowwise_sum).
#pragma once

namespace nn::opencl
{

// clang-format off
inline constexpr const char* KERNEL_SOURCE_LINEAR_ALGEBRA = R"(
__kernel void matmul_kernel(
    __global const float* A,
    __global const float* B,
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
        // xt::xarray<float> is column-major by default:
        // idx(row,col) = row + col * rows
        sum += A[row + i * M] * B[i + col * K];
    }
    
    C[row + col * M] = sum;
}

__kernel void matmul_rhs_transposed_kernel(
    __global const float* A,
    __global const float* B,
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
        // B is stored as original (N x K) column-major tensor.
        // Access B(col, i) directly instead of materializing transpose.
        sum += A[row + i * M] * B[col + i * N];
    }

    C[row + col * M] = sum;
}

__kernel void matmul_lhs_transposed_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0); // output row in [0, K)
    const uint col = get_global_id(1); // output col in [0, N)
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    const uint TILE = 16;
    __local float a_tile[16][16];
    __local float b_tile[16][16];

    float sum = 0.0f;
    for (uint t = 0; t < M; t += TILE) {
        const uint a_i = t + ly;
        const uint b_i = t + lx;

        // A is (M x K) in column-major: A(i,row) -> i + row*M
        if (row < K && a_i < M) {
            a_tile[ly][lx] = A[a_i + row * M];
        } else {
            a_tile[ly][lx] = 0.0f;
        }

        // B is (M x N) in column-major: B(i,col) -> i + col*M
        if (col < N && b_i < M) {
            b_tile[lx][ly] = B[b_i + col * M];
        } else {
            b_tile[lx][ly] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint j = 0; j < TILE; ++j) {
            sum += a_tile[j][lx] * b_tile[j][ly];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < K && col < N) {
        C[row + col * K] = sum;
    }
}

__kernel void matmul_rhs_transposed_bias_kernel(
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

    C[row + col * M] = sum + bias[col];
}

// Generic 2-D strided copy over column-major storage.
//
// Every view/slice op in OpenCLTensorBackend (block, setBlock, row, col,
// topRows, leftCols, slice_time, set_time_slice, slice_batch, set_batch_slice)
// is a rectangular copy between two strided views of the same layout, so they
// all dispatch here instead of doing a host round-trip.
//
// Element (i, j) of the region maps to base + i*stride_i + j*stride_j on each
// side. Column-major 2-D is (stride_i, stride_j) = (1, rows); the third
// dimension of a rank-3 tensor just contributes a larger stride.
__kernel void strided_copy_2d_kernel(
    __global const float* src,
    __global float* dst,
    const uint src_base,
    const uint src_stride_i,
    const uint src_stride_j,
    const uint dst_base,
    const uint dst_stride_i,
    const uint dst_stride_j,
    const uint ni,
    const uint nj
) {
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    if (i >= ni || j >= nj) return;

    dst[dst_base + i * dst_stride_i + j * dst_stride_j] =
        src[src_base + i * src_stride_i + j * src_stride_j];
}

__kernel void transpose_kernel(
    __global const float* input,
    __global float* output,
    const uint rows,
    const uint cols
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);
    
    if (row >= rows || col >= cols) return;
    
    // Column-major transpose mapping:
    // input(row,col)  idx = row + col * rows
    // output(col,row) idx = col + row * cols
    output[col + row * cols] = input[row + col * rows];
}
)";
// clang-format on

} // namespace nn::opencl
