/**
 * @file opencl_tensor_backend_inplace_gtest.cpp
 * @brief In-place binary/scalar/broadcast op correctness on the OpenCL backend.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "layers/spiking/Lif.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLProfiling.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace
{

TEST(OpenCLTensorBackendTest, GpuResidentInplaceFillAndClamp)
{
    nn::OpenCLTensorBackend a(1, 4);
    a.at(0, 0) = -2.0f;
    a.at(0, 1) = -1.0f;
    a.at(0, 2) = 2.0f;
    a.at(0, 3) = 5.0f;

    // relu() should create a GPU-accelerated output tensor.
    auto t = a.relu();

    t.clamp_inplace(0.5f, 1.5f);
    EXPECT_NEAR(t.at(0, 0), 0.5f, 1e-6f);
    EXPECT_NEAR(t.at(0, 1), 0.5f, 1e-6f);
    EXPECT_NEAR(t.at(0, 2), 1.5f, 1e-6f);
    EXPECT_NEAR(t.at(0, 3), 1.5f, 1e-6f);

    t.fill(2.0f);
    EXPECT_NEAR(t.at(0, 0), 2.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 2), 2.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 3), 2.0f, 1e-6f);
}

TEST(OpenCLTensorBackendTest, GpuResidentInplaceScalarAndMathOps)
{
    nn::OpenCLTensorBackend a(1, 4);
    a.at(0, 0) = -1.0f;
    a.at(0, 1) = 1.0f;
    a.at(0, 2) = 2.0f;
    a.at(0, 3) = 3.0f;

    // Keep this tensor on GPU path first.
    auto t = a.relu(); // [0,1,2,3]

    t.add_scalar_inplace(1.0f);      // [1,2,3,4]
    t.multiply_scalar_inplace(2.0f); // [2,4,6,8]
    t.divide_scalar_inplace(2.0f);   // [1,2,3,4]

    EXPECT_NEAR(t.at(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 2), 3.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 3), 4.0f, 1e-6f);

    t.square_inplace(); // [1,4,9,16]
    t.sqrt_inplace();   // [1,2,3,4]

    EXPECT_NEAR(t.at(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(t.at(0, 1), 2.0f, 1e-5f);
    EXPECT_NEAR(t.at(0, 2), 3.0f, 1e-5f);
    EXPECT_NEAR(t.at(0, 3), 4.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, GpuResidentInplaceBinaryOps)
{
    nn::OpenCLTensorBackend a(1, 4);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(0, 2) = 3.0f;
    a.at(0, 3) = 4.0f;

    nn::OpenCLTensorBackend b(1, 4);
    b.at(0, 0) = 2.0f;
    b.at(0, 1) = 3.0f;
    b.at(0, 2) = 4.0f;
    b.at(0, 3) = 5.0f;

    // Produce GPU-accelerated tensors first.
    auto lhs = a.relu();
    auto rhs = b.relu();

    lhs.add_inplace(rhs); // [3,5,7,9]
    EXPECT_NEAR(lhs.at(0, 0), 3.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 1), 5.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 2), 7.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 3), 9.0f, 1e-6f);

    lhs.subtract_inplace(rhs); // [1,2,3,4]
    EXPECT_NEAR(lhs.at(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 2), 3.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 3), 4.0f, 1e-6f);

    lhs.multiply_inplace(rhs); // [2,6,12,20]
    EXPECT_NEAR(lhs.at(0, 0), 2.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 1), 6.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 2), 12.0f, 1e-6f);
    EXPECT_NEAR(lhs.at(0, 3), 20.0f, 1e-6f);

    lhs.divide_inplace(rhs); // [1,2,3,4]
    EXPECT_NEAR(lhs.at(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(0, 1), 2.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(0, 2), 3.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(0, 3), 4.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, GpuResidentAddRowBroadcastInplace)
{
    nn::OpenCLTensorBackend a(3, 2);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f;
    a.at(1, 1) = 4.0f;
    a.at(2, 0) = 5.0f;
    a.at(2, 1) = 6.0f;

    nn::OpenCLTensorBackend row(1, 2);
    row.at(0, 0) = 10.0f;
    row.at(0, 1) = 20.0f;

    // Put both through OpenCL-producing ops before in-place broadcast.
    auto lhs = a.relu();
    auto bias = row.relu();

    lhs.add_row_broadcast_inplace(bias);

    EXPECT_NEAR(lhs.at(0, 0), 11.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(0, 1), 22.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(1, 0), 13.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(1, 1), 24.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(2, 0), 15.0f, 1e-5f);
    EXPECT_NEAR(lhs.at(2, 1), 26.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, InplaceArithmeticOps)
{
    nn::OpenCLTensorBackend a(2, 2);
    nn::OpenCLTensorBackend b(2, 2);
    a.at(0, 0) = 10.0f;
    a.at(1, 1) = 20.0f;
    b.at(0, 0) = 3.0f;
    b.at(1, 1) = 4.0f;

    a.add_inplace(b);
    EXPECT_NEAR(a.at(0, 0), 13.0f, 1e-5f);

    a.subtract_inplace(b);
    EXPECT_NEAR(a.at(0, 0), 10.0f, 1e-5f);

    a.multiply_inplace(b);
    EXPECT_NEAR(a.at(0, 0), 30.0f, 1e-5f);

    a.divide_inplace(b);
    EXPECT_NEAR(a.at(0, 0), 10.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, InplaceScalarOps)
{
    nn::OpenCLTensorBackend a(2, 2);
    a.at(0, 0) = 4.0f;
    a.at(1, 1) = 9.0f;

    a.add_scalar_inplace(1.0f);
    EXPECT_NEAR(a.at(0, 0), 5.0f, 1e-5f);

    a.multiply_scalar_inplace(2.0f);
    EXPECT_NEAR(a.at(0, 0), 10.0f, 1e-5f);

    a.divide_scalar_inplace(2.0f);
    EXPECT_NEAR(a.at(0, 0), 5.0f, 1e-5f);

    nn::OpenCLTensorBackend b(2, 2);
    b.at(0, 0) = 4.0f;
    b.at(0, 1) = 9.0f;
    b.sqrt_inplace();
    EXPECT_NEAR(b.at(0, 0), 2.0f, 1e-5f);

    nn::OpenCLTensorBackend c(2, 2);
    c.at(0, 0) = 3.0f;
    c.square_inplace();
    EXPECT_NEAR(c.at(0, 0), 9.0f, 1e-5f);
}

TEST(OpenCLTensorBackendTest, AddColVectorToRowsInplace)
{
    nn::OpenCLTensorBackend a(3, 2);
    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f;
    a.at(1, 1) = 4.0f;
    a.at(2, 0) = 5.0f;
    a.at(2, 1) = 6.0f;

    // API expects shape (cols, 1): col[c] is broadcast to every row at column c
    nn::OpenCLTensorBackend col(2, 1);
    col.at(0, 0) = 10.0f;
    col.at(1, 0) = 20.0f;

    a.add_col_vector_to_rows_inplace(col);
    // Each column gets its bias added to every row at that column
    EXPECT_NEAR(a.at(0, 0), 11.0f, 1e-5f); // 1 + col[0]=10
    EXPECT_NEAR(a.at(0, 1), 22.0f, 1e-5f); // 2 + col[1]=20
    EXPECT_NEAR(a.at(1, 0), 13.0f, 1e-5f); // 3 + col[0]=10
    EXPECT_NEAR(a.at(1, 1), 24.0f, 1e-5f); // 4 + col[1]=20
    EXPECT_NEAR(a.at(2, 0), 15.0f, 1e-5f); // 5 + col[0]=10
    EXPECT_NEAR(a.at(2, 1), 26.0f, 1e-5f); // 6 + col[1]=20
}

} // namespace
