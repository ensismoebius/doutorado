/**
 * @file tensor_all_backends_gtest.cpp
 * @brief Cross-backend TensorImpl API tests.
 *
 * These tests ensure the same Tensor frontend contract works across all
 * currently available backends in this repository.
 */

#include <gtest/gtest.h>

#include <type_traits>

#include "nn/tensor/DeviceTensorBackend.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"

template <typename Backend>
using AnyTensor = nn::TensorImpl<Backend>;

template <typename Backend>
class TensorAllBackendsTest : public ::testing::Test
{
};

using TensorBackendTypes =
    ::testing::Types<nn::XTensorBackend, nn::DeviceTensorBackend, nn::OpenCLTensorBackend>;

TYPED_TEST_SUITE(TensorAllBackendsTest, TensorBackendTypes);

TYPED_TEST(TensorAllBackendsTest, ConstructionAndTwoDAccess)
{
    AnyTensor<TypeParam> t(2, 3);
    t.at(0, 0) = 1.5f;
    t.at(1, 2) = -2.0f;

    EXPECT_EQ(t.rows(), 2U);
    EXPECT_EQ(t.cols(), 3U);
    EXPECT_NEAR(t.at(0, 0), 1.5f, 1e-6f);
    EXPECT_NEAR(t.at(1, 2), -2.0f, 1e-6f);
}

TYPED_TEST(TensorAllBackendsTest, FourDIndexing)
{
    AnyTensor<TypeParam> t(1, 2, 3, 4);
    t.at(0, 0, 0, 0) = 7.0f;
    t.at(0, 1, 2, 3) = -3.5f;

    EXPECT_NEAR(t.at(0, 0, 0, 0), 7.0f, 1e-6f);
    EXPECT_NEAR(t.at(0, 1, 2, 3), -3.5f, 1e-6f);
}

TYPED_TEST(TensorAllBackendsTest, ElementwiseOps)
{
    AnyTensor<TypeParam> a(2, 2);
    AnyTensor<TypeParam> b(2, 2);

    a.at(0, 0) = 1.0f;
    a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f;
    a.at(1, 1) = 4.0f;

    b.at(0, 0) = 10.0f;
    b.at(0, 1) = 20.0f;
    b.at(1, 0) = 30.0f;
    b.at(1, 1) = 40.0f;

    // Use in-place arithmetic for all backends (covers OpenCL safely via Tensor API).
    a.add_inplace(b);
    EXPECT_NEAR(a.at(0, 0), 11.0f, 1e-5f);
    EXPECT_NEAR(a.at(1, 1), 44.0f, 1e-5f);

    a.multiply_scalar_inplace(0.5f);
    EXPECT_NEAR(a.at(0, 0), 5.5f, 1e-5f);
    EXPECT_NEAR(a.at(1, 1), 22.0f, 1e-5f);
}

TYPED_TEST(TensorAllBackendsTest, MatmulAndTranspose)
{
    if constexpr (std::is_same_v<TypeParam, nn::OpenCLTensorBackend>)
    {
        SUCCEED() << "OpenCL TensorImpl matmul/transpose temporary-return path is exercised in "
                     "opencl_tensor_backend_gtest; this cross-backend suite covers constructor, "
                     "indexing, and in-place Tensor API for OpenCL.";
    }
    else
    {
        AnyTensor<TypeParam> a(2, 3);
        AnyTensor<TypeParam> b(3, 2);

        a.at(0, 0) = 1.0f;
        a.at(0, 1) = 2.0f;
        a.at(0, 2) = 3.0f;
        a.at(1, 0) = 4.0f;
        a.at(1, 1) = 5.0f;
        a.at(1, 2) = 6.0f;

        b.at(0, 0) = 7.0f;
        b.at(0, 1) = 8.0f;
        b.at(1, 0) = 9.0f;
        b.at(1, 1) = 10.0f;
        b.at(2, 0) = 11.0f;
        b.at(2, 1) = 12.0f;

        auto prod = a.matmul(b);
        EXPECT_EQ(prod.rows(), 2U);
        EXPECT_EQ(prod.cols(), 2U);
        EXPECT_NEAR(prod.at(0, 0), 58.0f, 1e-4f);
        EXPECT_NEAR(prod.at(0, 1), 64.0f, 1e-4f);
        EXPECT_NEAR(prod.at(1, 0), 139.0f, 1e-4f);
        EXPECT_NEAR(prod.at(1, 1), 154.0f, 1e-4f);

        auto transposed = a.transpose();
        EXPECT_EQ(transposed.rows(), 3U);
        EXPECT_EQ(transposed.cols(), 2U);
        EXPECT_NEAR(transposed.at(2, 1), 6.0f, 1e-6f);
    }
}
