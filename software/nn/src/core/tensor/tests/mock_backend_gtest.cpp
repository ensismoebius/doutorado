#include <gtest/gtest.h>

#include <cmath>

#include "MockTensorBackend.hpp"
#include "core/tensor/Tensor.hpp"

namespace nn::tests
{

TEST(MockTensorBackend, TensorAddLogsAndComputes)
{
    std::vector<float> lhs_data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> rhs_data{1.0f, 1.0f, 1.0f, 1.0f};

    auto lhs_backend = std::make_unique<MockTensorBackend>(std::vector<Index>{2, 2}, lhs_data);
    auto rhs_backend = std::make_unique<MockTensorBackend>(std::vector<Index>{2, 2}, rhs_data);

    auto* lhs_ptr = lhs_backend.get();

    Tensor lhs(std::move(lhs_backend));
    Tensor rhs(std::move(rhs_backend));

    auto result = lhs.add(rhs);

    // Backend should have recorded the operation
    ASSERT_FALSE(lhs_ptr->log().empty());
    const auto& last = lhs_ptr->log().back();
    EXPECT_NE(last.find("add:[2x2]"), std::string::npos);

    // Result should contain element-wise sums; validate via norm to avoid peeking into backend
    const float expected_norm = std::sqrt(2.0f * 2.0f + 3.0f * 3.0f + 4.0f * 4.0f + 5.0f * 5.0f);
    EXPECT_NEAR(result.norm(), expected_norm, 1e-5f);
}

TEST(MockTensorBackend, TensorMatmulLogs)
{
    std::vector<float> a_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // 2x3
    std::vector<float> b_data{1.0f, 0.0f, 0.0f};                   // 3x1

    auto a_backend = std::make_unique<MockTensorBackend>(std::vector<Index>{2, 3}, a_data);
    auto b_backend = std::make_unique<MockTensorBackend>(std::vector<Index>{3, 1}, b_data);

    auto* a_ptr = a_backend.get();

    Tensor a(std::move(a_backend));
    Tensor b(std::move(b_backend));

    auto result = a.matmul(b);

    ASSERT_FALSE(a_ptr->log().empty());
    const auto& last = a_ptr->log().back();
    EXPECT_NE(last.find("matmul:[2x3]x[3x1]"), std::string::npos);

    // With b as [1,0,0]^T, the matmul should yield the first column of a
    // Expected output vector: [1, 4]^T => norm = sqrt(1^2 + 4^2) = sqrt(17)
    EXPECT_NEAR(result.norm(), std::sqrt(17.0f), 1e-5f);
}

} // namespace nn::tests
