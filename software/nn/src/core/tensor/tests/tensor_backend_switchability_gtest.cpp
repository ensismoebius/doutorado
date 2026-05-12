#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "tensor/Tensor.hpp"

namespace
{
class MockSwitchBackend
{
   public:
    MockSwitchBackend() = default;

    MockSwitchBackend(const MockSwitchBackend& other)
        : shape_(other.shape_), data_(other.data_)
    {
        if (other.grad_)
        {
            grad_ = std::make_unique<MockSwitchBackend>(*other.grad_);
        }
    }

    auto operator=(const MockSwitchBackend& other) -> MockSwitchBackend&
    {
        if (this == &other)
        {
            return *this;
        }
        shape_ = other.shape_;
        data_ = other.data_;
        grad_ = other.grad_ ? std::make_unique<MockSwitchBackend>(*other.grad_) : nullptr;
        return *this;
    }

    MockSwitchBackend(MockSwitchBackend&&) noexcept = default;
    auto operator=(MockSwitchBackend&&) noexcept -> MockSwitchBackend& = default;

    explicit MockSwitchBackend(nn::Index rows, nn::Index cols)
        : shape_{rows, cols}, data_(rows * cols, 0.0F)
    {
    }

    explicit MockSwitchBackend(nn::Index d1, nn::Index d2, nn::Index d3)
        : shape_{d1, d2, d3}, data_(d1 * d2 * d3, 0.0F)
    {
    }

    explicit MockSwitchBackend(nn::Index d1, nn::Index d2, nn::Index d3, nn::Index d4)
        : shape_{d1, d2, d3, d4}, data_(d1 * d2 * d3 * d4, 0.0F)
    {
    }

    explicit MockSwitchBackend(const std::vector<nn::Index>& shape)
        : shape_(shape), data_(product(shape), 0.0F)
    {
    }

    static auto zeros(nn::Index rows, nn::Index cols) -> MockSwitchBackend
    {
        return MockSwitchBackend(rows, cols);
    }

    static auto ones(nn::Index rows, nn::Index cols) -> MockSwitchBackend
    {
        MockSwitchBackend out(rows, cols);
        out.set_ones();
        return out;
    }

    static auto random(nn::Index rows, nn::Index cols) -> MockSwitchBackend
    {
        std::mt19937 rng(42U);
        return random(rows, cols, rng);
    }

    static auto random(nn::Index rows, nn::Index cols, std::mt19937& rng) -> MockSwitchBackend
    {
        MockSwitchBackend out(rows, cols);
        std::uniform_real_distribution<float> dist(0.0F, 1.0F);
        for (auto& value : out.data_) value = dist(rng);
        return out;
    }

    static auto random(nn::Index d1, nn::Index d2, nn::Index d3) -> MockSwitchBackend
    {
        std::mt19937 rng(42U);
        return random(d1, d2, d3, rng);
    }

    static auto random(
        nn::Index d1, nn::Index d2, nn::Index d3, std::mt19937& rng) -> MockSwitchBackend
    {
        MockSwitchBackend out(d1, d2, d3);
        std::uniform_real_distribution<float> dist(0.0F, 1.0F);
        for (auto& value : out.data_) value = dist(rng);
        return out;
    }

    auto shape() const -> const std::vector<nn::Index>&
    {
        return shape_;
    }

    void reshape(const std::vector<nn::Index>& new_shape)
    {
        if (product(new_shape) != data_.size())
        {
            throw std::invalid_argument("MockSwitchBackend::reshape size mismatch");
        }
        shape_ = new_shape;
    }

    auto rows() const -> nn::Index
    {
        return shape_.empty() ? 0U : shape_[0];
    }

    auto cols() const -> nn::Index
    {
        if (shape_.size() <= 1U) return 1U;
        nn::Index c = 1U;
        for (std::size_t i = 1; i < shape_.size(); ++i) c *= shape_[i];
        return c;
    }

    auto size() const -> nn::Index
    {
        return static_cast<nn::Index>(data_.size());
    }

    auto at(nn::Index i) -> float&
    {
        return data_.at(i);
    }

    auto at(nn::Index i) const -> const float&
    {
        return data_.at(i);
    }

    auto at(nn::Index row, nn::Index col) -> float&
    {
        return data_.at(row * cols() + col);
    }

    auto at(nn::Index row, nn::Index col) const -> const float&
    {
        return data_.at(row * cols() + col);
    }

    auto at(nn::Index d1, nn::Index d2, nn::Index d3) -> float&
    {
        return data_.at((d1 * shape_.at(1) + d2) * shape_.at(2) + d3);
    }

    auto at(nn::Index d1, nn::Index d2, nn::Index d3) const -> const float&
    {
        return data_.at((d1 * shape_.at(1) + d2) * shape_.at(2) + d3);
    }

    auto at(nn::Index d1, nn::Index d2, nn::Index d3, nn::Index d4) -> float&
    {
        const auto d2_size = shape_.at(1);
        const auto d3_size = shape_.at(2);
        const auto d4_size = shape_.at(3);
        return data_.at(((d1 * d2_size + d2) * d3_size + d3) * d4_size + d4);
    }

    auto at(nn::Index d1, nn::Index d2, nn::Index d3, nn::Index d4) const -> const float&
    {
        const auto d2_size = shape_.at(1);
        const auto d3_size = shape_.at(2);
        const auto d4_size = shape_.at(3);
        return data_.at(((d1 * d2_size + d2) * d3_size + d3) * d4_size + d4);
    }

    auto at(const std::vector<nn::Index>& indices) -> float&
    {
        return data_.at(flat_index(indices));
    }

    auto at(const std::vector<nn::Index>& indices) const -> const float&
    {
        return data_.at(flat_index(indices));
    }

    auto row(nn::Index i) const -> MockSwitchBackend
    {
        MockSwitchBackend out(1, cols());
        for (nn::Index c = 0; c < cols(); ++c) out.at(0, c) = at(i, c);
        return out;
    }

    auto col(nn::Index j) const -> MockSwitchBackend
    {
        MockSwitchBackend out(rows(), 1);
        for (nn::Index r = 0; r < rows(); ++r) out.at(r, 0) = at(r, j);
        return out;
    }

    auto leftCols(nn::Index n) const -> MockSwitchBackend
    {
        MockSwitchBackend out(rows(), n);
        for (nn::Index r = 0; r < rows(); ++r)
            for (nn::Index c = 0; c < n; ++c) out.at(r, c) = at(r, c);
        return out;
    }

    auto topRows(nn::Index n) const -> MockSwitchBackend
    {
        MockSwitchBackend out(n, cols());
        for (nn::Index r = 0; r < n; ++r)
            for (nn::Index c = 0; c < cols(); ++c) out.at(r, c) = at(r, c);
        return out;
    }

    auto block(nn::Index row, nn::Index col, nn::Index rows, nn::Index cols) const
        -> MockSwitchBackend
    {
        MockSwitchBackend out(rows, cols);
        for (nn::Index r = 0; r < rows; ++r)
            for (nn::Index c = 0; c < cols; ++c) out.at(r, c) = at(row + r, col + c);
        return out;
    }

    void setBlock(nn::Index row, nn::Index col, const MockSwitchBackend& block)
    {
        for (nn::Index r = 0; r < block.rows(); ++r)
            for (nn::Index c = 0; c < block.cols(); ++c) at(row + r, col + c) = block.at(r, c);
    }

    auto slice(std::span<const int> indices) const -> MockSwitchBackend
    {
        MockSwitchBackend out(static_cast<nn::Index>(indices.size()), cols());
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            for (nn::Index c = 0; c < cols(); ++c)
            {
                out.at(static_cast<nn::Index>(i), c) = at(static_cast<nn::Index>(indices[i]), c);
            }
        }
        return out;
    }

    auto slice_batch(nn::Index b) const -> MockSwitchBackend
    {
        if (shape_.size() != 3U)
        {
            throw std::invalid_argument("MockSwitchBackend::slice_batch expects 3D");
        }
        MockSwitchBackend out(shape_[1], shape_[2]);
        for (nn::Index t = 0; t < shape_[1]; ++t)
            for (nn::Index f = 0; f < shape_[2]; ++f) out.at(t, f) = at(b, t, f);
        return out;
    }

    void set_batch_slice(nn::Index b, const MockSwitchBackend& value)
    {
        for (nn::Index t = 0; t < value.rows(); ++t)
            for (nn::Index f = 0; f < value.cols(); ++f) at(b, t, f) = value.at(t, f);
    }

    auto slice_time(nn::Index t) const -> MockSwitchBackend
    {
        if (shape_.size() != 3U)
        {
            throw std::invalid_argument("MockSwitchBackend::slice_time expects 3D");
        }
        MockSwitchBackend out(shape_[0], shape_[2]);
        for (nn::Index b = 0; b < shape_[0]; ++b)
            for (nn::Index f = 0; f < shape_[2]; ++f) out.at(b, f) = at(b, t, f);
        return out;
    }

    void set_time_slice(nn::Index t, const MockSwitchBackend& value)
    {
        for (nn::Index b = 0; b < value.rows(); ++b)
            for (nn::Index f = 0; f < value.cols(); ++f) at(b, t, f) = value.at(b, f);
    }

    auto add_row_broadcast(const MockSwitchBackend& row) const -> MockSwitchBackend
    {
        MockSwitchBackend out(*this);
        out.add_row_broadcast_inplace(row);
        return out;
    }

    void add_row_broadcast_inplace(const MockSwitchBackend& row)
    {
        for (nn::Index r = 0; r < rows(); ++r)
            for (nn::Index c = 0; c < cols(); ++c) at(r, c) += row.at(0, c);
    }

    auto sum_rows() const -> MockSwitchBackend
    {
        MockSwitchBackend out(rows(), 1);
        for (nn::Index r = 0; r < rows(); ++r)
        {
            float sum = 0.0F;
            for (nn::Index c = 0; c < cols(); ++c) sum += at(r, c);
            out.at(r, 0) = sum;
        }
        return out;
    }

    void fill(float value)
    {
        std::fill(data_.begin(), data_.end(), value);
    }

    void set_zero()
    {
        fill(0.0F);
    }

    void set_ones()
    {
        fill(1.0F);
    }

    auto data_ptr() const -> const float*
    {
        return data_.data();
    }

    auto mutable_data_ptr() -> float*
    {
        return data_.data();
    }

    void set_grad(const MockSwitchBackend& grad)
    {
        grad_ = std::make_unique<MockSwitchBackend>(grad);
    }

   private:
    static auto product(const std::vector<nn::Index>& shape) -> std::size_t
    {
        std::size_t total = 1U;
        for (const auto dim : shape) total *= static_cast<std::size_t>(dim);
        return total;
    }

    auto flat_index(const std::vector<nn::Index>& indices) const -> nn::Index
    {
        if (indices.size() != shape_.size())
        {
            throw std::invalid_argument("MockSwitchBackend::flat_index rank mismatch");
        }
        nn::Index idx = 0U;
        nn::Index stride = 1U;
        for (std::size_t i = shape_.size(); i-- > 0;)
        {
            idx += indices[i] * stride;
            stride *= shape_[i];
        }
        return idx;
    }

    std::vector<nn::Index> shape_{};
    std::vector<float> data_{};
    std::unique_ptr<MockSwitchBackend> grad_{};
};

static_assert(nn::TensorBackendParityContract<MockSwitchBackend>);

TEST(TensorBackendSwitchability, WorksWithArbitraryCustomBackend)
{
    nn::TensorImpl<MockSwitchBackend> tensor(2, 3);
    tensor.set_zero();
    tensor.at(0, 0) = 1.0F;
    tensor.at(1, 2) = 5.0F;

    EXPECT_EQ(tensor.rows(), 2U);
    EXPECT_EQ(tensor.cols(), 3U);
    EXPECT_FLOAT_EQ(tensor.at(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(tensor.at(1, 2), 5.0F);

    const auto row_sum = tensor.sum_rows();
    EXPECT_EQ(row_sum.rows(), 2U);
    EXPECT_EQ(row_sum.cols(), 1U);
    EXPECT_FLOAT_EQ(row_sum.at(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(row_sum.at(1, 0), 5.0F);
}

TEST(TensorBackendSwitchability, TensorHeaderHasNoConcreteBackendLeak)
{
#ifdef NN_PROJECT_ROOT
    const std::filesystem::path tensor_header =
        std::filesystem::path(NN_PROJECT_ROOT) / "include/tensor/Tensor.hpp";
#else
    GTEST_SKIP() << "NN_PROJECT_ROOT is not defined for this test target";
#endif

    std::ifstream input(tensor_header);
    ASSERT_TRUE(input.good()) << "Unable to open " << tensor_header;

    std::stringstream buffer;
    buffer << input.rdbuf();
    const std::string contents = buffer.str();

    EXPECT_EQ(contents.find("class OpenCLTensorBackend;"), std::string::npos);
    EXPECT_EQ(contents.find("template <typename Backend = XTensorBackend>"), std::string::npos);
    EXPECT_EQ(contents.find("#include \"tensor/xtensor/XTensorBackend.hpp\""),
        std::string::npos);
}

} // namespace
