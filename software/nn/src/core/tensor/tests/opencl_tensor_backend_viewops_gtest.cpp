/**
 * @file opencl_tensor_backend_viewops_gtest.cpp
 * @brief Device-side view/slice op correctness on the OpenCL backend
 * (row/col/block/topRows/leftCols/time-slice/batch-slice).
 */

#include <gtest/gtest.h>

#include <cmath>

#include "layers/spiking/Lif.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLProfiling.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace
{

// ---------------------------------------------------------------------------
// View / slice ops.
//
// These run on the device via strided_copy_2d_kernel over column-major storage.
// The index math differs per op and a mistake corrupts data silently rather
// than failing loudly, so every op is checked against explicit expected values
// using non-square shapes and non-zero offsets (a square shape or a zero offset
// would let a row/column mix-up pass).
// ---------------------------------------------------------------------------

// (rows, cols) filled so each element encodes its own position: r*100 + c.
nn::OpenCLTensorBackend make_2d(nn::Index rows, nn::Index cols)
{
    nn::OpenCLTensorBackend t(rows, cols);
    for (nn::Index r = 0; r < rows; ++r)
        for (nn::Index c = 0; c < cols; ++c) t.at(r, c) = static_cast<float>(r * 100 + c);
    return t;
}

// (d0, d1, d2) filled as d0*10000 + d1*100 + d2.
nn::OpenCLTensorBackend make_3d(nn::Index d0, nn::Index d1, nn::Index d2)
{
    nn::OpenCLTensorBackend t(d0, d1, d2);
    for (nn::Index a = 0; a < d0; ++a)
        for (nn::Index b = 0; b < d1; ++b)
            for (nn::Index c = 0; c < d2; ++c)
                t.at(a, b, c) = static_cast<float>(a * 10000 + b * 100 + c);
    return t;
}

TEST(OpenCLViewOpsTest, RowExtractsCorrectRow)
{
    auto t = make_2d(4, 5);
    auto r = t.row(2);
    ASSERT_EQ(r.rows(), 1u);
    ASSERT_EQ(r.cols(), 5u);
    for (nn::Index c = 0; c < 5; ++c) EXPECT_FLOAT_EQ(r.at(0, c), 200.0f + c) << "c=" << c;
}

TEST(OpenCLViewOpsTest, ColExtractsCorrectColumn)
{
    auto t = make_2d(4, 5);
    auto c = t.col(3);
    ASSERT_EQ(c.rows(), 4u);
    ASSERT_EQ(c.cols(), 1u);
    for (nn::Index r = 0; r < 4; ++r) EXPECT_FLOAT_EQ(c.at(r, 0), r * 100.0f + 3.0f) << "r=" << r;
}

TEST(OpenCLViewOpsTest, BlockExtractsOffsetRegion)
{
    auto t = make_2d(5, 6);
    auto b = t.block(1, 2, 3, 4); // rows 1..3, cols 2..5
    ASSERT_EQ(b.rows(), 3u);
    ASSERT_EQ(b.cols(), 4u);
    for (nn::Index r = 0; r < 3; ++r)
        for (nn::Index c = 0; c < 4; ++c)
            EXPECT_FLOAT_EQ(b.at(r, c), (r + 1) * 100.0f + (c + 2)) << "r=" << r << " c=" << c;
}

TEST(OpenCLViewOpsTest, SetBlockWritesRegionAndPreservesRest)
{
    auto t = make_2d(5, 6);
    auto patch = make_2d(2, 3);
    t.setBlock(2, 1, patch);

    for (nn::Index r = 0; r < 5; ++r)
    {
        for (nn::Index c = 0; c < 6; ++c)
        {
            const bool inside = (r >= 2 && r < 4 && c >= 1 && c < 4);
            const float want = inside ? static_cast<float>((r - 2) * 100 + (c - 1))
                                      : static_cast<float>(r * 100 + c);
            EXPECT_FLOAT_EQ(t.at(r, c), want) << "r=" << r << " c=" << c;
        }
    }
}

TEST(OpenCLViewOpsTest, TopRowsAndLeftCols)
{
    auto t = make_2d(5, 6);

    auto top = t.topRows(2);
    ASSERT_EQ(top.rows(), 2u);
    ASSERT_EQ(top.cols(), 6u);
    for (nn::Index r = 0; r < 2; ++r)
        for (nn::Index c = 0; c < 6; ++c) EXPECT_FLOAT_EQ(top.at(r, c), r * 100.0f + c);

    auto left = t.leftCols(3);
    ASSERT_EQ(left.rows(), 5u);
    ASSERT_EQ(left.cols(), 3u);
    for (nn::Index r = 0; r < 5; ++r)
        for (nn::Index c = 0; c < 3; ++c) EXPECT_FLOAT_EQ(left.at(r, c), r * 100.0f + c);
}

TEST(OpenCLViewOpsTest, SliceTimeExtractsTimestep)
{
    auto t = make_3d(3, 4, 5); // (B, T, D)
    auto s = t.slice_time(2);
    ASSERT_EQ(s.rows(), 3u); // B
    ASSERT_EQ(s.cols(), 5u); // D
    for (nn::Index b = 0; b < 3; ++b)
        for (nn::Index d = 0; d < 5; ++d)
            EXPECT_FLOAT_EQ(s.at(b, d), b * 10000.0f + 200.0f + d) << "b=" << b << " d=" << d;
}

TEST(OpenCLViewOpsTest, SetTimeSliceWritesTimestepAndPreservesRest)
{
    auto t = make_3d(3, 4, 5);
    nn::OpenCLTensorBackend v(3, 5);
    for (nn::Index b = 0; b < 3; ++b)
        for (nn::Index d = 0; d < 5; ++d) v.at(b, d) = -static_cast<float>(b * 10 + d + 1);

    t.set_time_slice(1, v);

    for (nn::Index b = 0; b < 3; ++b)
        for (nn::Index tt = 0; tt < 4; ++tt)
            for (nn::Index d = 0; d < 5; ++d)
            {
                const float want = (tt == 1) ? -static_cast<float>(b * 10 + d + 1)
                                             : static_cast<float>(b * 10000 + tt * 100 + d);
                EXPECT_FLOAT_EQ(t.at(b, tt, d), want) << "b=" << b << " t=" << tt << " d=" << d;
            }
}

TEST(OpenCLViewOpsTest, SliceBatchExtractsBatchElement)
{
    auto t = make_3d(3, 4, 5);
    auto s = t.slice_batch(1);
    ASSERT_EQ(s.rows(), 4u); // T
    ASSERT_EQ(s.cols(), 5u); // D
    for (nn::Index tt = 0; tt < 4; ++tt)
        for (nn::Index d = 0; d < 5; ++d)
            EXPECT_FLOAT_EQ(s.at(tt, d), 10000.0f + tt * 100.0f + d) << "t=" << tt << " d=" << d;
}

TEST(OpenCLViewOpsTest, SetBatchSliceWritesBatchAndPreservesRest)
{
    auto t = make_3d(3, 4, 5);
    nn::OpenCLTensorBackend v(4, 5);
    for (nn::Index tt = 0; tt < 4; ++tt)
        for (nn::Index d = 0; d < 5; ++d) v.at(tt, d) = -static_cast<float>(tt * 10 + d + 1);

    t.set_batch_slice(2, v);

    for (nn::Index b = 0; b < 3; ++b)
        for (nn::Index tt = 0; tt < 4; ++tt)
            for (nn::Index d = 0; d < 5; ++d)
            {
                const float want = (b == 2) ? -static_cast<float>(tt * 10 + d + 1)
                                            : static_cast<float>(b * 10000 + tt * 100 + d);
                EXPECT_FLOAT_EQ(t.at(b, tt, d), want) << "b=" << b << " t=" << tt << " d=" << d;
            }
}

// Round-trip through the ops the LSTM inner loop actually uses: read a
// timestep, transform it, write it back, for every timestep.
TEST(OpenCLViewOpsTest, TimeSliceRoundTripAcrossAllTimesteps)
{
    const nn::Index B = 2, T = 6, D = 3;
    auto t = make_3d(B, T, D);

    for (nn::Index step = 0; step < T; ++step)
    {
        auto slice = t.slice_time(step);
        slice.multiply_scalar_inplace(2.0f);
        t.set_time_slice(step, slice);
    }

    for (nn::Index b = 0; b < B; ++b)
        for (nn::Index tt = 0; tt < T; ++tt)
            for (nn::Index d = 0; d < D; ++d)
                EXPECT_FLOAT_EQ(t.at(b, tt, d), 2.0f * static_cast<float>(b * 10000 + tt * 100 + d))
                    << "b=" << b << " t=" << tt << " d=" << d;
}

} // namespace
