/**
 * @file gpu_buffer_pool_gtest.cpp
 * @brief Unit tests for GPUBufferPool.
 */

#include <CL/cl.h>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "tensor/opencl/GPUBufferPool.hpp"

using namespace nn::tensor;

class GPUBufferPoolTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        // Get default platform
        cl_uint num_platforms = 0;
        ASSERT_EQ(clGetPlatformIDs(0, nullptr, &num_platforms), CL_SUCCESS);
        ASSERT_GT(num_platforms, 0);

        std::vector<cl_platform_id> platforms(num_platforms);
        ASSERT_EQ(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), CL_SUCCESS);

        // Get first device
        cl_uint num_devices = 0;
        ASSERT_EQ(
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices), CL_SUCCESS);
        ASSERT_GT(num_devices, 0);

        std::vector<cl_device_id> devices(num_devices);
        ASSERT_EQ(
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr),
            CL_SUCCESS);

        device_ = devices[0];

        // Create context
        cl_int err = CL_SUCCESS;
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        ASSERT_EQ(err, CL_SUCCESS);
        ASSERT_NE(context_, nullptr);

        // Create command queue.
        // Prefer OpenCL 2.0+ API and keep a compile-time fallback for older headers.
#if defined(CL_VERSION_2_0)
        const cl_queue_properties queue_props[] = {
            CL_QUEUE_PROPERTIES,
            static_cast<cl_queue_properties>(0),
            0,
        };
        queue_ = clCreateCommandQueueWithProperties(context_, device_, queue_props, &err);
        ASSERT_EQ(err, CL_SUCCESS);
        ASSERT_NE(queue_, nullptr);
#else
        queue_ = clCreateCommandQueue(context_, device_, 0, &err);
        ASSERT_EQ(err, CL_SUCCESS);
        ASSERT_NE(queue_, nullptr);
#endif

        pool_ = std::make_unique<GPUBufferPool>(context_, queue_);
    }

    void TearDown() override
    {
        pool_.reset();
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    std::unique_ptr<GPUBufferPool> pool_;
};

TEST_F(GPUBufferPoolTest, AcquireAndRelease)
{
    auto handle = pool_->acquire(1024);
    ASSERT_TRUE(handle);
    EXPECT_EQ(handle->size_bytes, 1024);
}

TEST_F(GPUBufferPoolTest, ReusesBuffers)
{
    auto [total1, num1, avail1] = pool_->get_stats();
    EXPECT_EQ(num1, 0);

    cl_mem mem1 = nullptr;
    {
        auto handle = pool_->acquire(256);
        mem1 = handle->buffer;
        ASSERT_NE(mem1, nullptr);
    }

    auto [total2, num2, avail2] = pool_->get_stats();
    EXPECT_EQ(num2, 1); // Buffer was returned to pool

    // Acquire again - should get same buffer
    auto handle2 = pool_->acquire(256);
    EXPECT_EQ(handle2->buffer, mem1);
}

TEST_F(GPUBufferPoolTest, PoolSizeRounding)
{
    // Request 100 bytes - optimized bucket for NN
    auto handle = pool_->acquire(100);
    EXPECT_EQ(handle->size_bytes, 1024);
}

TEST_F(GPUBufferPoolTest, MultipleSizes)
{
    auto h1 = pool_->acquire(64);
    auto h2 = pool_->acquire(512);
    auto h3 = pool_->acquire(2048);

    EXPECT_EQ(h1->size_bytes, 1024);
    EXPECT_EQ(h2->size_bytes, 1024);
    EXPECT_EQ(h3->size_bytes, 4096);
}

TEST_F(GPUBufferPoolTest, PoolClear)
{
    auto h1 = pool_->acquire(1024);
    {
        auto h2 = pool_->acquire(512);
    } // h2 returned to pool

    auto [total, num, avail] = pool_->get_stats();
    EXPECT_GT(num, 0);

    pool_->clear();
    auto [total2, num2, avail2] = pool_->get_stats();
    EXPECT_EQ(num2, 0);
}

TEST_F(GPUBufferPoolTest, RoundsLargeRequestsTo64KBBuckets)
{
    auto handle = pool_->acquire(5000);
    ASSERT_TRUE(handle);
    EXPECT_EQ(handle->size_bytes, 16384);
}

TEST_F(GPUBufferPoolTest, OversizedAllocationReturnsInvalidHandle)
{
    auto handle = pool_->acquire(std::numeric_limits<size_t>::max() / 2);
    EXPECT_FALSE(handle);
}
