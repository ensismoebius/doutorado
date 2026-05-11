/**
 * @file device_backend_gtest.cpp
 * @brief Unit tests for DeviceTensorBackend skeleton to validate the backend API
 * contract and behaviour using a host-mirroring implementation.
 */

#include <gtest/gtest.h>

#include "nn/device/Device.hpp"
#include "nn/tensor/DeviceTensorBackend.hpp"
#include "nn/tensor/Tensor.hpp"

using DeviceTensor = nn::TensorImpl<nn::DeviceTensorBackend>;

TEST(DeviceBackendTest, ConstructionAndAssignment)
{
    DeviceTensor t1(2, 2);
    t1.at(0, 0) = 1.0f;
    t1.at(0, 1) = 2.0f;
    t1.at(1, 0) = 3.0f;
    t1.at(1, 1) = 4.0f;

    DeviceTensor t2 = t1;
    ASSERT_EQ(t1.rows(), 2);
    ASSERT_EQ(t1.cols(), 2);
    EXPECT_NEAR(t1.sum(), 10.0f, 1e-6f);
    EXPECT_EQ(t1, t2);

    t2.at(0, 0) += 1.0f;
    ASSERT_NE(t1, t2);
}

TEST(DeviceBackendTest, TwoDAccess)
{
    DeviceTensor t(2, 3);
    t.at(0, 0) = 1.0f;
    t.at(1, 2) = 2.0f;
    EXPECT_NEAR(t.at(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(t.at(1, 2), 2.0f, 1e-6f);
    EXPECT_THROW(t.at(2, 0), std::out_of_range);
    EXPECT_THROW(t.at(0, 3), std::out_of_range);
}

TEST(DeviceBackendTest, FourDAccess)
{
    DeviceTensor t(2, 3, 4, 5);
    t.at(0, 0, 0, 0) = 1.0f;
    t.at(1, 2, 3, 4) = 2.0f;
    EXPECT_NEAR(t.at(0, 0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(t.at(1, 2, 3, 4), 2.0f, 1e-6f);
}

TEST(DeviceBackendTest, ZeroGrad)
{
    DeviceTensor t(2, 3);
    DeviceTensor g(2, 3);
    g.setZero();
    g.at(0, 0) = 5.0f;
    g.at(0, 1) = 6.0f;
    g.at(1, 0) = 7.0f;
    g.at(1, 1) = 8.0f;

    t.set_grad(g);
    EXPECT_NEAR(t.grad().at(0, 0), 5.0f, 1e-6f);
    EXPECT_NEAR(t.grad().at(0, 1), 6.0f, 1e-6f);

    t.zero_grad();
    EXPECT_NEAR(t.grad().sum(), 0.0f, 1e-6f);
}

TEST(DeviceBackendTest, MatrixOperations)
{
    DeviceTensor a(2, 3);
    DeviceTensor b(3, 2);

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

    auto res = a.matmul(b);
    EXPECT_EQ(res.at(0, 0), 58.0f);
    EXPECT_EQ(res.at(0, 1), 64.0f);
    EXPECT_EQ(res.at(1, 0), 139.0f);
    EXPECT_EQ(res.at(1, 1), 154.0f);
}

TEST(DeviceBackendTest, DeviceCopySemantics)
{
    DeviceTensor t(2, 3);
    // Populate host data
    t.at(0, 0) = 1.0f;
    t.at(0, 1) = 2.0f;
    t.at(0, 2) = 3.0f;
    t.at(1, 0) = 4.0f;
    t.at(1, 1) = 5.0f;
    t.at(1, 2) = 6.0f;

    auto& backend = t.get_backend();
    EXPECT_FALSE(backend.is_on_device());

    // Copy host->device and verify device buffer contains the same values
    backend.copy_to_device();
    EXPECT_TRUE(backend.is_on_device());
    const float* dptr = backend.device_data_ptr();
    ASSERT_NE(dptr, nullptr);
    EXPECT_NEAR(dptr[0], 1.0f, 1e-6f);
    EXPECT_NEAR(dptr[5], 6.0f, 1e-6f);

    // Mutate device buffer and ensure host unchanged until we copy back
    float* mdptr = backend.mutable_device_data_ptr();
    mdptr[0] = 123.0f;
    EXPECT_NE(t.at(0, 0), 123.0f);

    // Copy device->host and verify host sees the change
    backend.copy_to_host();
    EXPECT_EQ(t.at(0, 0), 123.0f);
}

TEST(DeviceBackendTest, DeviceGradCopySemantics)
{
    DeviceTensor t(2, 3);
    // Set host gradient values
    DeviceTensor g(2, 3);
    g.setZero();
    g.at(0, 0) = 5.0f;
    g.at(0, 1) = 6.0f;
    g.at(0, 2) = 7.0f;
    g.at(1, 0) = 8.0f;
    g.at(1, 1) = 9.0f;
    g.at(1, 2) = 10.0f;

    t.set_grad(g);
    EXPECT_NEAR(t.grad().at(0, 0), 5.0f, 1e-6f);

    auto& backend = t.get_backend();
    EXPECT_FALSE(backend.is_grad_on_device());

    // Copy grad to device and verify device buffer
    backend.copy_grad_to_device();
    EXPECT_TRUE(backend.is_grad_on_device());
    const float* dgptr = backend.device_grad_ptr();
    ASSERT_NE(dgptr, nullptr);
    EXPECT_NEAR(dgptr[0], 5.0f, 1e-6f);
    EXPECT_NEAR(dgptr[5], 10.0f, 1e-6f);

    // Mutate device grad and ensure host grad unchanged until copy back
    float* mdgptr = backend.mutable_device_grad_ptr();
    mdgptr[0] = 321.0f;
    EXPECT_EQ(t.grad().at(0, 0), 5.0f);

    // Copy device grad back to host and verify host sees change
    backend.copy_grad_to_host();
    EXPECT_EQ(t.grad().at(0, 0), 321.0f);
}

TEST(DeviceBackendTest, DeviceDescriptorHelpers)
{
    const nn::Device default_cpu = nn::Device::from_string("");
    EXPECT_TRUE(default_cpu.is_cpu());
    EXPECT_FALSE(default_cpu.is_opencl());
    EXPECT_EQ(default_cpu.to_string(), "cpu");
    EXPECT_FALSE(default_cpu.profiling_enabled);

    const nn::Device explicit_cpu = nn::Device::from_string("cpu");
    EXPECT_TRUE(explicit_cpu.is_cpu());
    EXPECT_EQ(explicit_cpu.to_string(), "cpu");

    const nn::Device opencl = nn::Device::from_string("opencl:0");
    EXPECT_TRUE(opencl.is_opencl());
    EXPECT_FALSE(opencl.is_cpu());
    EXPECT_EQ(opencl.to_string(), "opencl:0");

    const nn::Device custom = nn::Device::from_string("gpu-like");
    EXPECT_TRUE(custom.is_cpu());
    EXPECT_EQ(custom.to_string(), "gpu-like");

    const nn::Device profiled = opencl.with_profiling(true);
    EXPECT_TRUE(profiled.profiling_enabled);
    EXPECT_FALSE(opencl.profiling_enabled);
    EXPECT_EQ(profiled.to_string(), opencl.to_string());
}
