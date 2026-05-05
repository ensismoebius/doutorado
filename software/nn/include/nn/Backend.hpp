#pragma once
// Single authoritative backend declaration.
// Change this one typedef to switch backends project-wide.

#if defined(NN_BACKEND_OPENCL)
// OpenCL backend: requires OpenCL headers and runtime.
// Provides GPU acceleration on compatible devices.
#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"
namespace nn
{
using Backend = OpenCLTensorBackend;
}
#elif defined(NN_BACKEND_DEVICE)
// Device backend: provides: Just a placeholder.
#include "nn/tensor/DeviceTensorBackend.hpp"
namespace nn
{
using Backend = DeviceTensorBackend;
}
#else
// Default to xtensor backend: CPU-only, header-only, no external dependencies.
// Good for development and testing.
#include "nn/tensor/xtensor/XTensorBackend.hpp"
namespace nn
{
using Backend = XTensorBackend;
}
#endif
