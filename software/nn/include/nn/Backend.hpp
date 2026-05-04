#pragma once
// Single authoritative backend declaration.
// Change this one typedef to switch backends project-wide.

#if defined(NN_BACKEND_OPENCL)
#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"
namespace nn
{
using Backend = OpenCLTensorBackend;
}
#elif defined(NN_BACKEND_DEVICE)
#include "nn/tensor/DeviceTensorBackend.hpp"
namespace nn
{
using Backend = DeviceTensorBackend;
}
#else
#include "nn/tensor/xtensor/XTensorBackend.hpp"
namespace nn
{
using Backend = XTensorBackend;
} 
#endif
