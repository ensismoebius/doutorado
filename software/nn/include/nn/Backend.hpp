#pragma once
// Single authoritative backend declaration.
// Change this one typedef to switch backends project-wide.
#include "nn/tensor/DeviceTensorBackend.hpp"
#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"
#include "nn/tensor/xtensor/XTensorBackend.hpp"

namespace nn
{
#if defined(NN_BACKEND_OPENCL)
using Backend = OpenCLTensorBackend;
#elif defined(NN_BACKEND_DEVICE)
using Backend = DeviceTensorBackend;
#else
using Backend = XTensorBackend;
#endif
} // namespace nn
