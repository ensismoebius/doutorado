#pragma once
// Single authoritative backend declaration.
// Change this one typedef to switch backends project-wide.

#if defined(NN_BACKEND_DEVICE)
// Device backend: provides: Just a placeholder.
#include "tensor/DeviceTensorBackend.hpp"
namespace nn
{
using Backend = DeviceTensorBackend;
}
#else
// Default to xtensor backend: CPU-only, header-only, no external dependencies.
// Good for development and testing.
#include "tensor/xtensor/XTensorBackend.hpp"
namespace nn
{
using Backend = XTensorBackend;
}
#endif
