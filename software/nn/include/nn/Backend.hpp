#pragma once
// Single authoritative backend declaration.
// Change this one typedef to switch backends project-wide.
#include "nn/tensor/eigen/EigenTensorBackend.hpp"
namespace nn {
    using Backend = EigenTensorBackend;
}
