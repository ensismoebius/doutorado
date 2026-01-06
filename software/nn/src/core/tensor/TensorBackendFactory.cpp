#include "nn/tensor/TensorBackendFactory.hpp"

#include "EigenTensorBackend.hpp"

namespace nn
{

// -----------------------------------------------------------------------------
// TensorBackendFactory
// Simple factory for creating concrete `ITensorBackend` instances. The factory
// stores a function (BackendCreator) that is used to materialize backends on
// demand. By default the creator constructs a minimal Eigen-based backend.
//
// Notes:
// - The default creator returns a small (1x1 zero) Eigen-backed tensor; callers
//   should immediately reshape/resize as required by their use-case.
// - The creator is stored in a static variable; if the project requires
//   thread-safety during early static initialization in multithreaded contexts,
//   ensure `set_default_backend` is called during single-threaded startup.
// -----------------------------------------------------------------------------

// Static member initialization: default creator returns a tiny Eigen backend.
TensorBackendFactory::BackendCreator TensorBackendFactory::s_default_creator = []()
{ return std::make_unique<EigenTensorBackend>(Eigen::MatrixXf::Zero(1, 1)); };

// Replace the default backend creator. The caller provides a callable that
// returns a `std::unique_ptr<ITensorBackend>`; ownership is transferred into
// the factory. Use this to inject alternate backends for testing or
// experimentation.
void TensorBackendFactory::set_default_backend(BackendCreator creator)
{
    s_default_creator = std::move(creator);
}

// Create a new backend instance using the currently registered creator.
std::unique_ptr<ITensorBackend> TensorBackendFactory::create_backend()
{
    return s_default_creator();
}

} // namespace nn