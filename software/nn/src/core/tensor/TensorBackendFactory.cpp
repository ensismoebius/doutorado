#include "TensorBackendFactory.hpp"

#include "EigenTensorBackend.hpp"

namespace nn
{

// Static member initialization
TensorBackendFactory::BackendCreator TensorBackendFactory::s_default_creator = []()
{ return std::make_unique<EigenTensorBackend>(); };

void TensorBackendFactory::set_default_backend(BackendCreator creator)
{
    s_default_creator = std::move(creator);
}

std::unique_ptr<ITensorBackend> TensorBackendFactory::create_backend()
{
    return s_default_creator();
}

} // namespace nn