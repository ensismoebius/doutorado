#ifndef TENSOR_BACKEND_FACTORY_HPP
#define TENSOR_BACKEND_FACTORY_HPP

#include <functional>
#include <memory>

#include "nn/tensor/ITensorBackend.hpp"

namespace nn
{

class TensorBackendFactory
{
   public:
    using BackendCreator = std::function<std::unique_ptr<ITensorBackend>()>;

    static void set_default_backend(BackendCreator creator);
    static std::unique_ptr<ITensorBackend> create_backend();

   private:
    static BackendCreator s_default_creator;
};

} // namespace nn

#endif // TENSOR_BACKEND_FACTORY_HPP