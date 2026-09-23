#ifndef NN_UTILITY_PARAMETERCOUNT_HPP
#define NN_UTILITY_PARAMETERCOUNT_HPP

#include <cstddef>
#include <span>

namespace nn::utils
{

// Total element count across a Module::params() span (e.g. model.params()). Null
// entries are skipped rather than dereferenced, since some layers report fixed-size
// param spans with optional slots (bias-free layers, disabled adapters).
template <typename Tensor>
inline auto parameter_count(std::span<Tensor*> params) -> std::size_t
{
    std::size_t count = 0;
    for (Tensor* p : params)
    {
        if (!p) continue;
        count += static_cast<std::size_t>(p->size());
    }
    return count;
}

} // namespace nn::utils

#endif // NN_UTILITY_PARAMETERCOUNT_HPP
