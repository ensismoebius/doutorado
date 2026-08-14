#pragma once

#include <map>
#include <string>

#include "tensor/Tensor.hpp"

namespace pga
{

// Persist a trained model's full parameter map (Module::state_dict()) to a generic
// NumPy .npz — one array per key, no architecture assumptions (unlike
// NetworkSerializer, which is coupled to Sequential/specific layer types; the Protocol
// autoencoders used here are not Sequential). Read back generically by
// scripts/data/npz_to_pytorch.py (--models-dir results/paraconsistentGA/models), which
// already handles arbitrary .npz key->array pairs.
//
// Throws if `state_dict` is empty — an empty snapshot means the caller has a bug (see
// .wiki/Experiments/ParaconsistentGA-Design.md §6.2), not a legitimate "nothing to
// save" case; per the no-fallback policy this must not silently write an empty file.
void save_state_dict_npz(
    const std::map<std::string, nn::Tensor>& state_dict, const std::string& path);

} // namespace pga
