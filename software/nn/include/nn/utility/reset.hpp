#ifndef RESET_STATE_HPP
#define RESET_STATE_HPP

#include "nn/layers/Module.hpp"

namespace nn::utility
{

/**
 * @brief Resets the state of a module (e.g., membrane potentials).
 * Wrapper for snnTorch-like syntax utils.reset(net).
 *
 * @param net The module to reset.
 */
inline void reset(Module& net)
{
    net.reset_state();
}

} // namespace nn::utility

#endif // RESET_STATE_HPP
