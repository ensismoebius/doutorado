#ifndef RESET_STATE_HPP
#define RESET_STATE_HPP

#include "layers/base/Module.hpp"

namespace nn::utility
{

/**
 * @file reset.hpp
 * @brief Minimal helper matching snnTorch-style `utils.reset(net)` semantics.
 *
 * Why this exists:
 * - Many spiking layers have persistent state (e.g., membrane potential) that should be
 *   reset between independent sequences/epochs.
 * - By routing everything through `Module::reset_state()`, callers can reset an entire
 *   model (including nested `Sequential`s) without knowing its concrete types.
 *
 * What this does NOT do:
 * - It does not touch parameters (weights, thresholds, etc.).
 * - It does not zero gradients (use the Optimizer's `zero_grad` for that).
 */

/**
 * @brief Resets the state of a module (e.g., membrane potentials).
 * Wrapper for snnTorch-like syntax utils.reset(net).
 *
 * @param net The module to reset.
 */
template <typename Backend>
inline void reset(Module<Backend>& net)
{
    // Delegates to the module's virtual reset_state().
    // Sequential containers typically forward the reset into child layers.
    net.reset_state();
}

} // namespace nn::utility

#endif // RESET_STATE_HPP
