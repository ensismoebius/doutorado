/**
 * @file SpikingAutoencoder.hpp
 * @brief Base class for spiking neural network (SNN) autoencoders.
 *
 * This class provides the foundation for SNN (Spiking Neural Network) autoencoders,
 * which use leaky integrate-and-fire (LIF) neurons instead of RELU activation.
 *
 * SNN vs ANN Key Differences:
 *   - Time-stepped simulation (discrete time steps)
 *   - Membrane potential (voltage) decays over time (leak)
 *   - Binary spiking output (0 or 1 threshold crossings)
 *   - More biologically realistic, energy-efficient
 *
 * Pattern (SNNTorch-style):
 *   For each time step t:
 *     1. Integrate input into membrane potential
 *     2. Apply leak (decay toward rest)
 *     3. Fire if threshold exceeded
 *     4. Reset after spike
 *
 * @note SNN variants ignore ANN parameters (depth, hidden_size) and use
 *       delta_t, resistance, capacitance from config instead.
 */
#ifndef NN_MODELS_AUTOENCODER_SPIKING_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_SPIENCODER_HPP

#include "BaseAutoencoder.hpp"
#include "Config.hpp"

namespace nn::models::autoencoder
{

/**
 * @class SpikingAutoencoder
 * @brief Abstract base for spiking autoencoder models.
 *
 * SNN Theory:
 *   The leaky integrate-and-fire (LIF) neuron model:
 *     V[t] = leak * V[t-1] + (1 - leak) * input
 *     if V[t] > threshold: spike = 1, V[t] = reset
 *     else: spike = 0
 *
 *   Leak rate determined by: tau = RC (time constant)
 *   delta_t (dt) controls simulation resolution.
 */
template <typename Backend>
class SpikingAutoencoder : public BaseAutoencoder<Backend>
{
   public:
    /**
     * @brief Construct SNN autoencoder
     *
     * @param cfg Configuration with SNN hyperparameters
     *   - delta_t: Simulation time step (e.g., 0.1 ms)
     *   - resistance, capacitance: Membrane constants
     */
    explicit SpikingAutoencoder(
        const AutoencoderConfig& cfg, const std::string& name = "SpikingAutoencoder")
        : BaseAutoencoder<Backend>(name),
          delta_t_(cfg.delta_t),
          resistance_(cfg.resistance),
          capacitance_(cfg.capacitance)
    {
    }

    /**
     * @brief Reset membrane state for new sequence
     *
     * Call this at the start of each sequence or epoch
     * to clear accumulated membrane potential.
     */
    virtual void reset_state() = 0;

   protected:
    /** @brief Simulation time step (seconds) */
    float delta_t_;

    /** @brief Membrane resistance (Ohms) */
    float resistance_;

    /** @brief Membrane capacitance (Farads) */
    float capacitance_;
};

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_SPIKING_AUTOENCODER_HPP