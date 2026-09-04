#ifndef NN_MODELS_AUTOENCODER_ENCODER_DECODER_AUTOENCODER_HPP
#define NN_MODELS_AUTOENCODER_ENCODER_DECODER_AUTOENCODER_HPP

#include <vector>

#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

/**
 * @brief Concatenate several modules' parameter pointers into `storage`.
 *
 * Every autoencoder here has to answer `params()` with one flat span over
 * ALL its parts, and each was spelling out the same accumulate loop once per
 * part -- six times over in the fused variants. The loop is not the
 * interesting part of those functions; which modules are in the model is.
 *
 * `storage` must be a member of the caller: the returned span points into
 * it, so it has to outlive the call (the Module contract in CLAUDE.md).
 */
template <typename... Modules>
auto collect_params(std::vector<nn::Tensor*>& storage, Modules&... parts) -> std::span<nn::Tensor*>
{
    storage.clear();
    (
        [&storage](auto& part)
        {
            auto part_params = part.params();
            storage.insert(storage.end(), part_params.begin(), part_params.end());
        }(parts),
        ...);
    return std::span<nn::Tensor*>{storage.data(), storage.size()};
}

/**
 * @file EncoderDecoderAutoencoder.hpp
 * @brief The plumbing every "two Sequentials" autoencoder shares.
 *
 * The window autoencoders (audio/EEG, ANN/spiking) differ in exactly one
 * thing: which layers their encoder and decoder are built from. Everything
 * else -- run the encoder, run the decoder, chain them for forward, chain
 * them backwards for backward, concatenate the two parameter lists, reset
 * both states -- was written out identically in each of the four, four
 * copies of the same thirty lines.
 *
 * Copies drift. A fix applied to one `params()` and not the other three is
 * invisible until an optimizer silently trains half a network, which is the
 * kind of defect that does not crash and does not show up in a loss curve
 * until much later.
 *
 * So a subclass supplies the two Sequentials and nothing else:
 *
 *     AudioWindowAutoencoder::AudioWindowAutoencoder(const AutoencoderConfig& cfg)
 *         : EncoderDecoderAutoencoder(build_ann_encoder(cfg, ...),
 *                                     build_ann_decoder(cfg, ...))
 *     {
 *     }
 *
 * Deliberately NOT used by the fused and protocol autoencoders: their
 * `encode`/`decode` do real work of their own (multi-branch inputs, per-modality
 * heads), so they are not "two Sequentials" and forcing them into this base
 * would hide that difference rather than remove duplication.
 */
struct EncoderDecoderAutoencoder : Module<nn::Backend>
{
    using Tensor = typename Module<nn::Backend>::Tensor;

    nn::Sequential encoder_;
    nn::Sequential decoder_;

    EncoderDecoderAutoencoder(nn::Sequential encoder, nn::Sequential decoder);

    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;
    auto decode(const Tensor& latent, bool requires_grad = true) -> Tensor;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;

    /// Owned concatenation of encoder + decoder parameter pointers.
    ///
    /// A member, not a local: `params()` returns a span over it, so it has to
    /// outlive the call (Module contract -- see CLAUDE.md).
    std::vector<Tensor*> param_ptrs_;
    auto params() -> std::span<Tensor*> override;

    /// Clears both halves' layer state between independent sequences.
    ///
    /// A no-op for the ANN variants (their layers hold no state) and the
    /// whole job for the spiking ones (membrane potentials), which is why it
    /// lives here rather than being repeated in each spiking subclass.
    void reset_state() override;
};

#endif // NN_MODELS_AUTOENCODER_ENCODER_DECODER_AUTOENCODER_HPP
