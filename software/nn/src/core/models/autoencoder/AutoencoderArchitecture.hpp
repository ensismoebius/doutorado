#ifndef NN_MODELS_AUTOENCODER_ARCHITECTURE_HPP
#define NN_MODELS_AUTOENCODER_ARCHITECTURE_HPP

namespace nn::models::autoencoder
{

enum class AutoencoderArchitecture
{
    Auto,
    ResidualDense,
    DualBranchFusion
};

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_ARCHITECTURE_HPP