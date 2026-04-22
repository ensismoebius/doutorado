/**
 * @file include/nn/layers/eigen/Layers.hpp
 * @brief Eigen-backend type aliases for all layer templates.
 *
 * This is the ONLY file in the codebase that may reference `nn::EigenTensorBackend`
 * together with layer template names. All layer template headers (`Linear.hpp`,
 * `Conv2d.hpp`, etc.) are backend-agnostic; the concrete names (`Linear`, `ReLU`,
 * etc.) that downstream code uses are defined here.
 *
 * Usage:
 *   #include "nn/layers/eigen/Layers.hpp"  // provides Linear, ReLU, Conv2d, ...
 *
 * Convention:
 *   `Xxx = XxxImpl<nn::EigenTensorBackend>` for every layer in this project.
 */

#ifndef NN_LAYERS_EIGEN_LAYERS_HPP
#define NN_LAYERS_EIGEN_LAYERS_HPP

// --- Eigen backend (the only Eigen reference allowed in the layers subtree) ---
#include "nn/tensor/eigen/EigenTensorBackend.hpp"

// --- Layer templates ---
#include "nn/layers/activations/LeakyReLU.hpp"
#include "nn/layers/activations/ReLU.hpp"
#include "nn/layers/base/Sequential.hpp"
#include "nn/layers/convolution/Conv1d.hpp"
#include "nn/layers/convolution/Conv2d.hpp"
#include "nn/layers/convolution/MaxPool1d.hpp"
#include "nn/layers/convolution/MaxPool2d.hpp"
#include "nn/layers/dense/Linear.hpp"
#include "nn/layers/losses/CrossEntropyLoss.hpp"
#include "nn/layers/losses/MAELoss.hpp"
#include "nn/layers/losses/MSELoss.hpp"
#include "nn/layers/losses/SpikeCountLoss.hpp"
#include "nn/layers/residual/ResNetBlock.hpp"
#include "nn/layers/residual/ResidualBlock.hpp"
#include "nn/layers/residual/SimpleResNet.hpp"
#include "nn/layers/spiking/Leaky.hpp"
#include "nn/layers/spiking/LeakyBPTT.hpp"
#include "nn/layers/spiking/LeakyIntegrator.hpp"

// --- Concrete aliases (Eigen backend) ---

/// Fully-connected affine layer on the Eigen backend.
using Linear = LinearImpl<nn::EigenTensorBackend>;

/// Rectified-linear activation on the Eigen backend.
using ReLU = ReLUImpl<nn::EigenTensorBackend>;

/// Leaky-ReLU activation on the Eigen backend.
using LeakyReLU = LeakyReLUImpl<nn::EigenTensorBackend>;

/// Leaky Integrate-and-Fire neuron layer on the Eigen backend.
using Leaky = LeakyImpl<nn::EigenTensorBackend>;

/// Leaky BPTT neuron layer on the Eigen backend.
using LeakyBPTT = LeakyBPTTImpl<nn::EigenTensorBackend>;

/// Continuous leaky integrator (non-spiking readout) on the Eigen backend.
using LeakyIntegrator = LeakyIntegratorImpl<nn::EigenTensorBackend>;

/// 1-D convolution layer on the Eigen backend.
using Conv1d = Conv1dImpl<nn::EigenTensorBackend>;

/// 2-D convolution layer on the Eigen backend.
using Conv2d = Conv2dImpl<nn::EigenTensorBackend>;

/// MaxPooling layer (1D) on the Eigen backend.
using MaxPool1d = MaxPool1dImpl<nn::EigenTensorBackend>;

/// MaxPooling layer (2D) on the Eigen backend.
using MaxPool2d = MaxPool2dImpl<nn::EigenTensorBackend>;

/// Mean-squared-error loss on the Eigen backend.
using MSELoss = MSELossImpl<nn::EigenTensorBackend>;

/// Mean-absolute-error loss on the Eigen backend.
using MAELoss = MAELossImpl<nn::EigenTensorBackend>;

/// Cross-entropy loss on the Eigen backend.
using CrossEntropyLoss = CrossEntropyLossImpl<nn::EigenTensorBackend>;

/// Spike-count loss on the Eigen backend.
using SpikeCountLoss = SpikeCountLossImpl<nn::EigenTensorBackend>;

/// Residual block on the Eigen backend.
using ResidualBlock = ResidualBlockImpl<nn::EigenTensorBackend>;

/// ResNet block on the Eigen backend.
using ResNetBlock = ResNetBlockImpl<nn::EigenTensorBackend>;

/// Simple residual network on the Eigen backend.
using SimpleResNet = SimpleResNetImpl<nn::EigenTensorBackend>;

/// Sequential container on the Eigen backend.
using Sequential = SequentialImpl<nn::EigenTensorBackend>;

// --- Explicit instantiation declarations (defined in *_impl.cpp / *_utils.cpp) ---
extern template class Conv1dImpl<nn::EigenTensorBackend>;
extern template class Conv2dImpl<nn::EigenTensorBackend>;

#endif // NN_LAYERS_EIGEN_LAYERS_HPP
