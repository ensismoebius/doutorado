#ifndef NN_TRAINING_TRAINER_CONFIG_HPP
#define NN_TRAINING_TRAINER_CONFIG_HPP

namespace nn::training
{

/**
 * @brief Configuration for the generic Trainer loop.
 *
 * SNN-specific fields (snn_lr_scale, nested_cv_*) are ignored for pure ANN models.
 *
 * **SNN learning rate guidance** (Frontiers Neuroscience 2025):
 * SNN biophysical parameters (R, C, V_th) are sensitive to large updates.
 * Literature recommends lr ≈ 1e-4 for SNN params vs 1e-3 for weight matrices.
 * Set snn_lr_scale = 0.1 to apply this 10× reduction automatically when the
 * optimizer is called with per-group learning rates via Adam::attach_with_scales().
 *
 * **Nested CV** (PMC guide 2023 [41]):
 * For unbiased hyperparameter evaluation on biomedical data, enable nested k-fold:
 * - nested_cv_outer_folds: outer test folds (unbiased performance estimate)
 * - nested_cv_inner_folds: inner val folds (hyperparameter selection)
 */
struct TrainerConfig
{
    int epochs = 10;
    float learning_rate = 0.001F;
    float adam_beta1 = 0.9F;
    float adam_beta2 = 0.999F;
    float adam_epsilon = 1e-8F;
    float grad_clip_norm = 0.0F;

    /// L2 weight decay (decoupled / AdamW-style). 0 = disabled.
    /// Applied only to 2-D weight matrices; biases and SNN biophysical scalars
    /// (R, C, V_th — shape 1×1 or N×1) are excluded so the membrane time
    /// constant tau=R·C and the threshold are never pulled toward zero.
    /// Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019.
    float weight_decay = 0.0F;

    int batch_size = 1;
    unsigned int sampler_shuffle_seed = 42;

    /// Learning rate scale for SNN biophysical parameters (R, C, V_th).
    /// Effective SNN lr = learning_rate * snn_lr_scale.
    /// Recommended: 0.1 (10× smaller than weight lr). Set 1.0 to disable.
    float snn_lr_scale = 0.1F;

    /// Number of outer folds for nested cross-validation (0 = disabled).
    int nested_cv_outer_folds = 0;

    /// Number of inner folds per outer fold for hyperparameter selection.
    int nested_cv_inner_folds = 5;
};

} // namespace nn::training

#endif // NN_TRAINING_TRAINER_CONFIG_HPP
