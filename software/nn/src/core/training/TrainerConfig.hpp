#ifndef NN_TRAINING_TRAINER_CONFIG_HPP
#define NN_TRAINING_TRAINER_CONFIG_HPP

namespace nn::training
{

struct TrainerConfig
{
    int epochs = 10;
    float learning_rate = 0.001F;
    float adam_beta1 = 0.9F;
    float adam_beta2 = 0.999F;
    float adam_epsilon = 1e-8F;
    float grad_clip_norm = 0.0F;
    int batch_size = 1;
    unsigned int sampler_shuffle_seed = 42;
};

} // namespace nn::training

#endif // NN_TRAINING_TRAINER_CONFIG_HPP