#ifndef NN_TRAINING_EPOCH_RESULT_HPP
#define NN_TRAINING_EPOCH_RESULT_HPP

namespace nn::training
{

struct EpochResult
{
    int epoch = 0;
    float train_loss = 0.0F;
    float val_loss = 0.0F;
    float epoch_ms = 0.0F;
};

} // namespace nn::training

#endif // NN_TRAINING_EPOCH_RESULT_HPP