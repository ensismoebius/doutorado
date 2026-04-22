#ifndef NN_TRAINING_EPOCH_RESULT_HPP
#define NN_TRAINING_EPOCH_RESULT_HPP

namespace nn::training
{

struct EpochResult
{
    int epoch;
    float train_loss;
    float val_loss;
    float epoch_ms;
};

} // namespace nn::training

#endif // NN_TRAINING_EPOCH_RESULT_HPP