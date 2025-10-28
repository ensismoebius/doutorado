#ifndef SURROGATE_GRADIENT_HPP
#define SURROGATE_GRADIENT_HPP

#include <Eigen/Dense>

// Interface for surrogate gradient functions
class ISurrogateGradient
{
   public:
    virtual ~ISurrogateGradient() = default;

    [[nodiscard]] virtual auto calculate(const Eigen::MatrixXf& v_mem_pre_spike,
                                         float voltage_threshold) const -> Eigen::MatrixXf = 0;
};

// Exponential / SuperSpike surrogate gradient
class ExponentialSurrogate : public ISurrogateGradient
{
   public:
    explicit ExponentialSurrogate(float sharpness = 1.0F) : sharpness_(sharpness) {}

    [[nodiscard]] auto calculate(const Eigen::MatrixXf& v_mem_pre_spike,
                                 float voltage_threshold) const -> Eigen::MatrixXf override
    {
        const Eigen::MatrixXf diff_abs = (v_mem_pre_spike.array() - voltage_threshold).abs();

        return (1.0F / sharpness_) * ((-diff_abs / sharpness_).array().exp());
    }

   private:
    float sharpness_;
};

// Boxcar surrogate gradient
class BoxcarSurrogate : public ISurrogateGradient
{
   public:
    explicit BoxcarSurrogate(float window = 0.5F) : window_(window) {}

    [[nodiscard]] auto calculate(const Eigen::MatrixXf& v_mem_pre_spike,
                                 float voltage_threshold) const -> Eigen::MatrixXf override
    {
        const Eigen::MatrixXf diff_abs = (v_mem_pre_spike.array() - voltage_threshold).abs();

        return (diff_abs.array() < (window_ / 2.0F)).cast<float>();
    }

   private:
    float window_;
};

#endif // SURROGATE_GRADIENT_HPP
