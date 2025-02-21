#ifndef RELU
#define RELU

#include "Layer.cpp"

class ReLU : public Layer
{
private:
    Eigen::MatrixXd input;

public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &x) override
    {
        input = x;
        return x.unaryExpr([](double elem)
                           { return std::max(0.0, elem); });
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
    {
        return grad_output.array() * (input.array() > 0).cast<double>();
    }

    std::map<std::string, Eigen::MatrixXd> parameters() override
    {
        return {};
    }

    std::map<std::string, Eigen::MatrixXd> gradients() override
    {
        return {};
    }
};

#endif