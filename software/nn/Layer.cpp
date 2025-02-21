#ifndef LAYER
#define LAYER

#include <map>
#include <string>
#include <Eigen/Dense>

class Layer
{
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) = 0;
    virtual std::map<std::string, Eigen::MatrixXd> parameters() = 0;
    virtual std::map<std::string, Eigen::MatrixXd> gradients() = 0;
    virtual ~Layer() = default;
};

#endif