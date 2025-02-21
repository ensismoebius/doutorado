#ifndef LINEAR
#define LINEAR

#include <map>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include "Layer.cpp"

// Linear Layer
class Linear : public Layer
{
private:
    Eigen::MatrixXd W, b, input, gradW, gradb;

public:
    Linear(int in_features, int out_features)
    {
        W = Eigen::MatrixXd::Ones(in_features, out_features) * 0.1;
        b = Eigen::MatrixXd::Zero(1, out_features);
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd &x) override
    {
        // Input is stored for later use in the backward pass to compute gradients
        input = x;
        // Adds b to each row from (x * W) making 
        // it work even when the input is a batch of vectors
        auto result = (x * W).rowwise() + b.row(0);

        std::cout << "\nWeight:\n" << W << std::endl;
        std::cout << "\nParcial output:\n" << result << std::endl;

        return result;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
    {
        gradW = input.transpose() * grad_output;
        gradb = grad_output.colwise().sum();

        std::cout << "\nInput transpose:\n" << input.transpose();
        std::cout << "\nWeight gradients:\n" << gradW;
        std::cout << "\nBias gradients:\n" << gradb;
        std::cout << "\nNew weights:\n" << grad_output * W.transpose();
        
        return grad_output * W.transpose();
    }

    std::map<std::string, Eigen::MatrixXd> parameters() override
    {
        return {{"W", W}, {"b", b}};
    }

    std::map<std::string, Eigen::MatrixXd> gradients() override
    {
        return {{"W", gradW}, {"b", gradb}};
    }
};

#endif