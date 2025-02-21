#ifndef MODEL
#define MODEL

#include "Layer.cpp"
#include <iostream>

class Model
{
private:
    std::vector<Layer *> layers;

public:
    Model(std::initializer_list<Layer *> layer_list) : layers(layer_list) {}

    ~Model()
    {
        for (auto *layer : layers)
        {
            delete layer;
        }
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input)
    {
        std::cout << "\nInput:\n"
                  << input << std::endl;

        Eigen::MatrixXd output = input;
        for (auto *layer : layers)
        {
            output = layer->forward(output);
        }

        std::cout << "\nOutput:\n"
                  << output << std::endl;
        return output;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output)
    {
        std::cout << "\nGradient input:\n"
                  << grad_output << std::endl;

        Eigen::MatrixXd grad = grad_output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it)->backward(grad);
        }

        std::cout << "\nGradient output:\n"
                  << grad << std::endl;

        return grad;
    }

    std::map<std::string, Eigen::MatrixXd> parameters()
    {
        std::map<std::string, Eigen::MatrixXd> params;
        int idx = 0;
        for (auto *layer : layers)
        {
            auto layer_params = layer->parameters();
            for (const auto &[name, param] : layer_params)
            {
                params[std::to_string(idx) + "." + name] = param;
            }
            idx++;
        }
        return params;
    }

    std::map<std::string, Eigen::MatrixXd> gradients()
    {
        std::map<std::string, Eigen::MatrixXd> grads;
        int idx = 0;
        for (auto *layer : layers)
        {
            auto layer_grads = layer->gradients();
            for (const auto &[name, grad] : layer_grads)
            {
                grads[std::to_string(idx) + "." + name] = grad;
            }
            idx++;
        }
        return grads;
    }
};

#endif