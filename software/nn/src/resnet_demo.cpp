#include <iostream>
#include <memory>
#include <vector>

#include "dataLoaders/MatFileUtils.h"
#include "initializers/kaiming_snn.hpp"
#include "layers/CrossEntropyLoss.hpp"
#include "layers/Linear.hpp"
#include "layers/ReLU.hpp"
#include "layers/ResidualBlock.hpp"
#include "layers/Sequential.hpp"
#include "optimizers/Adam.hpp"
#include "util/batching.hpp"

using namespace std;

int main()
{
    cout << "ResNet demo: load .mat, train small ResNet-like MLP" << endl;

    const string mat_path = "utils_test.mat"; // example mat in repo root
    const string var_name =
        "data"; // user should ensure this variable exists and is (N x D+1) where last col is label

    auto mat_opt = matioCpp::utils::load_named_variable_as_matrix(mat_path, var_name);
    if (!mat_opt)
    {
        cerr << "Failed to load variable '" << var_name << "' from " << mat_path << endl;
        return 1;
    }

    Eigen::MatrixXf mat = *mat_opt;
    if (mat.cols() < 2)
    {
        cerr << "Matrix must have at least 2 columns (features + label)" << endl;
        return 1;
    }

    const int n_samples = static_cast<int>(mat.rows());
    const int n_features = static_cast<int>(mat.cols() - 1);

    // Build inputs/targets vectors compatible with create_batches
    vector<Tensor> inputs;
    vector<Tensor> targets;
    inputs.reserve(n_samples);
    targets.reserve(n_samples);

    // Determine number of classes from labels
    int max_label = 0;
    for (int i = 0; i < n_samples; ++i)
    {
        int lbl = static_cast<int>(mat(i, mat.cols() - 1));
        max_label = max(max_label, lbl);
    }
    int n_classes = max_label + 1;

    for (int i = 0; i < n_samples; ++i)
    {
        Eigen::MatrixXf x = mat.row(i).leftCols(n_features);
        Eigen::MatrixXf y = Eigen::MatrixXf::Zero(1, n_classes);
        int lbl = static_cast<int>(mat(i, mat.cols() - 1));
        if (lbl >= 0 && lbl < n_classes) y(0, lbl) = 1.0F;

        inputs.emplace_back(x);
        targets.emplace_back(y);
    }

    // Model: input -> Linear -> ReLU -> ResidualBlock x2 -> Linear(output)
    auto fc_in = make_shared<Linear>(n_features, 64);
    auto act = make_shared<ReLU>();
    auto rb1 = make_shared<ResidualBlock>(64);
    auto rb2 = make_shared<ResidualBlock>(64);
    auto fc_out = make_shared<Linear>(64, n_classes);

    Sequential model({fc_in, act, rb1, rb2, fc_out});

    // init weights
    kaimingSNNInitializer(fc_in);
    kaimingSNNInitializer(fc_out);
    kaimingSNNInitializer(rb1->fc1);
    kaimingSNNInitializer(rb1->fc2);
    kaimingSNNInitializer(rb2->fc1);
    kaimingSNNInitializer(rb2->fc2);

    // Loss and optimizer
    CrossEntropyLoss loss;
    vector<Tensor*> params = model.params();
    Adam optimizer(0.001F);
    optimizer.attach(params);

    const int batch_size = 16;
    const int epochs = 10;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto batches = create_batches(inputs, targets, batch_size);
        float epoch_loss = 0.0F;

        for (const auto& b : batches)
        {
            loss.set_target(b.targets);
            Tensor logits = model.forward(b.inputs);
            Tensor loss_tensor = loss.forward(logits);
            Tensor grad_loss = loss.backward(logits);

            model.backward(grad_loss);
            optimizer.step(params);

            epoch_loss += loss_tensor.data(0, 0);
        }

        cout << "Epoch " << epoch << " loss: " << epoch_loss / static_cast<float>(batches.size())
             << "\n";
    }

    cout << "Training finished." << endl;
    return 0;
}
