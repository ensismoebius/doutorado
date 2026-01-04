#include <iostream>
#include <memory>
#include <vector>

#include "../../core/utility/batching.hpp"
#include "core/dataLoaders/mat_file_utils.hpp"
#include "core/initializers/kaiming_snn.hpp"
#include "core/layers/CrossEntropyLoss.hpp"
#include "core/layers/Linear.hpp"
#include "core/layers/ReLU.hpp"
#include "core/layers/ResidualBlock.hpp"
#include "core/layers/Sequential.hpp"
#include "core/optimizers/Adam.hpp"
#include "core/tensor/Tensor.hpp"

using namespace std;

auto main() -> int
{
    cout << "ResNet demo: load .mat, train small ResNet-like MLP" << '\n';

    const string mat_path =
        "/home/ensismoebius/Documentos/UNESP/"
        "doutorado/databases/BasedeDatosHablaImaginada/S02/"
        "S02_Audio.mat";

    const string var_name = "Audio";

    auto var_names = matioCpp::utils::list_variable_names(mat_path);
    std::cout << "Variables in '" << mat_path << "': ";
    if (var_names.empty())
    {
        std::cout << "(file could not be opened)\n";
    }
    else
    {
        for (const auto& n : var_names)
        {
            std::cout << n << " ";
        }
        std::cout << "\n";
    }

    auto mat_opt = matioCpp::utils::load_named_variable_as_matrix(mat_path, var_name);
    if (!mat_opt)
    {
        cerr << "Failed to load variable '" << var_name << "' from " << mat_path << '\n';
        return 1;
    }

    nn::Tensor mat = std::move(*mat_opt);
    if (mat.cols() < 2)
    {
        cerr << "Matrix must have at least 2 columns (features + label)" << '\n';
        return 1;
    }

    const int n_samples = static_cast<int>(mat.rows());
    const int n_features = static_cast<int>(mat.cols() - 1);

    // Build inputs/targets vectors compatible with create_batches
    vector<nn::Tensor> inputs;
    vector<nn::Tensor> targets;
    inputs.reserve(n_samples);
    targets.reserve(n_samples);

    // Determine number of classes from labels
    int max_label = 0;
    for (int i = 0; i < n_samples; ++i)
    {
        int lbl = static_cast<int>(mat.at(i, mat.cols() - 1));
        max_label = max(max_label, lbl);
    }
    int n_classes = max_label + 1;

    for (int i = 0; i < n_samples; ++i)
    {
        nn::Tensor x = mat.row(i).leftCols(n_features);
        nn::Tensor y(1, n_classes);
        int lbl = static_cast<int>(mat.at(i, mat.cols() - 1));
        if (lbl >= 0 && lbl < n_classes)
        {
            y.at(0, lbl) = 1.0F;
        }

        inputs.emplace_back(std::move(x));
        targets.emplace_back(std::move(y));
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
    vector<nn::Tensor*> params = model.params();
    Adam optimizer(0.001F);
    optimizer.attach(params);

    const int batch_size = 16;
    const int epochs = 1; // single-epoch demo run

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto batches = create_batches(inputs, targets, batch_size);
        float epoch_loss = 0.0F;

        for (const auto& b : batches)
        {
            loss.set_target(b.targets);
            nn::Tensor logits = model.forward(b.inputs);
            nn::Tensor loss_tensor = loss.forward(logits);
            nn::Tensor grad_loss = loss.backward(logits);

            model.backward(grad_loss);
            optimizer.step(params);

            epoch_loss += loss_tensor(0, 0);
        }

        cout << "Epoch " << epoch << " loss: " << epoch_loss / static_cast<float>(batches.size())
             << "\n";
    }

    cout << "Training finished." << '\n';
    return 0;
}
