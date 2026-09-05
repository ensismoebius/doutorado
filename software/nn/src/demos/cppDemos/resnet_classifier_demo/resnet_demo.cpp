/**
 * @file resnet_demo.cpp
 * @brief Minimal classification demo: load a `.mat` matrix and train a small residual MLP.
 *
 * Notes:
 * - The current path is hard-coded for a local dataset checkout (demo convenience).
 * - Targets are built as one-hot vectors and trained with `CrossEntropyLoss`.
 */

#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "data_loaders/mat_io/mat_file_utils.hpp"
#include "initializers/kaiming_snn.hpp"
#include "layers/Layers.hpp"
#include "logging/Logger.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"
#include "testing.hpp"
#include "utility/batching.hpp"

using nn::CrossEntropyLoss;
using nn::Linear;
using nn::ReLU;
using nn::ResidualBlock;
using nn::Sequential;

namespace
{

// Lists the .mat file's variables (logged for diagnostics), loads `var_name` as a matrix, and
// validates it has at least a features column and a label column. Returns nullopt (after
// logging the specific cause) when any step fails.
auto load_and_validate_mat_matrix(const std::string& mat_path, const std::string& var_name)
    -> std::optional<nn::Tensor>
{
    auto var_names = matioCpp::utils::list_variable_names(mat_path);
    if (var_names.empty())
    {
        NN_LOG_WARN("Variables in '" + mat_path + "': (file could not be opened)");
    }
    else
    {
        std::string vars;
        for (const auto& n : var_names) vars += n + " ";
        NN_LOG_INFO("Variables in '" + mat_path + "': " + vars);
    }

    auto mat_opt = matioCpp::utils::load_named_variable_as_matrix(mat_path, var_name);
    if (!mat_opt)
    {
        NN_LOG_ERROR("Failed to load variable '" + var_name + "' from " + mat_path);
        return std::nullopt;
    }

    nn::Tensor mat = std::move(*mat_opt);
    if (mat.cols() < 2)
    {
        NN_LOG_ERROR("Matrix must have at least 2 columns (features + label)");
        return std::nullopt;
    }

    return mat;
}

// Splits the last column off as an integer class label, one-hot encodes it, and returns
// (inputs, targets, n_features, n_classes) ready for create_batches().
auto build_classification_dataset(const nn::Tensor& mat)
    -> std::tuple<std::vector<nn::Tensor>, std::vector<nn::Tensor>, int, int>
{
    const int n_samples = static_cast<int>(mat.rows());
    const int n_features = static_cast<int>(mat.cols() - 1);

    // Build inputs/targets vectors compatible with create_batches
    std::vector<nn::Tensor> inputs;
    std::vector<nn::Tensor> targets;
    inputs.reserve(n_samples);
    targets.reserve(n_samples);

    // Determine number of classes from labels
    int max_label = 0;
    for (int i = 0; i < n_samples; ++i)
    {
        int lbl = static_cast<int>(mat.at(i, mat.cols() - 1));
        max_label = std::max(max_label, lbl);
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

    return {std::move(inputs), std::move(targets), n_features, n_classes};
}

// Model: input -> Linear -> ReLU -> ResidualBlock x2 -> Linear(output), Kaiming-initialized.
auto build_resnet_classifier(int n_features, int n_classes) -> Sequential
{
    auto fc_in = std::make_shared<Linear>(n_features, 64);
    auto act = std::make_shared<ReLU>();
    auto rb1 = std::make_shared<ResidualBlock>(64);
    auto rb2 = std::make_shared<ResidualBlock>(64);
    auto fc_out = std::make_shared<Linear>(64, n_classes);

    // init weights
    kaimingSNNInitializer(fc_in, nn::testing::kSeed);
    kaimingSNNInitializer(fc_out, nn::testing::kSeed);
    kaimingSNNInitializer(rb1->fc1, nn::testing::kSeed);
    kaimingSNNInitializer(rb1->fc2, nn::testing::kSeed);
    kaimingSNNInitializer(rb2->fc1, nn::testing::kSeed);
    kaimingSNNInitializer(rb2->fc2, nn::testing::kSeed);

    return Sequential({fc_in, act, rb1, rb2, fc_out});
}

// Runs `epochs` passes over the batched dataset, printing mean per-epoch loss.
void run_training_loop(Sequential& model,
    CrossEntropyLoss& loss,
    Adam& optimizer,
    std::span<nn::Tensor*> params,
    const std::vector<nn::Tensor>& inputs,
    const std::vector<nn::Tensor>& targets,
    int batch_size,
    int epochs)
{
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

        std::cout << "Epoch " << epoch
                  << " loss: " << epoch_loss / static_cast<float>(batches.size()) << "\n";
    }
}

} // namespace

auto main() -> int
{
    try
    {
        NN_LOG_INFO("ResNet demo: load .mat, train small ResNet-like MLP");

        const std::string mat_path =
            "/home/ensismoebius/Documentos/UNESP/"
            "doutorado/databases/BasedeDatosHablaImaginada/S02/"
            "S02_Audio.mat";

        const std::string var_name = "Audio";

        auto mat_opt = load_and_validate_mat_matrix(mat_path, var_name);
        if (!mat_opt) return 1;
        nn::Tensor mat = std::move(*mat_opt);

        auto [inputs, targets, n_features, n_classes] = build_classification_dataset(mat);

        Sequential model = build_resnet_classifier(n_features, n_classes);

        // Loss and optimizer
        CrossEntropyLoss loss;
        auto params = model.params();
        Adam optimizer(0.001F);
        optimizer.attach(params);

        const int batch_size = 16;
        const int epochs = 1; // single-epoch demo run

        run_training_loop(model, loss, optimizer, params, inputs, targets, batch_size, epochs);

        std::cout << "Training finished." << '\n';
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred." << std::endl;
        return 1;
    }
}
