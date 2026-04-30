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
#include <vector>

#include "nn/dataLoaders/io/mat_file_utils.hpp"
#include "nn/initializers/kaiming_snn.hpp"
#include "nn/layers/Layers.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/testing.hpp"
#include "nn/utility/batching.hpp"

using namespace std;

auto main() -> int
{
    try
    {
        NN_LOG_INFO("ResNet demo: load .mat, train small ResNet-like MLP");

        const string mat_path =
            "/home/ensismoebius/Documentos/UNESP/"
            "doutorado/databases/BasedeDatosHablaImaginada/S02/"
            "S02_Audio.mat";

        const string var_name = "Audio";

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
            return 1;
        }

        nn::Tensor mat = std::move(*mat_opt);
        if (mat.cols() < 2)
        {
            NN_LOG_ERROR("Matrix must have at least 2 columns (features + label)");
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
        kaimingSNNInitializer(fc_in, nn::testing::kSeed);
        kaimingSNNInitializer(fc_out, nn::testing::kSeed);
        kaimingSNNInitializer(rb1->fc1, nn::testing::kSeed);
        kaimingSNNInitializer(rb1->fc2, nn::testing::kSeed);
        kaimingSNNInitializer(rb2->fc1, nn::testing::kSeed);
        kaimingSNNInitializer(rb2->fc2, nn::testing::kSeed);

        // Loss and optimizer
        CrossEntropyLoss loss;
        auto params = model.params();
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

            cout << "Epoch " << epoch
                 << " loss: " << epoch_loss / static_cast<float>(batches.size()) << "\n";
        }

        cout << "Training finished." << '\n';
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
