#include "phase00_training.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "core/layers/MSELoss.hpp"
#include "core/optimizers/Adam.hpp"

namespace phase00
{

auto tensor_from_slice(const std::vector<std::vector<double>>& features, size_t start, size_t end)
    -> nn::Tensor
{
    const int rows = static_cast<int>(end - start);
    const int cols = static_cast<int>(features.front().size());
    Eigen::MatrixXf mat(rows, cols);

    for (size_t i = 0; i < static_cast<size_t>(rows); ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            mat(static_cast<int>(i), j) = static_cast<float>(features[start + i][j]);
        }
    }

    return nn::Tensor{mat};
}

auto one_hot_from_slice(const std::vector<int>& labels, size_t start, size_t end, int num_classes)
    -> nn::Tensor
{
    const int rows = static_cast<int>(end - start);
    Eigen::MatrixXf mat = Eigen::MatrixXf::Zero(rows, num_classes);

    for (int i = 0; i < rows; ++i)
    {
        mat(i, labels[start + static_cast<size_t>(i)]) = 1.0F;
    }

    return nn::Tensor{mat};
}

auto compute_accuracy(const Eigen::MatrixXf& logits, const std::vector<int>& labels) -> double
{
    const auto samples = std::min<size_t>(logits.rows(), labels.size());
    if (samples == 0)
    {
        return 0.0;
    }

    size_t correct = 0;
    for (size_t i = 0; i < samples; ++i)
    {
        Eigen::Index idx = 0;
        logits.row(static_cast<int>(i)).maxCoeff(&idx);
        if (static_cast<int>(idx) == labels[i])
        {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(samples);
}

auto train_resnet_snn(const std::vector<std::vector<double>>& features,
                      const std::vector<int>& labels, const Config& cfg, int num_classes)
    -> TrainResult
{
    const int input_dim = static_cast<int>(features.front().size());
    auto model = std::make_unique<SimpleResNet>(
        input_dim, cfg.resnet_hidden_dim, num_classes, cfg.resnet_depth);

    Adam optimizer(cfg.learning_rate);
    auto params = model->params();
    optimizer.attach(params);

    MSELoss loss;

    const int total = static_cast<int>(features.size());
    const int batch_size = std::min(cfg.batch_size, total);

    for (int epoch = 0; epoch < cfg.max_epochs; ++epoch)
    {
        for (int start = 0; start < total; start += batch_size)
        {
            const int end = std::min(start + batch_size, total);

            auto batch_x =
                tensor_from_slice(features, static_cast<size_t>(start), static_cast<size_t>(end));
            auto batch_y = one_hot_from_slice(
                labels, static_cast<size_t>(start), static_cast<size_t>(end), num_classes);

            loss.set_target(batch_y);
            optimizer.zero_grad(params);

            auto preds = model->forward(batch_x, true);
            (void) loss.forward(preds, true);
            auto grad = loss.backward(preds);
            model->backward(grad);
            optimizer.step(params);
        }
    }

    auto full_input = tensor_from_slice(features, 0, features.size());
    auto final_logits = model->forward(full_input, false);
    double accuracy = compute_accuracy(final_logits.get_data_ref(), labels);

    return {accuracy,
            std::move(model),
            input_dim,
            num_classes,
            cfg.resnet_hidden_dim,
            cfg.resnet_depth};
}

auto save_results(const std::filesystem::path& path, double alpha, double beta, double g1,
                  double g2, double accuracy) -> void
{
    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("Unable to open metrics file for writing");
    }

    file << "alpha,beta,g1,g2,accuracy\n";
    file << alpha << "," << beta << "," << g1 << "," << g2 << "," << accuracy << "\n";
}

auto save_torch_state(const std::filesystem::path& path, const TrainResult& trained) -> void
{
    YAML::Node root;
    root["phase"] = 0;
    root["architecture"]["type"] = "SimpleResNet";
    root["architecture"]["input_dim"] = trained.input_dim;
    root["architecture"]["hidden_dim"] = trained.hidden_dim;
    root["architecture"]["output_dim"] = trained.output_dim;
    root["architecture"]["depth"] = trained.depth;

    YAML::Node state_dict(YAML::NodeType::Sequence);
    auto params = trained.model->params();
    for (size_t idx = 0; idx < params.size(); ++idx)
    {
        const auto* param = params[idx];
        YAML::Node param_node;
        param_node["name"] = "param_" + std::to_string(idx);
        param_node["shape"] = YAML::Load("[]");
        param_node["shape"].push_back(static_cast<int>(param->rows()));
        param_node["shape"].push_back(static_cast<int>(param->cols()));

        std::vector<float> flat;
        const auto& mat = param->get_data_ref();
        flat.reserve(static_cast<size_t>(mat.size()));
        for (int r = 0; r < mat.rows(); ++r)
        {
            for (int c = 0; c < mat.cols(); ++c)
            {
                flat.push_back(mat(r, c));
            }
        }

        param_node["data"] = flat;
        state_dict.push_back(param_node);
    }

    root["state_dict"] = state_dict;

    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Unable to open torch state file for writing");
    }
    out << root;
}

} // namespace phase00
