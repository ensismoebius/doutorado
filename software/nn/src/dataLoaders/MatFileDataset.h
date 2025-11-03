#pragma once

#include "dataLoaders/MatFileUtils.h"
#include "dataLoaders/TensorDataset.h"

class MatFileDataset : public TensorDataset
{
   public:
    MatFileDataset(const std::string& mat_path, const std::string& input_var_name,
                   const std::string& target_var_name)
    {
        auto inputs_mat = matioCpp::utils::load_named_variable_as_matrix(mat_path, input_var_name);
        auto targets_mat =
            matioCpp::utils::load_named_variable_as_matrix(mat_path, target_var_name);

        if (!inputs_mat || !targets_mat)
        {
            // Diagnostic output for test failure investigation
            std::cerr << "MatFileDataset: failed to load variables from '" << mat_path << "'\n";
            if (!std::filesystem::exists(mat_path))
            {
                std::cerr << "MatFileDataset: file does not exist: " << mat_path << "\n";
            }
            else
            {
                std::cerr << "MatFileDataset: file exists. inputs_mat.has_value()="
                          << (inputs_mat.has_value() ? "true" : "false")
                          << ", targets_mat.has_value()="
                          << (targets_mat.has_value() ? "true" : "false") << "\n";
                auto names = matioCpp::utils::list_variable_names(mat_path);
                std::cerr << "MatFileDataset: variables in file:";
                for (const auto& n : names) std::cerr << ' ' << n;
                std::cerr << '\n';
            }

            throw std::runtime_error("Error loading data from MAT file");
        }

        Tensor inputs(inputs_mat.value());
        Tensor targets(targets_mat.value());

        // This is a hack to call the base class constructor
        new (this) TensorDataset(inputs, targets);
    }
};
