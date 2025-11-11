#pragma once

#include "MatFileUtils.h"
#include "TensorDataset.h"

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
            throw std::runtime_error("Error loading data from MAT file");
        }

        Tensor inputs(inputs_mat.value());
        Tensor targets(targets_mat.value());

        // This is a hack to call the base class constructor
        new (this) TensorDataset(inputs, targets);
    }
};
