#ifndef NN_MATFILEDATASET_H
#define NN_MATFILEDATASET_H

#include "TensorDataset.hpp"
#include "mat_file_utils.hpp"

/**
 * @file MatFileDataset.hpp
 * @brief Dataset wrapper that loads input/target matrices from a MATLAB .mat file.
 *
 * Expected .mat layout:
 * - `input_var_name` resolves to a numeric matrix (N x D_in)
 * - `target_var_name` resolves to a numeric matrix (N x D_out)
 *
 * This constructor eagerly loads both matrices into memory and then delegates to
 * `TensorDataset` for slicing/collation.
 */

class MatFileDataset : public TensorDataset
{
   public:
    MatFileDataset(const std::string& mat_path,
        const std::string& input_var_name,
        const std::string& target_var_name)
    {
        auto inputs_mat = matioCpp::utils::load_named_variable_as_matrix(mat_path, input_var_name);
        auto targets_mat =
            matioCpp::utils::load_named_variable_as_matrix(mat_path, target_var_name);

        if (!inputs_mat || !targets_mat)
        {
            throw std::runtime_error("Error loading data from MAT file");
        }

        nn::Tensor inputs = std::move(inputs_mat.value());
        nn::Tensor targets = std::move(targets_mat.value());

        // Check for mismatched sample counts (rows)
        if (inputs.rows() != targets.rows())
        {
            throw std::runtime_error("Mismatched sample counts between inputs and targets.");
        }

        // Initialize base storage via the protected setter to avoid UB
        set_tensors(std::move(inputs), std::move(targets));
    }
};
#endif // NN_MATFILEDATASET_H