/*************************************************************************
 * Spiking autoencoder demo (dataLoader_demo)
 *
 * Purpose:
 *  - Load the first numeric variable from a MATLAB .mat file using MatFile
 *  - Convert it to an Eigen::MatrixXf (with safety caps to avoid huge allocs)
 *  - Wrap the matrix in a Dataset/DataLoader and run forward passes through
 *    a tiny spiking autoencoder (Linear -> Leaky -> Linear)
 *  - Save the reconstruction of the first batch to `reconstructed.mat`
 *
 * Notes:
 *  - Defaults include: max_features = 512 and max_elements = 1,000,000
 *  - Use `--no-caps` to disable the safety limits (only for advanced users)
 *  - The demo is intentionally minimal and demonstrates I/O + model wiring.
 *************************************************************************/

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dataLoaders/DataLoader.h"
#include "dataLoaders/MatFile.h"
#include "dataLoaders/TensorDataset.h"
#include "layers/Leaky.hpp"
#include "layers/MSELoss.hpp"
#include "layers/Sequential.hpp"
#include "optimizers/Adam.hpp"

using namespace std;
using namespace matio;

struct DemoConfig {
    string mat_path;
    bool enable_trims = true;
    int max_features = 512;
    size_t max_elements = 1000000;
};

// pick_numeric_var
// ------------------
// Scan the opened MatFile and return the first top-level variable that
// contains numeric data. Supported numeric element types are double, float,
// and int32. Returns nullopt when no suitable variable is found.
static auto pick_numeric_var(MatFile& mat_file) -> optional<MatVar> {
    auto variables = mat_file.read_all_variables();
    for (auto& p : variables) {
        const MatVar& v = p.second;
        if (v.holds_type<double>() || v.holds_type<float>() || v.holds_type<int32_t>()) {
            return v;
        }
    }
    return nullopt;
}

// compute_capped_shape
// ---------------------
// Determine an (rows, cols) shape for the MatVar that respects the demo's
// element cap (cfg.max_elements). This avoids allocating huge Eigen matrices
// when the MAT variable declares an excessive number of elements.
static auto compute_trimmed_shape(const MatVar& var, const DemoConfig& cfg) -> std::pair<int, int> {
    int rows = 1;
    int cols = 1;
    if (!var.dimensions.empty()) {
        rows = var.dimensions.size() >= 1 ? var.dimensions[0] : 1;
        cols = var.dimensions.size() >= 2 ? var.dimensions[1] : 1;
    }

    size_t declared_elements = 1;
    for (auto d : var.dimensions) {
        declared_elements *= static_cast<size_t>(d);
    }

    if (cfg.enable_trims && declared_elements > cfg.max_elements) {
        size_t trimmed_cols = std::min<size_t>(      // Cap cols to at least 1
            static_cast<size_t>(std::max(1, cols)),  // Cap cols to at least 1
            cfg.max_elements                         // Cap cols to at most max_elements
        );

        size_t trimmed_rows = std::min<size_t>(      // Cap rows to at least 1
            static_cast<size_t>(std::max(1, rows)),  // Cap rows to at least 1
            cfg.max_elements / trimmed_cols          // Ensure we don't exceed element cap
        );

        trimmed_cols = std::max<size_t>(1, trimmed_cols);
        trimmed_rows = std::max<size_t>(1, trimmed_rows);

        rows = static_cast<int>(trimmed_rows);
        cols = static_cast<int>(trimmed_cols);
    }
    return {rows, cols};
}

// fill_matrix_from_vector
// ------------------------
// Copy elements from a typed std::vector<T> into the provided Eigen matrix.
// The source vector is assumed to be a linearized row-major buffer for a
// (rows x cols) array. Only the first `used_cols` columns are copied; if the
// source is shorter than the destination area the remainder is zero-filled.
// Small helper type to hold matrix dimensions so individual integer params
// are not adjacent in function signatures (reduces parameter-swap errors).
struct MatrixDims {
    Eigen::Index rows;
    Eigen::Index cols;
};

template <typename T>
static void fill_matrix_from_vector(Eigen::MatrixXf& data_mat, MatrixDims dims, int used_cols,
                                    const std::vector<T>& vec) {
    for (Eigen::Index r = 0; r < dims.rows; ++r) {
        for (int c = 0; c < used_cols; ++c) {
            int idx = static_cast<int>(r * dims.cols) + c;
            if (idx < static_cast<int>(vec.size())) {
                data_mat(static_cast<int>(r), c) = static_cast<float>(vec[idx]);
            } else {
                data_mat(static_cast<int>(r), c) = 0.0F;
            }
        }
    }
}

// build_matrix_from_var
// ---------------------
// Convert a MatVar into an Eigen::MatrixXf using the configured caps. The
// function returns a matrix of floats with shape (rows x used_cols).
static auto build_matrix_from_var(const MatVar& mat_var, const DemoConfig& cfg) -> Eigen::MatrixXf {
    // Determine trimmed shape based on config
    auto trimmed_shape = compute_trimmed_shape(mat_var, cfg);
    int num_rows = trimmed_shape.first;
    int num_cols = trimmed_shape.second;

    // Determine the number of used columns based on the configuration
    int num_used_cols =
        (cfg.enable_trims && num_cols > cfg.max_features) ? cfg.max_features : num_cols;

    // Allocate the Eigen matrix with the trimmed shape
    Eigen::MatrixXf data_matrix(num_rows, num_used_cols);

    // Define the matrix dimensions struct
    MatrixDims matrix_dims{
        .rows = static_cast<Eigen::Index>(num_rows),  //
        .cols = static_cast<Eigen::Index>(num_cols)   //
    };

    // Fill the matrix with data from the MatVar
    if (mat_var.holds_type<double>()) {
        fill_matrix_from_vector(
            data_matrix, matrix_dims, num_used_cols, mat_var.get_vector<double>());
    } else if (mat_var.holds_type<float>()) {
        fill_matrix_from_vector(
            data_matrix, matrix_dims, num_used_cols, mat_var.get_vector<float>());
    } else {
        fill_matrix_from_vector(
            data_matrix, matrix_dims, num_used_cols, mat_var.get_vector<int32_t>());
    }

    return data_matrix;
}

static auto run_demo(const DemoConfig& cfg) -> int {
    // The original demo implementation is intentionally commented out here.
    // To re-enable it, replace this `#if 0` with `#if 1` or remove the
    // preprocessor guards. The commented block below contains the full demo
    // which performs: MAT loading, matrix construction, dataset creation,
    // model construction, batch iteration, and saving the first reconstruction.

    MatFile mat_file;

    if (!mat_file.open(cfg.mat_path)) {
        cerr << "Failed to open MAT file: " << cfg.mat_path << '\n';
        return 1;
    }

    auto mat_var_opt = pick_numeric_var(mat_file);
    if (!mat_var_opt) {
        cerr << "No suitable numeric variable found in MAT file\n";
        return 1;
    }
    // Get the selected variable and convert from optional<MatVar> to MatVar
    const MatVar& mat_var = mat_var_opt.value();

    // Data loading and shaping
    Eigen::MatrixXf data_matrix;
    try {
        data_matrix = build_matrix_from_var(mat_var, cfg);
    } catch (const std::bad_alloc& e) {
        cerr << "Allocation failed: " << e.what() << '\n';
        return 1;
    }

    // Dataset and loader
    Tensor tensor_data(data_matrix);
    auto dataset = std::make_shared<TensorDataset>(tensor_data, tensor_data);
    DataLoader data_loader(dataset, static_cast<std::size_t>(16), true, 123U);

    // Model

    // Determine input dimension from data matrix
    const Eigen::Index input_dimension_index = data_matrix.cols();

    // Determine hidden dimension
    const int input_dimension = static_cast<int>(input_dimension_index);
    const int hidden_dimension = std::max(2, input_dimension / 2);

    // Encoder -> Leaky -> Decoder
    auto encoder = std::make_shared<Linear>(input_dimension, hidden_dimension);
    auto leaky_layer = std::make_shared<Leaky>();
    auto decoder = std::make_shared<Linear>(hidden_dimension, input_dimension);
    Sequential model({encoder, leaky_layer, decoder});
    // Prepare loss and optimizer
    auto mse_loss = std::make_shared<MSELoss>();
    std::vector<Tensor*> parameters = model.params();
    Adam optimizer(0.01F);
    optimizer.attach(parameters);

    // Training loop parameters
    const float target_loss = 1e-4F;
    const int max_epochs = 200000;
    int epoch = 0;
    bool converged = false;

    while (epoch < max_epochs && !converged) {
        float epoch_loss_sum = 0.0F;
        int batch_count = 0;

        for (const auto& batch_item : data_loader) {
            Tensor input = batch_item.inputs;

            // Zero gradients on parameters
            optimizer.zero_grad(parameters);

            // Forward
            Tensor prediction = model.forward(input);

            // Compute loss (autoencoder: target == input)
            mse_loss->set_target(input);
            Tensor loss_tensor = mse_loss->forward(prediction);
            float batch_loss = loss_tensor.data(0, 0);
            epoch_loss_sum += batch_loss;
            ++batch_count;

            // Backward: gradient w.r.t. prediction then propagate through model
            Tensor grad_pred = mse_loss->backward(prediction);
            model.backward(grad_pred);

            // Step optimizer
            optimizer.step(parameters);
        }

        if (batch_count > 0) {
            float epoch_loss_avg = epoch_loss_sum / static_cast<float>(batch_count);
            cout << "Epoch " << epoch << " avg loss = " << epoch_loss_avg << "\n";
            if (epoch_loss_avg <= target_loss) {
                converged = true;
                cout << "Converged (avg loss <= " << target_loss << ")\n";
                break;
            }
        }

        ++epoch;
    }

    // After training, save reconstruction of first batch (if available)
    int saved = 0;
    for (const auto& batch_item : data_loader) {
        Tensor input = batch_item.inputs;
        Tensor recon = model.forward(input);

        MatFile out_file;
        if (out_file.create("reconstructed.mat")) {
            std::vector<double> flat_vector;
            flat_vector.reserve(recon.data.rows() * recon.data.cols());
            for (int r = 0; r < recon.data.rows(); ++r) {
                for (int c = 0; c < recon.data.cols(); ++c) {
                    flat_vector.push_back(static_cast<double>(recon.data(r, c)));
                }
            }
            out_file.write_double_matrix(
                "reconstruction", flat_vector, {(int)recon.data.rows(), (int)recon.data.cols()});
            out_file.close();
            cout << "Saved reconstruction to reconstructed.mat\n";
        }
        ++saved;
        if (saved >= 1) {
            break;
        }
    }

    cout << "Done training." << '\n';
    return 0;
}

auto main(int argc, char** argv) -> int {
    DemoConfig cfg;
    if (argc < 2) {
        // No MAT file path provided. Create a small default MAT file in /tmp
        // so the demo can run without external parameters.
        const std::string default_mat = "/tmp/dataLoader_demo_default.mat";
        MatFile default_writer;
        if (default_writer.create(default_mat)) {
            std::vector<double> demo_data = {1, 2, 3, 4, 5, 6};
            // write as 2x3 to match earlier examples
            default_writer.write_double_matrix("sample", demo_data, {2, 3});
            default_writer.close();
            cout << "No MAT path provided; created default sample at: " << default_mat << "\n";
            cfg.mat_path = default_mat;
        } else {
            cerr << "Failed to create default MAT file at " << default_mat << "\n";
            return 1;
        }
    } else {
        cfg.mat_path = argv[1];
    }
    for (int i = 2; i < argc; ++i) {
        string arg(argv[i]);
        if (arg == "--no-caps") {
            cfg.enable_trims = false;
        }
        if (arg.starts_with("--max-features=")) {
            string val = arg.substr(string("--max-features=").size());
            cfg.max_features = stoi(val);
        }
        if (arg.starts_with("--max-elements=")) {
            string val = arg.substr(string("--max-elements=").size());
            cfg.max_elements = static_cast<size_t>(stoll(val));
        }
    }

    return run_demo(cfg);
}