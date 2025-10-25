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
#include "layers/Sequential.hpp"

using namespace std;
using namespace matio;

struct DemoConfig {
  string mat_path;
  bool enable_caps = true;
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
    if (v.holds_type<double>() || v.holds_type<float>() ||
        v.holds_type<int32_t>()) {
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
static auto compute_capped_shape(const MatVar& var, const DemoConfig& cfg)
    -> std::pair<int, int> {
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

  if (cfg.enable_caps && declared_elements > cfg.max_elements) {
    size_t capped_cols = std::min<size_t>(
        static_cast<size_t>(std::max(1, cols)), cfg.max_elements);
    size_t capped_rows = std::min<size_t>(
        static_cast<size_t>(std::max(1, rows)), cfg.max_elements / capped_cols);
    capped_cols = std::max<size_t>(1, capped_cols);
    capped_rows = std::max<size_t>(1, capped_rows);
    rows = static_cast<int>(capped_rows);
    cols = static_cast<int>(capped_cols);
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
static void fill_matrix_from_vector(Eigen::MatrixXf& data_mat, MatrixDims dims,
                                    int used_cols, const std::vector<T>& vec) {
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
static auto build_matrix_from_var(const MatVar& var, const DemoConfig& cfg)
    -> Eigen::MatrixXf {
  auto rc = compute_capped_shape(var, cfg);
  int rows = rc.first;
  int cols = rc.second;

  int used_cols = cols;
  if (cfg.enable_caps && cols > cfg.max_features) {
    used_cols = cfg.max_features;
  }

  Eigen::MatrixXf data_mat(rows, used_cols);
  MatrixDims dims{.rows = static_cast<Eigen::Index>(rows),
                  .cols = static_cast<Eigen::Index>(cols)};
  if (var.holds_type<double>()) {
    fill_matrix_from_vector(
        data_mat, dims, used_cols, var.get_vector<double>());
  } else if (var.holds_type<float>()) {
    fill_matrix_from_vector(data_mat, dims, used_cols, var.get_vector<float>());
  } else {
    fill_matrix_from_vector(
        data_mat, dims, used_cols, var.get_vector<int32_t>());
  }

  return data_mat;
}

static auto run_demo(const DemoConfig& cfg) -> int {
  MatFile mat_file;
  if (!mat_file.open(cfg.mat_path)) {
    cerr << "Failed to open MAT file: " << cfg.mat_path << '\n';
    return 1;
  }

  auto var_opt = pick_numeric_var(mat_file);
  if (!var_opt) {
    cerr << "No suitable numeric variable found in MAT file\n";
    return 1;
  }
  const MatVar& var = *var_opt;

  // ------------------------------------------------------------------
  // Data loading walkthrough
  // ------------------------------------------------------------------
  // We have opened the requested MAT file and selected a numeric top-level
  // variable. The MAT variable exposes a typed, flattened storage vector and
  // a `dimensions` vector describing its shape (MAT files are column-major
  // internally). To avoid attempting to allocate very large Eigen matrices
  // for enormous MAT variables, we compute a capped shape via
  // `compute_capped_shape` which respects cfg.max_elements and
  // cfg.max_features. The resulting Eigen matrix will be of type `float` and is
  // safe for use in the tiny model below.

  Eigen::MatrixXf data_mat;
  try {
    data_mat = build_matrix_from_var(var, cfg);
  } catch (const std::bad_alloc& e) {
    cerr << "Allocation failed: " << e.what() << '\n';
    return 1;
  }

  // ------------------------------------------------------------------
  // Dataset & DataLoader construction walkthrough
  // ------------------------------------------------------------------
  // The demo wraps the numeric matrix into our `Tensor` type and constructs a
  // `TensorDataset` that uses the same matrix for inputs and targets (auto-
  // encoding). We then create a `DataLoader` with a modest batch size (16)
  // and deterministic shuffling (seeded) so that runs are reproducible.

  Tensor used_data(data_mat);
  auto dataset = std::make_shared<TensorDataset>(used_data, used_data);
  DataLoader loader(dataset, static_cast<std::size_t>(16), true, 123U);

  // ------------------------------------------------------------------
  // Model construction walkthrough
  // ------------------------------------------------------------------
  // Build a tiny encoder-decoder model. The encoder is a Linear layer that
  // reduces dimensionality to `hidden_dim` (half of input_dim, at least 2).
  // A `Leaky` (LIF-like) nonlinearity sits between encoder and decoder. The
  // decoder projects back to the original input dimensionality. This simple
  // wiring demonstrates forward propagation through the layer stack.

  Eigen::Index input_dim_idx = data_mat.cols();
  int input_dim = static_cast<int>(input_dim_idx);
  int hidden_dim = std::max(2, input_dim / 2);
  auto enc = std::make_shared<Linear>(input_dim, hidden_dim);
  auto lif = std::make_shared<Leaky>();
  auto dec = std::make_shared<Linear>(hidden_dim, input_dim);
  Sequential model({enc, lif, dec});

  cout << "Running forward passes on MAT data variable '" << var.name
       << "' with shape " << data_mat.rows() << "x" << data_mat.cols() << "\n";

  // ------------------------------------------------------------------
  // Batch loop and save routine walkthrough
  // ------------------------------------------------------------------
  // Iterate the loader to obtain minibatches. For each batch we run a
  // forward pass through the model to obtain a reconstruction. The demo
  // prints shapes for visibility. For convenience we save the first batch's
  // reconstruction to a MAT file named `reconstructed.mat`. We also cap the
  // number of processed batches to avoid long runs when using large inputs.

  int batch_idx = 0;
  for (const auto& batch : loader) {
    Tensor input = batch.inputs;
    Tensor recon = model.forward(input);

    cout << "Batch " << batch_idx << ": input shape = (" << input.data.rows()
         << ", " << input.data.cols() << ") recon shape = ("
         << recon.data.rows() << ", " << recon.data.cols() << ")\n";

    if (batch_idx == 0) {
      MatFile out;
      if (out.create("reconstructed.mat")) {
        std::vector<double> flat;
        flat.reserve(recon.data.rows() * recon.data.cols());
        for (int r = 0; r < recon.data.rows(); ++r) {
          for (int c = 0; c < recon.data.cols(); ++c) {
            flat.push_back(static_cast<double>(recon.data(r, c)));
          }
        }
        out.write_double_matrix(
            "reconstruction",
            flat,
            {(int)recon.data.rows(), (int)recon.data.cols()});
        out.close();
        cout << "Saved reconstruction to reconstructed.mat\n";
      }
    }

    ++batch_idx;
    if (batch_idx >= 5) {
      break;
    }
  }

  cout << "Done." << '\n';
  return 0;
}

auto main(int argc, char** argv) -> int {
  DemoConfig cfg;
  if (argc < 2) {
    cerr << "Usage: dataLoader_demo <path-to-mat-file> [--no-caps] "
            "[--max-features=N] [--max-elements=M]\n";
    return 1;
  }

  cfg.mat_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    string arg(argv[i]);
    if (arg == "--no-caps") {
      cfg.enable_caps = false;
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