#include "ThesisPca.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace thesis
{

auto fit_pca(const std::vector<std::vector<float>>& rows, int n_components) -> PcaProjection
{
    if (rows.empty()) throw std::invalid_argument("fit_pca: no rows to fit");
    const auto n = static_cast<Eigen::Index>(rows.size());
    const auto D = static_cast<Eigen::Index>(rows.front().size());
    if (D == 0) throw std::invalid_argument("fit_pca: rows have no values");

    const Eigen::Index max_k = std::min<Eigen::Index>(D, n - 1);
    if (n_components < 1 || n_components > max_k)
        throw std::invalid_argument("fit_pca: n_components = " + std::to_string(n_components) +
                                    " is not in [1, min(D, n - 1)] = [1, " + std::to_string(max_k) +
                                    "] for " + std::to_string(n) + " samples of " +
                                    std::to_string(D) +
                                    " values. Lower feature_extraction.pca.n_components.");

    Eigen::MatrixXd X(n, D);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        const auto& r = rows[static_cast<std::size_t>(i)];
        if (static_cast<Eigen::Index>(r.size()) != D)
            throw std::invalid_argument("fit_pca: row " + std::to_string(i) + " has " +
                                        std::to_string(r.size()) + " values, row 0 has " +
                                        std::to_string(D));
        for (Eigen::Index d = 0; d < D; ++d) X(i, d) = r[static_cast<std::size_t>(d)];
    }

    const Eigen::RowVectorXd mean = X.colwise().mean();
    X.rowwise() -= mean;
    const Eigen::MatrixXd cov = (X.transpose() * X) / static_cast<double>(n - 1);

    // Ascending eigenvalues; the principal axes are the LAST columns.
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("fit_pca: eigendecomposition of the covariance failed");
    const Eigen::VectorXd& eigenvalues = solver.eigenvalues();
    const Eigen::MatrixXd& eigenvectors = solver.eigenvectors();

    // Relative tolerance for "zero variance": eigenvalues of a rank-r covariance computed
    // in double are ~1e-16 x the largest beyond index r, never exactly 0.
    const double largest = eigenvalues(D - 1);
    const double zero_tol = std::max(largest, 0.0) * 1e-10;

    PcaProjection pca;
    pca.mean.assign(mean.data(), mean.data() + D);
    for (int c = 0; c < n_components; ++c)
    {
        const Eigen::Index col = D - 1 - c;
        const double variance = eigenvalues(col);
        if (!(variance > zero_tol))
            throw std::invalid_argument(
                "fit_pca: the data has only " + std::to_string(c) +
                " direction(s) of nonzero variance, fewer than n_components = " +
                std::to_string(n_components) +
                ". The extra axes would be an arbitrary basis of a zero-variance subspace "
                "(constant features whose orientation is floating-point noise). Lower "
                "feature_extraction.pca.n_components.");

        Eigen::VectorXd axis = eigenvectors.col(col);
        // Orient the axis so its largest-magnitude coordinate is positive.
        Eigen::Index arg = 0;
        axis.cwiseAbs().maxCoeff(&arg);
        if (axis(arg) < 0.0) axis = -axis;

        pca.components.emplace_back(axis.data(), axis.data() + D);
        pca.explained_variance.push_back(variance);
    }
    return pca;
}

auto project_pca(const PcaProjection& pca, const std::vector<float>& row) -> std::vector<double>
{
    if (row.size() != pca.mean.size())
        throw std::invalid_argument("project_pca: row has " + std::to_string(row.size()) +
                                    " values, the projection was fitted on " +
                                    std::to_string(pca.mean.size()));
    std::vector<double> out;
    out.reserve(pca.components.size());
    for (const auto& axis : pca.components)
    {
        double dot = 0.0;
        for (std::size_t d = 0; d < row.size(); ++d)
            dot += (static_cast<double>(row[d]) - pca.mean[d]) * axis[d];
        out.push_back(dot);
    }
    return out;
}

} // namespace thesis
