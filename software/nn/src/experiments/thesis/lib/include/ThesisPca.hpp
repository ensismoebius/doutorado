#pragma once

/**
 * @file ThesisPca.hpp
 * @brief Principal component analysis: the linear, training-free extractor of the thesis.
 *
 * WHY. The autoencoders compress each sample to a short latent by training a network. PCA
 * compresses the same input to the same length with no training at all: it keeps the k
 * directions along which the samples vary most. If PCA separates speakers as well as an
 * autoencoder does, the training bought nothing; if it separates them worse, the gap is
 * what the nonlinear model contributes. It is the reference the learned extractors are
 * measured against (Jolliffe & Cadima 2016, doi:10.1098/rsta.2015.0202).
 *
 * WHAT. Rows are samples (here: 256 pooled, min-max normalized values each, exactly the
 * autoencoders' input). Fitting centers the rows by their mean, diagonalizes the D x D
 * covariance and keeps the eigenvectors of the k largest eigenvalues:
 *
 *     rows (n x 256)  --center-->  X  --cov = X^T X / (n-1)-->  eigenvectors v1..v256
 *                                                               sorted by variance
 *     feature of row r = [ (r - mean) . v1, ..., (r - mean) . vk ]       (k values)
 *
 * Two choices that would otherwise make the output arbitrary are fixed here:
 *   - SIGN. v and -v are equally valid eigenvectors; each axis is oriented so that its
 *     largest-magnitude coordinate is positive (deterministic across runs and platforms).
 *   - RANK. If the data has fewer than k directions of nonzero variance, the extra axes
 *     are an arbitrary basis of a zero-variance subspace — every sample projects to ~0 and
 *     which basis comes out depends on floating-point noise. Fitting refuses instead.
 */

#include <vector>

namespace thesis
{

struct PcaProjection
{
    std::vector<double> mean;                    ///< D: per-dimension mean of the fit rows
    std::vector<std::vector<double>> components; ///< k x D: principal axes, largest first
    std::vector<double> explained_variance;      ///< k: eigenvalue of each kept axis
};

/// Fit on `rows` (n rows of equal length D). Throws if the rows are ragged, if
/// n_components is not in [1, min(D, n - 1)], or if the data has fewer than n_components
/// directions of nonzero variance.
auto fit_pca(const std::vector<std::vector<float>>& rows, int n_components) -> PcaProjection;

/// Coordinates of `row` on the fitted axes (k values).
auto project_pca(const PcaProjection& pca, const std::vector<float>& row) -> std::vector<double>;

} // namespace thesis
