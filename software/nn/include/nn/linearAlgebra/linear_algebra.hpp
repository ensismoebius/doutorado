/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 29 de mar de 2020
 *
 */

/**
 * @file linear_algebra.hpp
 * @brief Small numeric/vector utilities used by preprocessing and analysis code.
 *
 * This namespace contains classic signal-processing helpers (convolution, DCT,
 * normalization) and simple linear algebra utilities for legacy experiments.
 *
 * Most functions operate on `std::vector<double>` or `std::span<double>` and do
 * not depend on the neural network modules.
 */
#ifndef SRC_LIB_LINEARALGEBRA_LINEARALGEBRA_H_
#define SRC_LIB_LINEARALGEBRA_LINEARALGEBRA_H_
#include <span>
#include <vector>

namespace linearAlgebra
{

/**
 * Calculates the derivative of a vector
 * and change the given vector
 * @param vector - A reference to the vector to be derived
 * @param level - Amount of derivatives over the same vector
 * @return resulting vector (optional)
 */
auto derivative(std::vector<double>& vector, long level = 1) -> std::vector<double>;

/**
 * Function that return dot product of two vectors.
 * @param a
 * @param b
 * @return
 */
auto dotProduct(std::span<const double> a, std::span<const double> b) -> double;

/**
 * Given a vector calculates the corresponding orthogonal vector
 * @param vector - source vector
 * @return orthogonal vector
 */
auto calcOrthogonalVector(std::span<const double> vector) -> std::vector<double>;

/**
 * Normalize the vector
 * @param signal
 * @param lowerLimit
 * @param upperLimit
 */
void normalizeVectorToRange(std::span<double> signal, double lowerLimit, double upperLimit);

/**
 * Normalize the vector to sum 1
 * @param signal
 */
void normalizeVectorToSum1(std::span<double> signal);

/**
 * Convolute a signal with a filter (kernel)
 * @param data
 * @param kernel
 * @return true if successful (always true with spans)
 */
auto convolution(std::span<double> data, std::span<const double> kernel) -> bool;

/**
 * Performs a DCT on vector
 * @param vector
 */
void discreteCosineTransform(std::span<double> vector);

/**
 * Scales a given matrix
 * @param matrix
 */
void scaleMatrix(std::vector<std::vector<double>>& matrix);

/**
 * Solves the linear system represented by
 * the matrix and return the results.
 * The matrix MUST be scaled before!!!
 * @see scaleMatrix
 * @param matrix - A scaled matrix
 * @return a vetor with results
 */
auto solveMatrix(const std::vector<std::vector<double>>& matrix) -> std::vector<double>;

/**
 * Normalize the vector to sum 1 and guarantees
 * that all values are positives
 * @param signal
 */
void normalizeVectorToSum1AllPositive(std::span<double> signal);

/**
 * Resizes a vector in a centered way
 * @example std::vector<double> vec{ 1, 2, 3, 4, 5 };
 * resizeCentered(vec, 9);
 * we get: { 0, 0, 1, 2, 3, 4, 5, 0, 0 }
 * resizeCentered(vec, 3);
 * we get: { 2, 3, 4 }
 * @param vector
 * @param newSize
 */
void resizeCentered(std::vector<double>& vector, long newSize, double defaultValue = 0);

/**
 * Normalize feature matrix to a given range (min-max normalization per feature)
 * @param features matrix of features (n_samples x n_features), modified in-place
 * @param range target range [min, max]
 */
void minMaxNormalizeFeatures(std::vector<std::vector<double>>& features,
                             const std::vector<double>& range = {0.0, 1.0});

} // namespace linearAlgebra
#endif /* SRC_LIB_LINEARALGEBRA_LINEARALGEBRA_H_ */
