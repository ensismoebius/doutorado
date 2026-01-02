/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 29 de mar de 2020
 *
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
auto dotProduct(std::vector<double> a, std::vector<double> b) -> double;

/**
 * Create an orthogonal vector given another one
 * @param originalVector
 * @param vectorSize
 * @return
 */
auto calcOrthogonalVector(const double* originalVector, long vectorSize) -> double*;

/**
 * Given a vector calculates the corresponding orthogonal vector
 * @param vector - source vector
 * @return orthogonal vector
 */
auto calcOrthogonalVector(const std::span<const double>& vector) -> std::vector<double>;

/**
 * Normalize the vector
 * @param signal
 * @param signalLength
 * @param lowerLimit
 * @param upperLimit
 */
void normalizeVectorToRange(double* signal, long signalLength, double lowerLimit,
                            double upperLimit);

/**
 * Normalize the vector
 * @param signal
 * @param lowerLimit
 * @param upperLimit
 */
void normalizeVectorToRange(std::vector<double>& signal, double lowerLimit, double upperLimit);

/**
 * Normalize the vector to sum 1
 * @param signal
 * @param signalLength
 */
void normalizeVectorToSum1(double* signal, long signalLength);

/**
 * Normalize the vector to sum 1
 * @param signal
 */
void normalizeVectorToSum1(std::vector<double>& signal);

/**
 * Convolute a signal with a filter (kernel)
 * @param data
 * @param dataLength
 * @param kernel
 * @param kernelSize
 * @return
 */
auto convolution(double* data, int dataLength, double* kernel, long kernelSize) -> bool;

/**
 * Performs a DCT on vector
 * @param vector
 * @param vectorLength
 */
void discreteCosineTransform(double* vector, long vectorLength);

/**
 * Performs a DCT on vector
 * @param vector
 */
void discreteCosineTransform(std::vector<double>& vector);

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
auto solveMatrix(std::vector<std::vector<double>>& matrix) -> std::vector<double>;

/**
 * Normalize the vector to sum 1 and guarantees
 * that all values are positives
 * @param signal
 */
void normalizeVectorToSum1AllPositive(std::vector<double>& signal);

/**
 * Normalize the vector to sum 1 and guarantees
 * that all values are positives
 * @param signal
 * @param signalLength
 */
void normalizeVectorToSum1AllPositive(double* signal, long signalLength);

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
