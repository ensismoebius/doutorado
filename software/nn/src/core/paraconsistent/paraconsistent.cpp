/**
 * @file paraconsistent.cpp
 * @brief Paraconsistent logic utilities and normalization helpers.
 */

/*
 * paraconsistent.cpp
 *
 *  Created on: 30 de abr de 2019
 *      Author: ensis
 */

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "nn/linearAlgebra/linear_algebra.hpp"
#include "nn/utility/comparison.h"

/**
 * @brief Calculates certainty degree g1 from alpha and beta.
 */
auto calculate_certainty_degree_g1(double alpha, double beta) -> double
{
    return alpha - beta;
}

/**
 * @brief Calculates contradiction degree g2 from alpha and beta.
 */
auto calculate_contradiction_degree_g2(double alpha, double beta) -> double
{
    return alpha + beta - 1;
}

/**
 * @brief Normalizes each feature subvector from a raw pointer matrix to sum 1.
 */
static void normalizeFeatureVectors(
    double**& featureVectors, unsigned int vectorSize, long subVectorsSize)
{
    for (unsigned int vi = 0; vi < vectorSize; vi++)
    {
        linearAlgebra::normalizeVectorToSum1(
            {featureVectors[vi], static_cast<size_t>(subVectorsSize)});
    }
}

/**
 * @brief Normalizes each feature subvector from a nested vector container to sum 1.
 */
static void normalizeFeatureVectors(
    std::vector<std::vector<double>>& featureVectors, unsigned int vectorSize)
{
    for (unsigned int i = 0; i < vectorSize; i++)
    {
        linearAlgebra::normalizeVectorToSum1(featureVectors[i]);
    }
}

/**
 * @brief Normalizes all class feature vectors from a triple-pointer structure.
 */
void normalize_class_feature_vectors(unsigned int amountOfClasses,
    unsigned int featureVectorsPerClass,
    unsigned int featureVectorSize,
    double*** arrClasses)
{
    for (unsigned int i = 0; i < amountOfClasses; i++)
    {
        normalizeFeatureVectors(arrClasses[i], featureVectorsPerClass, featureVectorSize);
    }
}

/**
 * @brief Normalizes all class feature vectors from nested vectors.
 */
void normalize_class_feature_vectors(unsigned int amountOfClasses,
    unsigned int featureVectorsPerClass,
    unsigned int featureVectorSize,
    std::vector<std::vector<std::vector<double>>>& arrClasses)
{
    for (unsigned int i = 0; i < amountOfClasses; i++)
    {
        normalizeFeatureVectors(arrClasses[i], featureVectorsPerClass);
    }
}

/**
 * @brief Normalizes all class feature vectors from a class-keyed map.
 */
void normalize_class_feature_vectors(unsigned int amountOfClasses,
    unsigned int featureVectorsPerClass,
    unsigned int featureVectorSize,
    std::map<std::string, std::vector<std::vector<double>>>& arrClasses)
{
    for (auto& entry : arrClasses)
    {
        normalizeFeatureVectors(entry.second, featureVectorsPerClass);
    }
}

/**
 * @brief Computes alpha as the minimum average compactness across classes.
 */
auto calculate_alpha(unsigned int amountOfClasses,
    unsigned int featureVectorsPerClass,
    unsigned int featureVectorSize,
    const std::map<std::string, std::vector<std::vector<double>>>& arrClasses) -> double
{
    std::map<std::string, std::vector<double>> arrLargestItems;
    std::map<std::string, std::vector<double>> arrSmallestItems;

    double alpha = std::numeric_limits<double>::max();
    double temp = 0;
    double item;

    // initializes the vectors
    for (const auto& clazz : arrClasses)
    {
        // creates sub vector
        arrLargestItems[clazz.first].resize(featureVectorSize, -std::numeric_limits<double>::max());
        arrSmallestItems[clazz.first].resize(featureVectorSize, std::numeric_limits<double>::max());
    }

    // Calculating the range vectors
    for (const auto& clazz : arrClasses)
    {
        for (unsigned int itemIndex = 0; itemIndex < featureVectorSize; itemIndex++)
        {
            for (unsigned int featureVectorIndex = 0; featureVectorIndex < featureVectorsPerClass;
                featureVectorIndex++)
            {
                item = clazz.second[featureVectorIndex][itemIndex];

                arrLargestItems[clazz.first][itemIndex] =
                    std::max(item, arrLargestItems[clazz.first][itemIndex]);
                arrSmallestItems[clazz.first][itemIndex] =
                    std::min(item, arrSmallestItems[clazz.first][itemIndex]);
            }
        }

        // Finding alpha
        for (unsigned int si = 0; si < featureVectorSize; ++si)
        {
            temp += 1 - (arrLargestItems[clazz.first][si] - arrSmallestItems[clazz.first][si]);
        }

        temp /= featureVectorSize;
        alpha = alpha > temp ? temp : alpha;
        temp = 0;
    }

    return alpha;
}

/**
 * @brief Computes beta as the overlap ratio between classes in feature space.
 */
auto calculate_beta(unsigned int amountOfClasses,
    unsigned int featureVectorsPerClass,
    unsigned int featureVectorSize,
    std::map<std::string, std::vector<std::vector<double>>>& arrClasses) -> double
{
    double item;
    std::map<std::string, std::vector<double>> arrLargestItems;
    std::map<std::string, std::vector<double>> arrSmallestItems;

    // initializes the range vectors
    for (const auto& clazz : arrClasses)
    {
        // creates sub vector
        arrLargestItems[clazz.first].resize(featureVectorSize, -std::numeric_limits<double>::max());
        arrSmallestItems[clazz.first].resize(featureVectorSize, std::numeric_limits<double>::max());
    }

    // Calculating the range vectors
    for (const auto& clazz : arrClasses)
    {
        for (unsigned int itemIndex = 0; itemIndex < featureVectorSize; itemIndex++)
        {
            for (unsigned int featureVectorIndex = 0; featureVectorIndex < featureVectorsPerClass;
                featureVectorIndex++)
            {
                item = clazz.second[featureVectorIndex][itemIndex];

                arrLargestItems[clazz.first][itemIndex] =
                    std::max(item, arrLargestItems[clazz.first][itemIndex]);
                arrSmallestItems[clazz.first][itemIndex] =
                    std::min(item, arrSmallestItems[clazz.first][itemIndex]);
            }
        }
    }

    // Calculating the R factor (amount of times that a feature
    // vector item overlaps an range vector item
    unsigned long int R = 0;

    // comparing all featureVector elements from a class
    // with all range vectors from another classes
    for (const auto& clazz : arrClasses)
    {
        for (unsigned int fvi = 0; fvi < featureVectorsPerClass; fvi++)
        {
            for (const auto& clazz2 : arrClasses)
            {
                // do not compare with the range vector from the same class
                if (clazz2.first == clazz.first)
                {
                    continue;
                }

                for (unsigned int ii = 0; ii < featureVectorSize; ii++)
                {
                    if (inRange(arrClasses[clazz.first][fvi][ii], // value
                            arrSmallestItems[clazz2.first][ii],   // lowerLimit
                            arrLargestItems[clazz2.first][ii])    // upperLimit
                    )
                    {
                        R++;
                    }
                }
            }
        }
    }

    // Return beta overlap ratio.
    return R / (double) (amountOfClasses * (amountOfClasses - 1.0) * featureVectorsPerClass *
                         featureVectorSize);
}
