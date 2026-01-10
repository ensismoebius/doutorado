/*
 * paraconsistent.h
 *
 *  Created on: 30 de abr de 2019
 *      Author: ensis
 */

#ifndef LIB_PARACONSISTENT_H_
#define LIB_PARACONSISTENT_H_

#include <map>
#include <string>
#include <vector>

/**
 * @file paraconsistent.h
 * @brief Paraconsistent logic utilities used by legacy experiments.
 *
 * These functions implement simple calculations around certainty/contradiction
 * degrees (g1/g2) from alpha/beta parameters.
 *
 * Notes:
 * - This module is independent of the NN/SNN layers.
 * - The API uses raw STL containers and doubles to keep dependencies minimal.
 */

/**
 * @brief Given the alfa and the betha calculates the certainty degree
 *
 * @param alpha
 * @param betha
 * @return double
 */
auto calculate_certainty_degree_g1(double alpha, double betha) -> double;

/**
 * @brief Given the alfa and the betha calculates the contradiction degree
 *
 * @param alpha
 * @param betha
 * @return double
 */
auto calculate_contradiction_degree_g2(double alpha, double betha) -> double;

/**
 * @brief Calculates the alpha value
 *
 * @param amountOfClasses Amount of classes
 * @param featureVectorsPerClass Amount of feature vectors per class
 * @param featureVectorSize Size of the feature vectors
 * @param arrClasses Holds all classes
 * @return double Value of alpha
 */
auto calculate_alpha(unsigned int amountOfClasses, unsigned int featureVectorsPerClass,
                     unsigned int featureVectorSize,
                     const std::map<std::string, std::vector<std::vector<double>>>& arrClasses)
    -> double;

/**
 * @brief Calculates the beta value
 *
 * @param amountOfClasses Amount of classes
 * @param featureVectorsPerClass Amount of feature vectors per class
 * @param featureVectorSize Size of the feature vectors
 * @param arrClasses Holds all classes
 * @return double Value of beta
 */
auto calculate_beta(unsigned int amountOfClasses, unsigned int featureVectorsPerClass,
                    unsigned int featureVectorSize,
                    std::map<std::string, std::vector<std::vector<double>>>& arrClasses) -> double;

/**
 * @brief Normalize all feature vectors from all classes
 *
 * @param amountOfClasses Amount of classes
 * @param featureVectorsPerClass Amount of feature vectors per class
 * @param featureVectorSize Size of the feature vectors
 * @param arrClasses Holds all classes
 */
void normalize_class_feature_vectors(
    unsigned int amountOfClasses, unsigned int featureVectorsPerClass,
    unsigned int featureVectorSize,
    std::map<std::string, std::vector<std::vector<double>>>& arrClasses);

#endif /* LIB_PARACONSISTENT_H_ */
