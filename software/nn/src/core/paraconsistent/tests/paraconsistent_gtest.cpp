#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <map>
#include <vector>

#include "nn/paraconsistent/paraconsistent.h"

unsigned int amountOfClasses;
unsigned int featureVectorSize;
unsigned int featureVectorsPerClass;
std::map<std::string, std::vector<std::vector<double>>> arrClasses;

void initializeClasses(std::map<std::string, std::vector<std::vector<double>>>& _arrClasses,
                       unsigned int& _amountOfClasses, unsigned int& _featureVectorsPerClass,
                       unsigned int& _featureVectorSize)
{
    // Setting up the quantities
    _amountOfClasses = 3;
    _featureVectorSize = 2;
    _featureVectorsPerClass = 4;
    std::vector<std::string> classNames = {"c1", "c2", "c3"};

    // Initializing the "lists"
    for (unsigned int classIndex = 0; classIndex < _amountOfClasses; ++classIndex)
    {
        std::vector<std::vector<double>> featureVectorSet(_featureVectorsPerClass);

        for (unsigned int featureVectorIndex = 0; featureVectorIndex < _featureVectorsPerClass;
             ++featureVectorIndex)
        {
            std::vector<double> featureVector(_featureVectorSize);

            featureVectorSet.push_back(featureVector);
        }

        _arrClasses[classNames[classIndex]] = featureVectorSet;
    }

    // Populating
    _arrClasses[classNames[0]][0] = {0.90, 0.12};
    _arrClasses[classNames[0]][1] = {0.88, 0.14};
    _arrClasses[classNames[0]][2] = {0.88, 0.13};
    _arrClasses[classNames[0]][3] = {0.89, 0.11};

    _arrClasses[classNames[1]][0] = {0.55, 0.53};
    _arrClasses[classNames[1]][1] = {0.53, 0.55};
    _arrClasses[classNames[1]][2] = {0.54, 0.54};
    _arrClasses[classNames[1]][3] = {0.56, 0.54};

    _arrClasses[classNames[2]][0] = {0.10, 0.88};
    _arrClasses[classNames[2]][1] = {0.11, 0.86};
    _arrClasses[classNames[2]][2] = {0.12, 0.87};
    _arrClasses[classNames[2]][3] = {0.11, 0.88};
}

TEST(paraconsistentTest, alpha)
{
    initializeClasses(arrClasses, amountOfClasses, featureVectorsPerClass, featureVectorSize);
    double alpha =
        calculate_alpha(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);
    ASSERT_NEAR(alpha, 0.9749999, 0.000001);
}

TEST(paraconsistentTest, betha)
{
    initializeClasses(arrClasses, amountOfClasses, featureVectorsPerClass, featureVectorSize);
    double betha =
        calculate_beta(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);
    ASSERT_EQ(betha, 0);
}

TEST(paraconsistentTest, distanceTo1_0)
{
    initializeClasses(arrClasses, amountOfClasses, featureVectorsPerClass, featureVectorSize);
    double alpha =
        calculate_alpha(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);
    double betha =
        calculate_beta(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);

    double certaintyDegree_G1 = calculate_certainty_degree_g1(alpha, betha);
    double contradictionDegree_G2 = calculate_contradiction_degree_g2(alpha, betha);
    double distanceTo1_0 =
        std::sqrt(std::pow(certaintyDegree_G1 - 1, 2) + std::pow(contradictionDegree_G2, 2));

    ASSERT_NEAR(distanceTo1_0, 0.035, 0.001);
}

TEST(paraconsistentTest, certaintyDegree_G1)
{
    initializeClasses(arrClasses, amountOfClasses, featureVectorsPerClass, featureVectorSize);
    double alpha =
        calculate_alpha(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);
    double betha =
        calculate_beta(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);
    double certaintyDegree_G1 = calculate_certainty_degree_g1(alpha, betha);

    ASSERT_NEAR(certaintyDegree_G1, 0.975, 0.0001);
}

TEST(paraconsistentTest, contradictionDegree_G2)
{
    initializeClasses(arrClasses, amountOfClasses, featureVectorsPerClass, featureVectorSize);
    double alpha =
        calculate_alpha(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);
    double betha =
        calculate_beta(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);
    double contradictionDegree_G2 = calculate_contradiction_degree_g2(alpha, betha);

    ASSERT_NEAR(contradictionDegree_G2, -0.025, 0.0001);
}