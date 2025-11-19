#ifndef EXPERIMENT01_UTILS_HPP
#define EXPERIMENT01_UTILS_HPP

#include <Eigen/Dense>
#include <string>

/**
 * @brief Helper function to create a Sequential model.
 * @param list An initializer list of shared pointers to Module objects.
 * @return A unique pointer to the created Sequential model.
 */
void processSubject(const std::string& subjectPath, const std::string& subjectName,
                    const std::string& audioFilePath, const std::string& eegFilePath);

#endif // EXPERIMENT01_UTILS_HPP
